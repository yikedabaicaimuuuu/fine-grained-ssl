"""
Main training script for semi-supervised fine-grained image classification.

This script implements model co-teaching with Swin Transformer and ViT,
using dynamic pseudo-labeling for semi-supervised learning.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.utils.class_weight import compute_class_weight

from src import (
    CustomDataset,
    get_train_transform,
    get_test_transform,
    create_swin_model,
    create_vit_model,
    train_one_epoch,
    generate_pseudo_labels,
    merge_datasets,
    evaluate_and_save,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Semi-supervised Fine-grained Classification")
    parser.add_argument("--train-labeled-dir", type=str, required=True,
                        help="Path to labeled training images")
    parser.add_argument("--train-label-file", type=str, required=True,
                        help="Path to CSV file with image labels")
    parser.add_argument("--unlabeled-dir", type=str, required=True,
                        help="Path to unlabeled training images")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to test images")
    parser.add_argument("--output-file", type=str, default="predictions.csv",
                        help="Output CSV file for predictions")
    parser.add_argument("--num-classes", type=int, default=135,
                        help="Number of classification classes")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--initial-threshold", type=float, default=0.8,
                        help="Initial confidence threshold for pseudo-labeling")
    parser.add_argument("--threshold-decay", type=float, default=0.02,
                        help="Threshold decay per epoch")
    parser.add_argument("--min-threshold", type=float, default=0.5,
                        help="Minimum confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data transforms
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    # Datasets
    train_dataset = CustomDataset(
        image_dir=args.train_labeled_dir,
        label_file=args.train_label_file,
        transform=train_transform
    )
    unlabeled_dataset = CustomDataset(
        image_dir=args.unlabeled_dir,
        transform=train_transform
    )
    test_dataset = CustomDataset(
        image_dir=args.test_dir,
        transform=test_transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Models
    print("Creating models...")
    model1 = create_swin_model(args.num_classes, pretrained=True, device=device)
    model2 = create_vit_model(args.num_classes, pretrained=True, device=device)

    # Class weights for imbalanced data
    unique_classes = train_dataset.data['id'].unique()
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=train_dataset.data['id']
    )
    full_class_weights = torch.zeros(args.num_classes, dtype=torch.float)
    for cls, weight in zip(unique_classes, class_weights):
        full_class_weights[cls] = weight

    # Loss, optimizers, schedulers
    criterion = nn.CrossEntropyLoss(
        weight=full_class_weights.to(device),
        label_smoothing=0.1
    )
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=2)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=10, T_mult=2)
    scaler1 = GradScaler()
    scaler2 = GradScaler()

    # Training loop with dynamic pseudo-labeling
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        # Phase 1: Train on labeled data only
        print("\nPhase 1: Training on labeled data...")
        loss1, acc1 = train_one_epoch(model1, train_loader, criterion, optimizer1, scheduler1, scaler1, device)
        loss2, acc2 = train_one_epoch(model2, train_loader, criterion, optimizer2, scheduler2, scaler2, device)
        print(f"Swin - Loss: {loss1:.4f}, Acc: {acc1:.2f}%")
        print(f"ViT  - Loss: {loss2:.4f}, Acc: {acc2:.2f}%")

        # Generate pseudo labels with dynamic threshold
        confidence_threshold = max(args.min_threshold, args.initial_threshold - epoch * args.threshold_decay)
        print(f"\nGenerating pseudo-labels (threshold: {confidence_threshold:.2f})...")
        pseudo_labels = generate_pseudo_labels(
            model1, model2, unlabeled_loader, device, confidence_threshold
        )
        print(f"Generated {len(pseudo_labels)} pseudo-labels")

        # Merge datasets
        merged_dataset = merge_datasets(
            train_dataset, pseudo_labels, args.unlabeled_dir, train_transform
        )
        merged_loader = DataLoader(
            merged_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        )

        # Phase 2: Train on merged data
        print("\nPhase 2: Training on merged data...")
        loss1, acc1 = train_one_epoch(model1, merged_loader, criterion, optimizer1, scheduler1, scaler1, device)
        loss2, acc2 = train_one_epoch(model2, merged_loader, criterion, optimizer2, scheduler2, scaler2, device)
        print(f"Swin - Loss: {loss1:.4f}, Acc: {acc1:.2f}%")
        print(f"ViT  - Loss: {loss2:.4f}, Acc: {acc2:.2f}%")

    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    evaluate_and_save(model1, model2, test_loader, device, args.output_file)


if __name__ == "__main__":
    main()
