"""Training utilities for semi-supervised learning with pseudo-labeling."""

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device):
    """Train model for one epoch with mixed precision.

    Args:
        model: PyTorch model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision training.
        device: Device to train on.

    Returns:
        Tuple of (average loss, accuracy percentage).
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def generate_pseudo_labels(model1, model2, dataloader, device, confidence_threshold=0.8):
    """Generate pseudo labels using dual model ensemble.

    Both models predict on unlabeled data, and their probabilities are averaged.
    Only samples with confidence above threshold are assigned pseudo labels.

    Args:
        model1: First model (e.g., Swin Transformer).
        model2: Second model (e.g., ViT).
        dataloader: DataLoader for unlabeled data.
        device: Device to run inference on.
        confidence_threshold: Minimum confidence to assign pseudo label.

    Returns:
        List of dicts with 'image' and 'id' keys for high-confidence predictions.
    """
    model1.eval()
    model2.eval()
    pseudo_labels = []

    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Generating Pseudo-Labels"):
            images = images.to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)

            # Average probabilities from both models
            probabilities = (nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2)) / 2
            confidences, predicted = probabilities.max(1)

            for filename, pred, conf in zip(filenames, predicted.cpu().numpy(), confidences.cpu().numpy()):
                if conf >= confidence_threshold:
                    pseudo_labels.append({"image": filename, "id": pred})

    return pseudo_labels


def merge_datasets(labeled_dataset, pseudo_labels, unlabeled_image_dir, transform):
    """Merge labeled dataset with pseudo-labeled data.

    Args:
        labeled_dataset: Original labeled dataset.
        pseudo_labels: List of pseudo label dicts.
        unlabeled_image_dir: Path to unlabeled images.
        transform: Transforms to apply.

    Returns:
        ConcatDataset combining labeled and pseudo-labeled data.
    """
    from .dataset import CustomDataset

    pseudo_data = pd.DataFrame(pseudo_labels)
    pseudo_dataset = CustomDataset(image_dir=unlabeled_image_dir, transform=transform)
    pseudo_dataset.data = pseudo_data
    return ConcatDataset([labeled_dataset, pseudo_dataset])


def evaluate_and_save(model1, model2, dataloader, device, output_file):
    """Evaluate using dual model ensemble and save predictions.

    Args:
        model1: First model.
        model2: Second model.
        dataloader: Test data loader.
        device: Device to run inference on.
        output_file: Path to save predictions CSV.
    """
    model1.eval()
    model2.eval()
    predictions = []

    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)

            # Ensemble by averaging probabilities
            probabilities = (nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2)) / 2
            _, predicted = probabilities.max(1)

            for filename, pred in zip(filenames, predicted.cpu().numpy()):
                predictions.append({"image": filename, "id": pred})

    pd.DataFrame(predictions).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
