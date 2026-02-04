import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据路径
train_image_dir = "/data-2u-1/tuhq/make_llava_xl_data/hw_clip_labels/train/labeled"
train_label_file = "/data-2u-1/tuhq/make_llava_xl_data/hw_clip_labels/train_labeled.csv"
unlabeled_image_dir = "/data-2u-1/tuhq/make_llava_xl_data/hw_clip_labels/train/unlabeled"
test_image_dir = "/data-2u-1/tuhq/make_llava_xl_data/hw_clip_labels/test"
output_csv = "test_predictions_with_dual_models.csv"

# 数据集定义
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        if label_file:
            self.data = pd.read_csv(label_file)
        else:
            self.data = pd.DataFrame(os.listdir(image_dir), columns=["image"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data['image'].iloc[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if "id" in self.data.columns:
            label = self.data['id'].iloc[idx]
            return image, torch.tensor(label, dtype=torch.long)
        
        return image, self.data['image'].iloc[idx]

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据加载
train_dataset = CustomDataset(image_dir=train_image_dir, label_file=train_label_file, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

unlabeled_dataset = CustomDataset(image_dir=unlabeled_image_dir, transform=train_transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

test_dataset = CustomDataset(image_dir=test_image_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

from torchvision.models import swin_b, vit_b_16

# Model 1: Swin Transformer (Base)
model1 = swin_b(pretrained=True).to(device)
num_features1 = model1.head.in_features
model1.head = nn.Linear(num_features1, 135)  # 替换分类头
model1 = nn.DataParallel(model1).to(device)

# Model 2: Vision Transformer (ViT-B/16)
model2 = vit_b_16(pretrained=True).to(device)
num_features2 = model2.heads.head.in_features
model2.heads.head = nn.Linear(num_features2, 135)  # 替换分类头
model2 = nn.DataParallel(model2).to(device)

# 损失函数与优化器
unique_classes = train_dataset.data['id'].unique()
class_weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=train_dataset.data['id']
)

full_class_weights = torch.zeros(135, dtype=torch.float)
for cls, weight in zip(unique_classes, class_weights):
    full_class_weights[cls] = weight

criterion = nn.CrossEntropyLoss(weight=full_class_weights.to(device), label_smoothing=0.1)
optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=2)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=10, T_mult=2)

scaler1 = GradScaler()
scaler2 = GradScaler()

# 训练函数
def train_model_amp(model, dataloader, criterion, optimizer, scheduler, scaler, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
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
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100.*correct/total:.2f}%")

# 伪标签生成（双模型协同生成）
def generate_pseudo_labels_dual_models(model1, model2, dataloader, device, confidence_threshold=0.8):
    model1.eval()
    model2.eval()
    pseudo_labels = []
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Generating Pseudo-Labels"):
            images = images.to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)
            
            # 取两个模型的平均概率
            probabilities = (nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2)) / 2
            confidences, predicted = probabilities.max(1)
            
            for filename, pred, conf in zip(filenames, predicted.cpu().numpy(), confidences.cpu().numpy()):
                if conf >= confidence_threshold:
                    pseudo_labels.append({"image": filename, "id": pred})
    return pseudo_labels

# 数据集合并
def merge_datasets(labeled_dataset, pseudo_labels, transform):
    pseudo_data = pd.DataFrame(pseudo_labels)
    pseudo_dataset = CustomDataset(
        image_dir=unlabeled_image_dir,
        transform=transform
    )
    pseudo_dataset.data = pseudo_data
    return ConcatDataset([labeled_dataset, pseudo_dataset])

# 测试评估（融合逻辑）
def evaluate_and_save_predictions_dual_models(model1, model2, dataloader, device, output_file):
    model1.eval()
    model2.eval()
    predictions = []
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # 获取两个模型的输出
            outputs1 = model1(images)
            outputs2 = model2(images)
            
            # 融合概率
            probabilities = (nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2)) / 2
            _, predicted = probabilities.max(1)  # 取融合后概率的最大值
            
            for filename, pred in zip(filenames, predicted.cpu().numpy()):
                predictions.append({"image": filename, "id": pred})
    
    # 保存融合后的预测结果
    pd.DataFrame(predictions).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# 动态伪标签训练
for epoch in range(10):
    print(f"Training epoch {epoch+1} with labeled data...")
    train_model_amp(model1, train_loader, criterion, optimizer1, scheduler1, scaler1, device, epochs=1)
    train_model_amp(model2, train_loader, criterion, optimizer2, scheduler2, scaler2, device, epochs=1)
    
    confidence_threshold = max(0.5, 0.8 - epoch * 0.02)
    pseudo_labels = generate_pseudo_labels_dual_models(model1, model2, unlabeled_loader, device, confidence_threshold)
    merged_dataset = merge_datasets(train_dataset, pseudo_labels, train_transform)
    merged_loader = DataLoader(merged_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    
    print(f"Training epoch {epoch+1} with merged data...")
    train_model_amp(model1, merged_loader, criterion, optimizer1, scheduler1, scaler1, device, epochs=1)
    train_model_amp(model2, merged_loader, criterion, optimizer2, scheduler2, scaler2, device, epochs=1)

# 测试评估（融合模型预测）
final_output_file = "predict.csv"
evaluate_and_save_predictions_dual_models(model1, model2, test_loader, device, final_output_file)

