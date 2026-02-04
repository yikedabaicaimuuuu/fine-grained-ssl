"""Custom dataset classes for semi-supervised image classification."""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Dataset class for loading images with optional labels.

    Args:
        image_dir: Path to directory containing images.
        label_file: Optional path to CSV file with 'image' and 'id' columns.
        transform: Optional torchvision transforms to apply.
    """

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
            import torch
            label = self.data['id'].iloc[idx]
            return image, torch.tensor(label, dtype=torch.long)

        return image, self.data['image'].iloc[idx]
