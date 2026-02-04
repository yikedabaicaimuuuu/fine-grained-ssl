"""Model definitions for dual-model co-teaching."""

import torch
import torch.nn as nn
from torchvision.models import swin_b, vit_b_16


def create_swin_model(num_classes, pretrained=True, device="cuda"):
    """Create Swin Transformer Base model.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        device: Device to place model on.

    Returns:
        Swin Transformer model with modified classification head.
    """
    model = swin_b(pretrained=pretrained).to(device)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    model = nn.DataParallel(model).to(device)
    return model


def create_vit_model(num_classes, pretrained=True, device="cuda"):
    """Create Vision Transformer (ViT-B/16) model.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        device: Device to place model on.

    Returns:
        ViT model with modified classification head.
    """
    model = vit_b_16(pretrained=pretrained).to(device)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    model = nn.DataParallel(model).to(device)
    return model
