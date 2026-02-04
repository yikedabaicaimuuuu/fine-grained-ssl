"""Data augmentation and preprocessing transforms."""

from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size=256):
    """Get training transforms with data augmentation.

    Args:
        image_size: Target image size after cropping.

    Returns:
        Composed transforms for training.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform(image_size=256):
    """Get test/validation transforms without augmentation.

    Args:
        image_size: Target image size.

    Returns:
        Composed transforms for evaluation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
