"""
liveness/transform.py
----------------------
The eval-time preprocessing transform, copied from
anti_spoof_attendance/src/dataset.py's get_eval_transform() (training-only
augmentations and the LivenessDataset class are not needed here -- this
subpackage only does inference). Values (IMAGENET_MEAN/STD, Resize order)
are unchanged from the original.
"""

from torchvision import transforms

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic preprocessing used for real-time inference -- must match
    training exactly: Resize -> ToTensor -> Normalize(ImageNet mean/std)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
