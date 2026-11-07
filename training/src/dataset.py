"""
dataset.py
----------
PyTorch Dataset for the Live-vs-Spoof liveness classifier.

Expected folder layout (produced by src/prepare_celeba_subset.py and
src/prepare_casia_fasd.py, or hand-built the same way):

    <root>/
        live/   *.jpg | *.png   -> label 0 (LABEL_LIVE)
        spoof/  *.jpg | *.png   -> label 1 (LABEL_SPOOF)

We keep this flat "live/spoof" convention (instead of torchvision's
generic ImageFolder alphabetical-class-index behaviour) so the label
mapping is explicit and guaranteed to match config.LABEL_LIVE /
config.LABEL_SPOOF regardless of folder name sorting.
"""

import os
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, LABEL_LIVE, LABEL_SPOOF

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class LivenessDataset(Dataset):
    """Loads (image, label) pairs from a root/{live,spoof} directory tree."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        self.samples = []  # list of (filepath, label)

        for class_name, label in (("live", LABEL_LIVE), ("spoof", LABEL_SPOOF)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(VALID_EXTENSIONS):
                    self.samples.append((os.path.join(class_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under '{root}'. Expected subfolders 'live/' and "
                f"'spoof/' containing images. Run prepare_celeba_subset.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict:
        live = sum(1 for _, l in self.samples if l == LABEL_LIVE)
        spoof = sum(1 for _, l in self.samples if l == LABEL_SPOOF)
        return {"live": live, "spoof": spoof}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Augmentations chosen specifically for anti-spoofing:
    - ColorJitter (brightness/contrast/saturation) mimics different lighting
      and camera sensors seen across live and replay-attack captures.
    - GaussianBlur mimics out-of-focus phone/printed photo attacks and
      generally improves robustness to webcam blur.
    - Small rotation + horizontal flip for pose variation.
    - RandomErasing mimics occlusion (relevant since faces may be masked).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])


def get_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic preprocessing used for validation, test, and real-time inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "../data/celeba_spoof/train"
    ds = LivenessDataset(root, transform=get_eval_transform())
    print(f"Loaded {len(ds)} samples from {root}")
    print(f"Class counts: {ds.class_counts()}")
    img, label = ds[0]
    print(f"Sample tensor shape: {img.shape}, label: {label}")
