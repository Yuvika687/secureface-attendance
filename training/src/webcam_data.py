"""
Shared helpers for webcam domain adaptation (finetune_webcam.py, evaluate_webcam.py).

Webcam crops come from Admin -> Test Mode -> Collection mode: data/domain_adaptation/
manifest.csv plus session_<n>/<label>/*.png, each PNG being exactly the crop_face()
output the live model received (before the 224x224 resize). Preprocessing here is the
app's own: liveness.transform.get_eval_transform() for evaluation, and the training
project's train-time augmentation (copied from training/src/dataset.py) for fine-tuning.

Labels: bona fide (real face, masked or not) -> 0 (LIVE), attack -> 1 (SPOOF), the same
convention as training/src/config.py (LABEL_LIVE = 0, LABEL_SPOOF = 1).
"""
import csv
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from liveness.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from liveness.transform import get_eval_transform  # noqa: E402

# WEBCAM_COLLECTION_DIR only exists for a pipeline dry run on a throwaway folder.
COLLECTION_DIR = os.environ.get("WEBCAM_COLLECTION_DIR") or os.path.join(REPO, "data", "domain_adaptation")
MANIFEST = os.path.join(COLLECTION_DIR, "manifest.csv")
CELEBA_CROPPED = os.path.expanduser("~/Documents/project/anti_spoof_attendance/data/celeba_spoof_cropped")
V2_CHECKPOINT = os.path.join(REPO, "liveness", "weights", "best_liveness.pth")
V3_DIR = os.path.join(REPO, "liveness", "weights", "v3_webcam")
V3_CHECKPOINT = os.path.join(V3_DIR, "best_liveness_v3_webcam.pth")
GRAY_ZONE_V2 = os.path.join(REPO, "liveness", "weights", "gray_zone_v2.json")


def get_train_transform(image_size: int = IMAGE_SIZE):
    """Identical to training/src/dataset.py get_train_transform() (used to train v2)."""
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


def read_manifest(path=MANIFEST):
    """Rows of the collection manifest whose crop file exists."""
    if not os.path.isfile(path):
        raise SystemExit(f"No collection manifest at {path}. Record data in Test Mode -> Collection mode first.")
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            full = os.path.join(COLLECTION_DIR, r["crop_path"])
            if os.path.isfile(full):
                rows.append({**r, "path": full, "y": 0 if r["bona_fide"] == "1" else 1})
    return rows


def dataset_items(split):
    """(path, y) for data/celeba_spoof_cropped/<split>/{live,spoof} -- the crops v2 was trained/tested on."""
    items = []
    for name, y in (("live", 0), ("spoof", 1)):
        d = os.path.join(CELEBA_CROPPED, split, name)
        items += [(os.path.join(d, f), y) for f in sorted(os.listdir(d))]
    return items


class ImageList(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), torch.tensor(float(y))


@torch.no_grad()
def predict(model, items, device, batch_size=64):
    """P(spoof) for each (path, y) item, with the app's eval preprocessing."""
    model.eval()
    loader = torch.utils.data.DataLoader(ImageList(items, get_eval_transform()), batch_size=batch_size)
    out = []
    for x, _ in loader:
        out += torch.sigmoid(model(x.to(device)).view(-1)).cpu().tolist()
    return out


def metrics(probs, ys, band=None, threshold=0.5):
    """APCER (attacks judged LIVE), BPCER (real judged SPOOF), HTER, accuracy at the threshold.
    With band=(low, high), probabilities inside the band count as UNCERTAIN and are excluded
    from APCER/BPCER (reported separately), exactly like the scanner's three-way decision."""
    n_att = n_bona = att_live = bona_spoof = unc_att = unc_bona = correct = 0
    for p, y in zip(probs, ys):
        unsure = band is not None and band[0] <= p <= band[1]
        spoof = p >= threshold
        if y == 1:
            n_att += 1
            if unsure: unc_att += 1
            elif not spoof: att_live += 1
        else:
            n_bona += 1
            if unsure: unc_bona += 1
            elif spoof: bona_spoof += 1
        correct += int(spoof == (y == 1))
    apcer = att_live / n_att if n_att else None
    bpcer = bona_spoof / n_bona if n_bona else None
    out = {"n": n_att + n_bona, "n_attack": n_att, "n_bona_fide": n_bona, "apcer": apcer, "bpcer": bpcer,
           "hter": (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None}
    if band is None:
        out["accuracy"] = correct / len(ys) if ys else None
    else:
        out.update(uncertain_attack=unc_att, uncertain_bona_fide=unc_bona)
    return out
