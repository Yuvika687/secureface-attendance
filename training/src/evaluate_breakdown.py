"""
evaluate_breakdown.py
-----------------------
Breaks the real test-set evaluation down by CelebA-Spoof's own rich
annotations: spoof type (idx 40), illumination condition (idx 41), and
environment (idx 42) -- using the dataset's official code->name mapping,
verbatim-confirmed from the official repo's README.md
(https://github.com/ZhangYuanhan-AI/CelebA-Spoof, "Rich Annotations"
section, fetched and grepped directly from the raw README source):

                        0      1        2        3      4          5                 6             7     8     9       10
    Spoof Type          Live   Photo    Poster   A4     Face Mask  Upper Body Mask    Region Mask   PC    Pad   Phone   3D Mask
    Illumination Cond.  Live   Normal   Strong   Back   Dark
    Environment         Live   Indoor   Outdoor

IMPORTANT, verified directly on our real data before writing this script:
illumination and environment codes are 0 ("N/A / Live") for *every* live
image in this dataset -- only spoof (attack) images carry a real
illumination/environment code. So illumination/environment breakdowns
below are computed over spoof images only, and report APCER (the only
metric that's meaningful for a spoof-only subgroup), not BPCER or
accuracy-with-live -- there is no live subgroup to compute those against
within a single illumination/environment bucket. This is stated plainly
in both the printed output and the saved JSON, not hidden.

"Face Mask", "Upper Body Mask", and "Region Mask" (spoof-type codes 4, 5,
6) are PAPER/PRINT attacks -- a printed photo of a face, upper body, or
specific face region, held up to the camera -- NOT a person wearing a
real cloth/surgical mask. They are renamed "paper face-mask attack" /
"paper upper-body-mask attack" / "paper region-mask attack" everywhere in
this script's output to avoid exactly that confusion.

Usage:
    python evaluate_breakdown.py --backbone mobilenet_v3_small
    python evaluate_breakdown.py --backbone resnet18   # once that checkpoint exists
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import LivenessDataset, get_eval_transform
from model import build_model

# Verbatim from https://github.com/ZhangYuanhan-AI/CelebA-Spoof/blob/master/README.md
# ("Rich Annotations" table), fetched and grepped directly from the raw
# README source on this date. "Face Mask" / "Upper Body Mask" / "Region
# Mask" renamed per this project's own convention to avoid confusion with
# someone wearing a real mask (see module docstring).
SPOOF_TYPE_NAMES = {
    1: "Photo", 2: "Poster", 3: "A4",
    4: "paper face-mask attack", 5: "paper upper-body-mask attack", 6: "paper region-mask attack",
    7: "PC", 8: "Pad", 9: "Phone", 10: "3D Mask",
}
ILLUMINATION_NAMES = {1: "Normal", 2: "Strong", 3: "Back", 4: "Dark"}
ENVIRONMENT_NAMES = {1: "Indoor", 2: "Outdoor"}  # README has a typo "Ourdoor"; corrected here

LOW_CONFIDENCE_N = 50

METAS_FILES = [
    "metas/intra_test/train_label.json",
    "metas/intra_test/test_label.json",
]


def load_official_labels(raw_root: str) -> dict:
    """relative_image_path -> full 44-element attribute vector."""
    labels = {}
    for rel in METAS_FILES:
        path = os.path.join(raw_root, rel)
        if not os.path.isfile(path):
            raise SystemExit(f"Official label file not found: {path}\n"
                              f"This script needs the raw CelebA-Spoof download (see README.md).")
        with open(path) as f:
            data = json.load(f)
        for img_path, attrs in data.items():
            labels[os.path.normpath(img_path)] = attrs
    return labels


def resolve_test_set_annotations(test_root: str, raw_root: str, official_labels: dict,
                                  annotation_lookup_root: str = None):
    """
    For every image actually in data/celeba_spoof/test/{live,spoof}/ (these
    are symlinks into the raw download), resolve the real file, look up its
    official annotation vector, and return a list of per-image dicts:
    {filepath, folder_label, spoof_type, illumination, environment, official_live_spoof}

    annotation_lookup_root: if test_root points at a dataset of REAL (non-
    symlink) files that nonetheless share filenames 1:1 with the original
    symlinked subset -- e.g. data/celeba_spoof_cropped, built by
    build_face_cropped_dataset.py from data/celeba_spoof with filenames
    preserved -- pass the original subset's root here. Annotations are then
    looked up via <annotation_lookup_root>/<class>/<same filename>'s symlink
    instead of test_root's own file, while `filepath` in the returned
    records still points at the real (cropped) image to load. Defaults to
    test_root itself, which is the original, unchanged behavior.
    """
    raw_root_abs = os.path.abspath(raw_root)
    lookup_root = annotation_lookup_root or test_root
    records = []
    missing = 0

    for class_name, folder_label in (("live", config.LABEL_LIVE), ("spoof", config.LABEL_SPOOF)):
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        lookup_dir = os.path.join(lookup_root, class_name)
        for fname in sorted(os.listdir(class_dir)):
            full = os.path.join(class_dir, fname)
            lookup_full = os.path.join(lookup_dir, fname)
            if not os.path.islink(lookup_full):
                continue
            real = os.path.realpath(lookup_full)
            rel = os.path.normpath(os.path.relpath(real, raw_root_abs))
            attrs = official_labels.get(rel)
            if attrs is None:
                missing += 1
                continue
            records.append({
                "filepath": full,
                "folder_label": folder_label,
                "spoof_type": attrs[40],
                "illumination": attrs[41],
                "environment": attrs[42],
                "official_live_spoof": attrs[43],
            })

    if missing:
        print(f"WARNING: {missing} test images had no matching official annotation "
              f"(skipped from the breakdown).")
    return records


class _RecordDataset(torch.utils.data.Dataset):
    """Module-level (not nested) so it can be pickled by multiprocessing
    DataLoader workers -- a locally-nested class here raised
    AttributeError: Can't get local object, caught by actually running this
    script rather than assuming it would work."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.records[idx]["filepath"]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def run_inference(model, records, device, image_size, batch_size=32):
    """Runs the real model over every record's image, batched, returns
    predicted labels (0=live, 1=spoof) in the same order as records."""
    transform = get_eval_transform(image_size)
    loader = DataLoader(_RecordDataset(records, transform), batch_size=batch_size,
                         shuffle=False, num_workers=config.NUM_WORKERS)

    preds = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.extend((probs >= config.DECISION_THRESHOLD).astype(int).tolist())
    return preds


def summarize_group(name: str, n: int, apcer: float, low_confidence: bool) -> dict:
    return {
        "name": name,
        "n": n,
        "apcer": apcer,
        "low_confidence": low_confidence,
    }


def plot_bar(groups: list, title: str, metric_key: str, out_path: str, ylabel: str):
    """Matches the visual style of evaluate.py's confusion matrix plot (same
    figure size convention, blue palette, clean grid)."""
    names = [g["name"] for g in groups]
    values = [g[metric_key] * 100 for g in groups]
    ns = [g["n"] for g in groups]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1), 5))
    colors = ["#c0392b" if g["low_confidence"] else "#2874a6" for g in groups]
    bars = ax.bar(names, values, color=colors)
    # Headroom above the tallest bar so its n= label never collides with
    # the title -- a real layout bug, not just a hypothetical, seen on the
    # first real render of this chart (the PC bar's label sat on the title).
    max_val = max(values) if values else 0
    ax.set_ylim(0, max_val * 1.18 + 1.5)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.02 + 0.15,
                 f"n={n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Per-attack-type/illumination/environment breakdown")
    parser.add_argument("--backbone", default=config.BACKBONE,
                         choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"])
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to models/best_liveness.pth for the default backbone, "
                              "or models/<backbone>/best_liveness.pth otherwise (matches train.py's "
                              "auto-namespacing).")
    parser.add_argument("--test_root", default=os.path.join(config.CELEBA_SPOOF_DIR, "test"))
    parser.add_argument("--annotation_lookup_root", default=None,
                         help="Use when --test_root points at real (non-symlink) files that still "
                              "share filenames 1:1 with the original symlinked subset -- e.g. "
                              "data/celeba_spoof_cropped/test, built by build_face_cropped_dataset.py. "
                              "Annotations are looked up via this root's matching symlink instead. "
                              "Defaults to --test_root itself (original behavior, unchanged).")
    parser.add_argument("--raw_root", default=os.path.join(config.PROJECT_ROOT, "raw_downloads", "CelebA_Spoof", "CelebA_Spoof"))
    parser.add_argument("--out_json", default=os.path.join(config.PROJECT_ROOT, "results", "breakdown.json"))
    parser.add_argument("--out_plots_dir", default=os.path.join(config.PROJECT_ROOT, "results", "plots"))
    args = parser.parse_args()

    if args.checkpoint is None:
        is_default = args.backbone == config.BACKBONE
        args.checkpoint = config.BEST_MODEL_PATH if is_default else os.path.join(config.MODELS_DIR, args.backbone, "best_liveness.pth")

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("backbone", args.backbone)
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded: backbone={backbone}, image_size={image_size}, trained_val_acc={checkpoint.get('val_acc', 'n/a')}")

    print(f"\nLoading official CelebA-Spoof annotations from {args.raw_root} ...")
    official_labels = load_official_labels(args.raw_root)
    print(f"Loaded {len(official_labels)} official annotation entries.")

    print(f"\nResolving test-set images against official annotations...")
    records = resolve_test_set_annotations(args.test_root, args.raw_root, official_labels,
                                            annotation_lookup_root=args.annotation_lookup_root)
    print(f"Resolved {len(records)} test images with real annotations.")

    # Sanity check: folder label should always match the official live/spoof
    # label (this was already guaranteed by prepare_celeba_subset.py's own
    # label-correction fix, but verify again here rather than assume).
    mismatches = sum(1 for r in records if r["folder_label"] != r["official_live_spoof"])
    print(f"Folder-label vs official-label mismatches in test set: {mismatches} (should be 0)")

    print(f"\nRunning real inference on {len(records)} test images...")
    preds = run_inference(model, records, device, image_size)
    for r, p in zip(records, preds):
        r["pred_label"] = p

    # ---------------------------------------------------------------
    # Breakdown 1: per spoof-type (spoof images only, by construction)
    # ---------------------------------------------------------------
    spoof_records = [r for r in records if r["official_live_spoof"] == config.LABEL_SPOOF]
    print(f"\n=== Per-spoof-type breakdown (attack rejection rate / APCER), n={len(spoof_records)} spoof test images ===")
    spoof_type_groups = []
    for code in sorted(SPOOF_TYPE_NAMES):
        group = [r for r in spoof_records if r["spoof_type"] == code]
        n = len(group)
        if n == 0:
            continue
        apcer = float(np.mean([r["pred_label"] == config.LABEL_LIVE for r in group]))
        low_conf = n < LOW_CONFIDENCE_N
        name = SPOOF_TYPE_NAMES[code]
        spoof_type_groups.append(summarize_group(name, n, apcer, low_conf))
        flag = "  [LOW CONFIDENCE, n<50]" if low_conf else ""
        print(f"  {name:32s} n={n:5d}  APCER={apcer*100:6.2f}%{flag}")

    # ---------------------------------------------------------------
    # Breakdown 2: per illumination (spoof images only -- live images
    # carry no real illumination code in this dataset, verified earlier)
    # ---------------------------------------------------------------
    print(f"\n=== Per-illumination breakdown (APCER, spoof images only -- live images have no real illumination label in this dataset) ===")
    illumination_groups = []
    for code in sorted(ILLUMINATION_NAMES):
        group = [r for r in spoof_records if r["illumination"] == code]
        n = len(group)
        if n == 0:
            continue
        apcer = float(np.mean([r["pred_label"] == config.LABEL_LIVE for r in group]))
        low_conf = n < LOW_CONFIDENCE_N
        name = ILLUMINATION_NAMES[code]
        illumination_groups.append(summarize_group(name, n, apcer, low_conf))
        flag = "  [LOW CONFIDENCE, n<50]" if low_conf else ""
        print(f"  {name:32s} n={n:5d}  APCER={apcer*100:6.2f}%{flag}")

    # ---------------------------------------------------------------
    # Breakdown 3: per environment (spoof images only, same reason)
    # ---------------------------------------------------------------
    print(f"\n=== Per-environment breakdown (APCER, spoof images only) ===")
    environment_groups = []
    for code in sorted(ENVIRONMENT_NAMES):
        group = [r for r in spoof_records if r["environment"] == code]
        n = len(group)
        if n == 0:
            continue
        apcer = float(np.mean([r["pred_label"] == config.LABEL_LIVE for r in group]))
        low_conf = n < LOW_CONFIDENCE_N
        name = ENVIRONMENT_NAMES[code]
        environment_groups.append(summarize_group(name, n, apcer, low_conf))
        flag = "  [LOW CONFIDENCE, n<50]" if low_conf else ""
        print(f"  {name:32s} n={n:5d}  APCER={apcer*100:6.2f}%{flag}")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(args.out_plots_dir, exist_ok=True)

    output = {
        "backbone": backbone,
        "checkpoint": args.checkpoint,
        "source_mapping_url": "https://github.com/ZhangYuanhan-AI/CelebA-Spoof/blob/master/README.md",
        "note": (
            "Illumination/environment breakdowns are computed over spoof images only "
            "(live images carry no real illumination/environment code in this dataset's "
            "own annotations -- verified directly on the raw label data). Only APCER is "
            "reported for those two breakdowns, not BPCER/accuracy, since there is no live "
            "subgroup within a single illumination/environment bucket to compute them against."
        ),
        "low_confidence_threshold_n": LOW_CONFIDENCE_N,
        "spoof_type_breakdown": spoof_type_groups,
        "illumination_breakdown": illumination_groups,
        "environment_breakdown": environment_groups,
    }
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved breakdown JSON to: {args.out_json}")

    suffix = "" if backbone == config.BACKBONE else f"_{backbone}"
    plot_bar(spoof_type_groups, "APCER by Spoof Attack Type", "apcer",
             os.path.join(args.out_plots_dir, f"breakdown_spoof_type{suffix}.png"),
             "APCER (%) -- attack wrongly accepted as live")
    plot_bar(illumination_groups, "APCER by Illumination Condition", "apcer",
             os.path.join(args.out_plots_dir, f"breakdown_illumination{suffix}.png"),
             "APCER (%)")
    plot_bar(environment_groups, "APCER by Environment", "apcer",
             os.path.join(args.out_plots_dir, f"breakdown_environment{suffix}.png"),
             "APCER (%)")
    print(f"Saved 3 bar charts to: {args.out_plots_dir}")

    # Weakest groups summary
    all_groups = [("spoof type", g) for g in spoof_type_groups] + \
                 [("illumination", g) for g in illumination_groups] + \
                 [("environment", g) for g in environment_groups]
    all_groups.sort(key=lambda t: t[1]["apcer"], reverse=True)
    print("\n=== Weakest groups (highest APCER) ===")
    for category, g in all_groups[:5]:
        flag = " [LOW CONFIDENCE]" if g["low_confidence"] else ""
        print(f"  [{category}] {g['name']}: APCER={g['apcer']*100:.2f}% (n={g['n']}){flag}")


if __name__ == "__main__":
    main()
