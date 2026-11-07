"""
generate_gradcam_gallery.py
-----------------------------
Generates a Grad-CAM gallery from the real test-set predictions
(outputs/predictions/predictions_test.csv, produced by evaluate.py) and
the real official CelebA-Spoof annotations (for spoof-type grouping, same
technique as evaluate_breakdown.py):

    - ALL 22 false accepts (true=spoof, predicted=live)
    - ALL 11 false rejects (true=live, predicted=spoof)
    - 2 correctly-classified examples per spoof type (10 types -> 20 images)
    - 6 correctly-classified live examples

Each is saved as a side-by-side PNG: original 224x224 crop | Grad-CAM
overlay (target="predicted" -- explains whichever class the model actually
predicted for that image; for false accepts/rejects this directly answers
"what made the model think this", which is the diagnostically useful
question).

Usage:
    python generate_gradcam_gallery.py
    python generate_gradcam_gallery.py --backbone resnet18   # once trained
"""

import argparse
import csv
import json
import os
import random

import cv2
import numpy as np
import torch

import config
from dataset import get_eval_transform
from evaluate_breakdown import SPOOF_TYPE_NAMES, load_official_labels
from face_utils import bgr_crop_to_pil
from gradcam import GradCAM, overlay_heatmap
from model import build_model

SEED = 42
N_CORRECT_PER_SPOOF_TYPE = 2
N_CORRECT_LIVE = 6
TILE_SIZE = 224


def load_predictions(csv_path: str):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["true_label"] = int(r["true_label"])
        r["pred_label"] = int(r["pred_label"])
        r["prob_spoof"] = float(r["prob_spoof"])
    return rows


def attach_spoof_type(rows, raw_root: str, annotation_lookup_root: str = None):
    """Looks up each row's official spoof_type (idx 40) via the same
    symlink-resolution technique as evaluate_breakdown.py. Only meaningful
    for spoof-labeled rows (live rows get spoof_type=0, the dataset's own
    'N/A' marker -- verified in evaluate_breakdown.py's investigation).

    annotation_lookup_root: set when rows[]['filepath'] points at real
    (non-symlink) files that still share filenames 1:1 with the original
    symlinked subset -- e.g. data/celeba_spoof_cropped, built by
    build_face_cropped_dataset.py -- so the lookup uses that original
    subset's matching symlink instead of realpath-ing the cropped file
    itself (which would just return itself and match nothing). Defaults to
    None, i.e. the original, unchanged behavior."""
    official_labels = load_official_labels(raw_root)
    raw_root_abs = os.path.abspath(raw_root)
    missing = 0
    for r in rows:
        if annotation_lookup_root:
            class_name = os.path.basename(os.path.dirname(r["filepath"]))
            lookup_path = os.path.join(annotation_lookup_root, class_name, os.path.basename(r["filepath"]))
        else:
            lookup_path = r["filepath"]
        real = os.path.realpath(lookup_path)
        rel = os.path.normpath(os.path.relpath(real, raw_root_abs))
        attrs = official_labels.get(rel)
        if attrs is None:
            r["spoof_type"] = None
            missing += 1
        else:
            r["spoof_type"] = attrs[40]
    if missing:
        print(f"WARNING: {missing} rows had no official annotation match.")
    return rows


def build_tile(crop_bgr: np.ndarray, heatmap: np.ndarray, label_text: str) -> np.ndarray:
    """original | overlay, stacked side by side, with a label strip below."""
    crop_resized = cv2.resize(crop_bgr, (TILE_SIZE, TILE_SIZE))
    overlay = overlay_heatmap(crop_resized, heatmap)

    pair = np.hstack([crop_resized, overlay])
    label_strip = np.zeros((28, pair.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_strip, label_text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([pair, label_strip])


def main():
    parser = argparse.ArgumentParser(description="Generate the Grad-CAM gallery from real test-set predictions")
    parser.add_argument("--backbone", default=config.BACKBONE,
                         choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions_csv", default=os.path.join(config.PREDICTIONS_DIR, "predictions_test.csv"))
    parser.add_argument("--raw_root", default=os.path.join(config.PROJECT_ROOT, "raw_downloads", "CelebA_Spoof", "CelebA_Spoof"))
    parser.add_argument("--out_dir", default=os.path.join(config.PROJECT_ROOT, "results", "gradcam"))
    parser.add_argument("--annotation_lookup_root", default=None,
                         help="Use when --predictions_csv was generated from real (non-symlink) files "
                              "sharing filenames 1:1 with the original symlinked subset -- e.g. "
                              "data/celeba_spoof_cropped. Defaults to None (original behavior).")
    args = parser.parse_args()

    if args.checkpoint is None:
        is_default = args.backbone == config.BACKBONE
        args.checkpoint = config.BEST_MODEL_PATH if is_default else os.path.join(config.MODELS_DIR, args.backbone, "best_liveness.pth")

    if not os.path.isfile(args.predictions_csv):
        raise SystemExit(f"{args.predictions_csv} not found -- run evaluate.py first.")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(SEED)

    device = config.DEVICE
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("backbone", args.backbone)
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (backbone={backbone})")

    transform = get_eval_transform(image_size)

    print(f"Loading predictions from {args.predictions_csv} ...")
    rows = load_predictions(args.predictions_csv)
    print(f"Loaded {len(rows)} real predictions.")

    print(f"Attaching official spoof-type annotations ...")
    rows = attach_spoof_type(rows, args.raw_root, annotation_lookup_root=args.annotation_lookup_root)

    false_accepts = [r for r in rows if r["true_label"] == config.LABEL_SPOOF and r["pred_label"] == config.LABEL_LIVE]
    false_rejects = [r for r in rows if r["true_label"] == config.LABEL_LIVE and r["pred_label"] == config.LABEL_SPOOF]
    correct_spoof = [r for r in rows if r["true_label"] == config.LABEL_SPOOF and r["pred_label"] == config.LABEL_SPOOF]
    correct_live = [r for r in rows if r["true_label"] == config.LABEL_LIVE and r["pred_label"] == config.LABEL_LIVE]

    print(f"False accepts: {len(false_accepts)}  False rejects: {len(false_rejects)}  "
          f"Correct spoof: {len(correct_spoof)}  Correct live: {len(correct_live)}")

    selected = []
    for r in false_accepts:
        selected.append((r, "false_accept"))
    for r in false_rejects:
        selected.append((r, "false_reject"))

    for code, name in sorted(SPOOF_TYPE_NAMES.items()):
        pool = [r for r in correct_spoof if r["spoof_type"] == code]
        if len(pool) < N_CORRECT_PER_SPOOF_TYPE:
            print(f"WARNING: only {len(pool)} correct examples available for spoof type '{name}' (wanted {N_CORRECT_PER_SPOOF_TYPE})")
        chosen = random.sample(pool, min(N_CORRECT_PER_SPOOF_TYPE, len(pool)))
        for r in chosen:
            selected.append((r, "correct"))

    chosen_live = random.sample(correct_live, min(N_CORRECT_LIVE, len(correct_live)))
    for r in chosen_live:
        selected.append((r, "correct"))

    print(f"\nTotal images selected for the gallery: {len(selected)}")

    gallery = []
    with GradCAM(model, model.feature_extractor.features[12]) as cam:
        for i, (r, category) in enumerate(selected):
            img_bgr = cv2.imread(r["filepath"])
            if img_bgr is None:
                print(f"  SKIP (could not read): {r['filepath']}")
                continue

            pil_img = bgr_crop_to_pil(img_bgr)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            heatmap, prob_spoof = cam(tensor, target="predicted")

            true_name = "LIVE" if r["true_label"] == config.LABEL_LIVE else "SPOOF"
            pred_name = "LIVE" if r["pred_label"] == config.LABEL_LIVE else "SPOOF"
            spoof_type_name = SPOOF_TYPE_NAMES.get(r["spoof_type"], "N/A (live)")

            out_name = f"{i:03d}_{category}_{os.path.basename(r['filepath'])}.png"
            out_path = os.path.join(args.out_dir, out_name)
            label_text = f"{category} | true={true_name} pred={pred_name} P(spoof)={prob_spoof:.3f} | {spoof_type_name}"
            tile = build_tile(img_bgr, heatmap, label_text)
            cv2.imwrite(out_path, tile)

            gallery.append({
                "file": out_name,
                "source_file": r["filepath"],
                "true_label": true_name,
                "spoof_type": spoof_type_name,
                "predicted_label": pred_name,
                "spoof_probability": prob_spoof,
                "category": category,
            })

    out_json = os.path.join(args.out_dir, "gallery.json")
    with open(out_json, "w") as f:
        json.dump(gallery, f, indent=2)

    print(f"\nSaved {len(gallery)} gallery tiles to: {args.out_dir}")
    print(f"Saved gallery index to: {out_json}")


if __name__ == "__main__":
    main()
