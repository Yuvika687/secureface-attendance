"""
build_face_cropped_dataset.py
-------------------------------
TASK G step 2: builds a face-cropped version of the EXISTING identity-
disjoint subset (data/celeba_spoof/{train,val,test}/{live,spoof}/*) --
same images, same splits, no re-sampling -- using the SAME crop function
the live deployment pipeline uses (face_utils.crop_face, BlazeFace
detection + 25% margin), instead of the raw full frames the subset
currently points at.

For each existing file:
  1. Resolve the real raw image behind the symlink.
  2. Run BlazeFace detection on it, same as deployment.
  3. If BlazeFace finds a face: crop with face_utils.crop_face (the exact
     function the live pipeline calls).
  4. If BlazeFace finds nothing: fall back to the dataset's OWN official
     bounding box (<image>_BB.txt, per the CelebA-Spoof README's
     real_w/224, real_h/224 scaling), wrapped in the same Detection
     dataclass so it goes through the identical crop_face() margin logic
     -- the fallback and the primary path share one crop implementation,
     so they can never silently diverge in how the margin is applied.
  5. If even the official bbox is missing/invalid: true failure, logged
     explicitly and excluded (never silently dropped without a count).

Output layout mirrors the existing subset exactly:
    data/celeba_spoof_cropped/{train,val,test}/{live,spoof}/<same filename>

Usage:
    python build_face_cropped_dataset.py --subset_root ../data/celeba_spoof \
        --out_root ../data/celeba_spoof_cropped
"""

import argparse
import json
import os

import cv2

import config
from evaluate_breakdown import SPOOF_TYPE_NAMES, load_official_labels
from face_utils import Detection, FaceDetector, crop_face

SPLITS = ("train", "val", "test")
CLASSES = ("live", "spoof")


def read_official_bbox(image_path: str):
    """Reads <image>_BB.txt and scales it to real pixel coordinates per the
    official CelebA-Spoof README formula. Returns None if missing/invalid."""
    bb_path = os.path.splitext(image_path)[0] + "_BB.txt"
    if not os.path.isfile(bb_path):
        return None
    try:
        with open(bb_path) as f:
            parts = f.read().strip().split()
        bbox = [float(x) for x in parts[:4]]
        score = float(parts[4]) if len(parts) > 4 else 1.0
    except (ValueError, IndexError):
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None
    real_h, real_w = img.shape[:2]

    x1 = int(bbox[0] * (real_w / 224))
    y1 = int(bbox[1] * (real_h / 224))
    w1 = int(bbox[2] * (real_w / 224))
    h1 = int(bbox[3] * (real_h / 224))
    if w1 <= 0 or h1 <= 0:
        return None

    x1 = max(0, min(x1, real_w - 1))
    y1 = max(0, min(y1, real_h - 1))
    w1 = min(w1, real_w - x1)
    h1 = min(h1, real_h - y1)

    return Detection(x1, y1, w1, h1, score)


def main():
    parser = argparse.ArgumentParser(description="Build a face-cropped version of the existing identity-disjoint subset")
    parser.add_argument("--subset_root", default=os.path.join(config.PROJECT_ROOT, "data", "celeba_spoof"))
    parser.add_argument("--out_root", default=os.path.join(config.PROJECT_ROOT, "data", "celeba_spoof_cropped"))
    parser.add_argument("--raw_root", default=os.path.join(config.PROJECT_ROOT, "raw_downloads", "CelebA_Spoof", "CelebA_Spoof"))
    parser.add_argument("--stats_out", default=os.path.join(config.PROJECT_ROOT, "results", "crop_dataset_build_stats.json"))
    args = parser.parse_args()

    official_labels = load_official_labels(args.raw_root)
    raw_root_abs = os.path.abspath(args.raw_root)

    detector = FaceDetector()

    stats = {}  # split -> {"blazeface": n, "bbox_fallback": n, "failed": n, "by_spoof_type": {...}}
    failed_files = []

    for split in SPLITS:
        stats[split] = {"blazeface": 0, "bbox_fallback": 0, "failed": 0, "by_spoof_type": {}}
        for cls in CLASSES:
            src_dir = os.path.join(args.subset_root, split, cls)
            if not os.path.isdir(src_dir):
                print(f"WARNING: {src_dir} does not exist, skipping.")
                continue
            dst_dir = os.path.join(args.out_root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)

            filenames = sorted(os.listdir(src_dir))
            print(f"[{split}/{cls}] {len(filenames)} images ...")

            for fname in filenames:
                src_path = os.path.join(src_dir, fname)
                real_path = os.path.realpath(src_path)
                dst_path = os.path.join(dst_dir, fname)

                # Spoof-type grouping for the fallback-count breakdown.
                rel = os.path.normpath(os.path.relpath(real_path, raw_root_abs))
                attrs = official_labels.get(rel)
                type_name = "LIVE" if cls == "live" else SPOOF_TYPE_NAMES.get(attrs[40] if attrs else None, "unknown")
                stats[split]["by_spoof_type"].setdefault(type_name, {"blazeface": 0, "bbox_fallback": 0, "failed": 0})

                frame_bgr = cv2.imread(real_path)
                if frame_bgr is None:
                    stats[split]["failed"] += 1
                    stats[split]["by_spoof_type"][type_name]["failed"] += 1
                    failed_files.append(real_path)
                    continue

                detections = detector.detect(frame_bgr)
                used_fallback = False
                if detections:
                    crop = crop_face(frame_bgr, detections[0])
                else:
                    used_fallback = True
                    fallback_det = read_official_bbox(real_path)
                    crop = crop_face(frame_bgr, fallback_det) if fallback_det is not None else None

                if crop is None or crop.size == 0:
                    stats[split]["failed"] += 1
                    stats[split]["by_spoof_type"][type_name]["failed"] += 1
                    failed_files.append(real_path)
                    continue

                cv2.imwrite(dst_path, crop)
                key = "bbox_fallback" if used_fallback else "blazeface"
                stats[split][key] += 1
                stats[split]["by_spoof_type"][type_name][key] += 1

    detector.close()

    print("\n=== Crop build summary ===")
    for split in SPLITS:
        s = stats[split]
        total = s["blazeface"] + s["bbox_fallback"] + s["failed"]
        print(f"[{split}] total={total}  blazeface={s['blazeface']}  "
              f"bbox_fallback={s['bbox_fallback']}  failed={s['failed']}")

    if failed_files:
        print(f"\n{len(failed_files)} images FAILED both BlazeFace and the official bbox fallback "
              f"(no crop produced -- excluded from the cropped dataset):")
        for p in failed_files[:20]:
            print(f"  {p}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")

    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    with open(args.stats_out, "w") as f:
        json.dump({"stats_by_split": stats, "failed_files": failed_files}, f, indent=2)
    print(f"\nSaved build stats to: {args.stats_out}")
    print(f"Cropped dataset written to: {os.path.abspath(args.out_root)}")


if __name__ == "__main__":
    main()
