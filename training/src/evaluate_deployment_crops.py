"""
evaluate_deployment_crops.py
-----------------------------
TASK F verification: measures what actually happens when the REAL
deployment preprocessing (face_utils.FaceDetector.detect + crop_face,
exactly as realtime_pipeline.py / webapp/backend/app.py use them) is run
on the test set, instead of feeding the model the raw full-frame image the
way training/evaluate.py currently does.

For each test image (same 3900 files/labels as predictions_test.csv):
  1. Run BlazeFace detection on the full raw frame (as the live pipeline
     sees a webcam frame).
  2. If a face is found: crop with the same +25% margin as deployment,
     run it through the SAME trained checkpoint, record the prediction.
  3. If no face is found: this is a genuine Stage-1 miss. The real
     deployment never makes a liveness decision in this case (the UI just
     shows "No face detected") -- so it is reported as its own explicit
     "miss" outcome, not silently dropped and not scored as a wrong
     prediction either way.

Reports accuracy/APCER/BPCER/HTER on the DETECTED subset (mirroring the
coverage-style reporting already used for the gray-zone analysis) plus the
miss rate overall and per spoof type, and saves N side-by-side
(training-input | deployment-crop) examples for visual inspection.

Usage:
    python evaluate_deployment_crops.py
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch

import config
from dataset import get_eval_transform
from evaluate_breakdown import SPOOF_TYPE_NAMES, load_official_labels
from face_utils import FaceDetector, bgr_crop_to_pil, crop_face
from model import build_model

N_EXAMPLES = 10


def load_test_rows(csv_path: str):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["true_label"] = int(r["true_label"])
    return rows


def attach_spoof_type(rows, raw_root: str):
    official_labels = load_official_labels(raw_root)
    raw_root_abs = os.path.abspath(raw_root)
    for r in rows:
        real = os.path.realpath(r["filepath"])
        rel = os.path.normpath(os.path.relpath(real, raw_root_abs))
        attrs = official_labels.get(rel)
        r["spoof_type"] = attrs[40] if attrs is not None else None
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate the real deployment face-crop pipeline on the test set")
    parser.add_argument("--test_csv", default=os.path.join(config.PREDICTIONS_DIR, "predictions_test.csv"))
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--raw_root", default=os.path.join(config.PROJECT_ROOT, "raw_downloads", "CelebA_Spoof", "CelebA_Spoof"))
    parser.add_argument("--out_json", default=os.path.join(config.PROJECT_ROOT, "results", "deployment_crop_eval.json"))
    parser.add_argument("--crop_check_dir", default=os.path.join(config.PROJECT_ROOT, "results", "crop_check"))
    args = parser.parse_args()

    if not os.path.isfile(args.test_csv):
        raise SystemExit(f"{args.test_csv} not found -- run evaluate.py first.")

    os.makedirs(args.crop_check_dir, exist_ok=True)

    device = config.DEVICE
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("backbone", config.BACKBONE)
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = get_eval_transform(image_size)
    print(f"Loaded checkpoint: {args.checkpoint} (backbone={backbone})")

    print(f"Loading test rows from {args.test_csv} ...")
    rows = load_test_rows(args.test_csv)
    print(f"Attaching official spoof-type annotations ...")
    rows = attach_spoof_type(rows, args.raw_root)
    print(f"Loaded {len(rows)} test images.")

    detector = FaceDetector()

    results = []
    t0 = time.time()

    for i, r in enumerate(rows):
        frame_bgr = cv2.imread(r["filepath"])
        if frame_bgr is None:
            results.append({**r, "detected": False, "pred_label": None, "prob_spoof": None})
            continue

        detections = detector.detect(frame_bgr)
        if not detections:
            results.append({**r, "detected": False, "pred_label": None, "prob_spoof": None})
            continue

        det = detections[0]  # real pipeline iterates all detections; CelebA-Spoof images have at most one subject
        face_crop = crop_face(frame_bgr, det)
        if face_crop is None or face_crop.size == 0:
            results.append({**r, "detected": False, "pred_label": None, "prob_spoof": None})
            continue

        pil_img = bgr_crop_to_pil(face_crop)
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            prob_spoof = torch.sigmoid(logit).item()
        pred_label = config.LABEL_SPOOF if prob_spoof >= config.DECISION_THRESHOLD else config.LABEL_LIVE

        results.append({**r, "detected": True, "pred_label": pred_label, "prob_spoof": prob_spoof})

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(rows)} processed ({elapsed:.0f}s elapsed)")

    detector.close()
    print(f"Done in {time.time()-t0:.0f}s.")

    # ------------------------------------------------------------------
    # Metrics on the DETECTED subset
    # ------------------------------------------------------------------
    detected = [r for r in results if r["detected"]]
    missed = [r for r in results if not r["detected"]]
    n_total = len(results)
    n_detected = len(detected)
    n_missed = len(missed)

    y_true = np.array([r["true_label"] for r in detected])
    y_pred = np.array([r["pred_label"] for r in detected])

    spoof_mask = y_true == config.LABEL_SPOOF
    live_mask = y_true == config.LABEL_LIVE
    apcer = float(np.mean(y_pred[spoof_mask] == config.LABEL_LIVE)) if spoof_mask.sum() else float("nan")
    bpcer = float(np.mean(y_pred[live_mask] == config.LABEL_SPOOF)) if live_mask.sum() else float("nan")
    hter = (apcer + bpcer) / 2
    accuracy = float(np.mean(y_true == y_pred)) if n_detected else float("nan")

    print("\n=== Deployment-crop evaluation on TEST set (detected subset) ===")
    print(f"Total test images: {n_total}")
    print(f"Face detected: {n_detected} ({n_detected/n_total*100:.2f}%)")
    print(f"Face detection MISSED: {n_missed} ({n_missed/n_total*100:.2f}%)")
    print(f"Accuracy (detected subset): {accuracy*100:.2f}%  (raw-frame baseline: 99.15%)")
    print(f"APCER: {apcer*100:.2f}%   BPCER: {bpcer*100:.2f}%   HTER: {hter*100:.2f}%")

    # Miss rate per spoof type (spoof images) + live
    miss_by_type = {}
    for r in results:
        if r["true_label"] == config.LABEL_LIVE:
            type_name = "LIVE"
        else:
            type_name = SPOOF_TYPE_NAMES.get(r["spoof_type"], f"unknown({r['spoof_type']})")
        miss_by_type.setdefault(type_name, {"total": 0, "missed": 0})
        miss_by_type[type_name]["total"] += 1
        if not r["detected"]:
            miss_by_type[type_name]["missed"] += 1

    print("\nFace-detection miss rate by category:")
    print(f"{'Category':<30}{'n':<8}{'missed':<10}{'miss_rate':<10}")
    for name in sorted(miss_by_type.keys()):
        d = miss_by_type[name]
        rate = d["missed"] / d["total"] * 100 if d["total"] else float("nan")
        print(f"{name:<30}{d['total']:<8}{d['missed']:<10}{rate:<10.2f}")

    # ------------------------------------------------------------------
    # Save N side-by-side (training-input | deployment-crop) examples
    # ------------------------------------------------------------------
    detector2 = FaceDetector()  # fresh instance for the example pass (first was closed)
    saved = 0
    for r in results:
        if saved >= N_EXAMPLES:
            break
        frame_bgr = cv2.imread(r["filepath"])
        if frame_bgr is None:
            continue
        training_input = cv2.resize(frame_bgr, (224, 224))

        if r["detected"]:
            detections = detector2.detect(frame_bgr)
            if not detections:
                continue
            crop = crop_face(frame_bgr, detections[0])
            if crop is None:
                continue
            deploy_crop = cv2.resize(crop, (224, 224))
            label = f"DETECTED | face/frame mismatch visible"
        else:
            deploy_crop = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.putText(deploy_crop, "NO FACE", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            label = "FACE DETECTION MISSED"

        pair = np.hstack([training_input, deploy_crop])
        strip = np.zeros((26, pair.shape[1], 3), dtype=np.uint8)
        cv2.putText(strip, f"train input | deploy crop -- {label}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tile = np.vstack([pair, strip])
        out_name = f"{saved:02d}_{os.path.basename(r['filepath'])}.png"
        cv2.imwrite(os.path.join(args.crop_check_dir, out_name), tile)
        saved += 1
    detector2.close()
    print(f"\nSaved {saved} side-by-side crop-check examples to {args.crop_check_dir}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "checkpoint": args.checkpoint,
        "n_total": n_total,
        "n_detected": n_detected,
        "n_missed": n_missed,
        "detection_rate": n_detected / n_total,
        "miss_rate": n_missed / n_total,
        "metrics_on_detected_subset": {
            "accuracy": accuracy,
            "apcer": apcer,
            "bpcer": bpcer,
            "hter": hter,
        },
        "baseline_raw_frame_accuracy": 0.9915384615384616,
        "miss_rate_by_category": {
            name: {"total": d["total"], "missed": d["missed"], "miss_rate": d["missed"] / d["total"] if d["total"] else None}
            for name, d in miss_by_type.items()
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved full results to: {args.out_json}")


if __name__ == "__main__":
    main()
