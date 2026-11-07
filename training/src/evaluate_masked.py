"""
evaluate_masked.py
--------------------
TASK I: synthetic masked-face evaluation.

For every real test-set image (data/celeba_spoof/test, live and spoof),
overlays a synthetic surgical-style mask (nose-to-chin, see
synthetic_mask.py) using real MediaPipe FaceLandmarker landmarks, cycling
through white/light_blue/black, then runs the masked image through the
REAL deployment pipeline (BlazeFace detection + crop_face + the v2
checkpoint) -- exactly the same path evaluate_deployment_crops.py uses
for the unmasked baseline, so masked vs unmasked is apples to apples.

Three distinct outcomes are tracked, never conflated:
  - mask_synthesis_failure: FaceLandmarker found no face/landmarks on the
    RAW image, so no mask could be synthesized at all.
  - stage1_miss: BlazeFace found no face on the MASKED image (a real
    Stage-1 detection failure, possibly caused by the mask itself).
  - detected: both succeeded; a real liveness prediction was made.

Also applies v2's real gray-zone band (same JSON evaluate_gray_zone.py
produced, never hardcoded) to the masked probabilities, reporting
coverage and decided-set metrics the same way evaluate_gray_zone.py does.

Usage:
    python evaluate_masked.py
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
from face_utils import FaceDetector, bgr_crop_to_pil, crop_face
from liveness_infer import classify_three_way, load_gray_zone_band
from model import build_model
from synthetic_mask import MASK_COLORS, FaceLandmarkerWrapper, apply_synthetic_mask

N_EXAMPLES = 12
SEED = 42


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


def metrics_from_outcomes(outcomes):
    """outcomes: list of dicts with true_label, pred_label (only the
    'detected' subset). Returns accuracy/APCER/BPCER/HTER."""
    if not outcomes:
        return {"n": 0, "accuracy": None, "apcer": None, "bpcer": None, "hter": None}
    y_true = np.array([o["true_label"] for o in outcomes])
    y_pred = np.array([o["pred_label"] for o in outcomes])
    spoof_mask = y_true == config.LABEL_SPOOF
    live_mask = y_true == config.LABEL_LIVE
    apcer = float(np.mean(y_pred[spoof_mask] == config.LABEL_LIVE)) if spoof_mask.sum() else None
    bpcer = float(np.mean(y_pred[live_mask] == config.LABEL_SPOOF)) if live_mask.sum() else None
    hter = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    accuracy = float(np.mean(y_true == y_pred))
    return {"n": len(outcomes), "accuracy": accuracy, "apcer": apcer, "bpcer": bpcer, "hter": hter,
            "n_live": int(live_mask.sum()), "n_spoof": int(spoof_mask.sum())}


def main():
    parser = argparse.ArgumentParser(description="Synthetic masked-face evaluation (TASK I)")
    parser.add_argument("--test_csv", default=os.path.join(config.PREDICTIONS_DIR, "predictions_test.csv"))
    parser.add_argument("--checkpoint", default=os.path.join(config.MODELS_DIR, "v2_cropped", "best_liveness.pth"))
    parser.add_argument("--raw_root", default=os.path.join(config.PROJECT_ROOT, "raw_downloads", "CelebA_Spoof", "CelebA_Spoof"))
    parser.add_argument("--out_json", default=os.path.join(config.PROJECT_ROOT, "results", "masked_eval.json"))
    parser.add_argument("--examples_dir", default=os.path.join(config.PROJECT_ROOT, "results", "masked", "examples"))
    parser.add_argument("--unmasked_baseline_json", default=os.path.join(config.PROJECT_ROOT, "results", "deployment_crop_eval_v2.json"))
    args = parser.parse_args()

    random.seed(SEED)
    os.makedirs(args.examples_dir, exist_ok=True)

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
    rows = attach_spoof_type(rows, args.raw_root)
    print(f"Loaded {len(rows)} test images.")

    colors = list(MASK_COLORS.keys())  # ["white", "light_blue", "black"]

    face_detector = FaceDetector()
    landmarker = FaceLandmarkerWrapper()

    results = []
    example_candidates = {c: {"live": [], "spoof": []} for c in colors}

    import time
    t0 = time.time()
    for i, r in enumerate(rows):
        color = colors[i % len(colors)]
        frame_bgr = cv2.imread(r["filepath"])
        if frame_bgr is None:
            results.append({**r, "color": color, "mask_synthesis_failure": True,
                             "stage1_detected": False, "pred_label": None, "prob_spoof": None})
            continue

        landmarks = landmarker.detect(frame_bgr)
        if landmarks is None:
            results.append({**r, "color": color, "mask_synthesis_failure": True,
                             "stage1_detected": False, "pred_label": None, "prob_spoof": None})
            continue

        masked_bgr = apply_synthetic_mask(frame_bgr, landmarks, color=color)

        detections = face_detector.detect(masked_bgr)
        if not detections:
            results.append({**r, "color": color, "mask_synthesis_failure": False,
                             "stage1_detected": False, "pred_label": None, "prob_spoof": None})
            continue

        det = detections[0]
        crop = crop_face(masked_bgr, det)
        if crop is None or crop.size == 0:
            results.append({**r, "color": color, "mask_synthesis_failure": False,
                             "stage1_detected": False, "pred_label": None, "prob_spoof": None})
            continue

        pil_img = bgr_crop_to_pil(crop)
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            prob_spoof = torch.sigmoid(logit).item()
        pred_label = config.LABEL_SPOOF if prob_spoof >= config.DECISION_THRESHOLD else config.LABEL_LIVE

        results.append({**r, "color": color, "mask_synthesis_failure": False,
                         "stage1_detected": True, "pred_label": pred_label, "prob_spoof": prob_spoof})

        cls = "live" if r["true_label"] == config.LABEL_LIVE else "spoof"
        example_candidates[color][cls].append((r["filepath"], masked_bgr))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)} processed ({time.time()-t0:.0f}s elapsed)")

    face_detector.close()
    landmarker.close()
    print(f"Done in {time.time()-t0:.0f}s.")

    # ------------------------------------------------------------------
    # Save example masked images: a mix of live/spoof across all 3 colors.
    # ------------------------------------------------------------------
    saved = 0
    per_cell_target = max(1, N_EXAMPLES // (len(colors) * 2))  # roughly even across color x {live,spoof}
    example_index = []
    for color in colors:
        for cls in ("live", "spoof"):
            pool = example_candidates[color][cls]
            if not pool:
                continue
            chosen = random.sample(pool, min(per_cell_target, len(pool)))
            for filepath, masked_bgr in chosen:
                if saved >= N_EXAMPLES:
                    break
                out_name = f"{saved:02d}_{color}_{cls}_{os.path.basename(filepath)}.png"
                cv2.imwrite(os.path.join(args.examples_dir, out_name), masked_bgr)
                example_index.append({"file": out_name, "color": color, "true_label": cls, "source_file": filepath})
                saved += 1
    print(f"Saved {saved} example masked images to {args.examples_dir}")

    # ------------------------------------------------------------------
    # Stage 1 (face detection) + mask-synthesis-failure stats
    # ------------------------------------------------------------------
    n_total = len(results)
    n_synthesis_failed = sum(1 for r in results if r["mask_synthesis_failure"])
    n_synthesis_ok = n_total - n_synthesis_failed
    n_stage1_detected = sum(1 for r in results if r["stage1_detected"])
    n_stage1_missed = n_synthesis_ok - n_stage1_detected

    print(f"\n=== Stage 1 on MASKED images ===")
    print(f"Total: {n_total}  Mask-synthesis failures: {n_synthesis_failed} ({n_synthesis_failed/n_total*100:.2f}%)")
    print(f"Of {n_synthesis_ok} successfully-masked images: "
          f"detected={n_stage1_detected} ({n_stage1_detected/n_synthesis_ok*100:.2f}%)  "
          f"missed={n_stage1_missed} ({n_stage1_missed/n_synthesis_ok*100:.2f}%)")

    # ------------------------------------------------------------------
    # Stage 2 metrics: masked (detected subset) vs real unmasked baseline
    # ------------------------------------------------------------------
    detected_outcomes = [r for r in results if r["stage1_detected"]]
    masked_metrics = metrics_from_outcomes(detected_outcomes)

    print(f"\n=== Stage 2 (masked, detected subset) ===")
    print(f"n={masked_metrics['n']}  accuracy={masked_metrics['accuracy']*100:.2f}%  "
          f"APCER={masked_metrics['apcer']*100:.2f}%  BPCER={masked_metrics['bpcer']*100:.2f}%  "
          f"HTER={masked_metrics['hter']*100:.2f}%")

    unmasked_baseline = None
    if os.path.isfile(args.unmasked_baseline_json):
        with open(args.unmasked_baseline_json) as f:
            unmasked_baseline = json.load(f)
        ub = unmasked_baseline["metrics_on_detected_subset"]
        print(f"\n=== Stage 2 (unmasked baseline, same v2 checkpoint, same real pipeline) ===")
        print(f"n={unmasked_baseline['n_detected']}  accuracy={ub['accuracy']*100:.2f}%  "
              f"APCER={ub['apcer']*100:.2f}%  BPCER={ub['bpcer']*100:.2f}%  HTER={ub['hter']*100:.2f}%")

    bpcer_live_masked = masked_metrics["bpcer"]
    bpcer_flag = bpcer_live_masked is not None and bpcer_live_masked > 0.05
    if bpcer_flag:
        print(f"\n*** FLAG: masked-live BPCER = {bpcer_live_masked*100:.2f}% (> 5% threshold). "
              f"Per TASK I instructions: reporting only, NOT fixing. See masked_eval.json for a v3 "
              f"retraining-cost estimate. ***")

    # ------------------------------------------------------------------
    # Per-spoof-type breakdown for masked spoofs
    # ------------------------------------------------------------------
    by_type = {}
    for r in detected_outcomes:
        if r["true_label"] != config.LABEL_SPOOF:
            continue
        name = SPOOF_TYPE_NAMES.get(r["spoof_type"], f"unknown({r['spoof_type']})")
        by_type.setdefault(name, {"n": 0, "false_accepts": 0})
        by_type[name]["n"] += 1
        if r["pred_label"] == config.LABEL_LIVE:
            by_type[name]["false_accepts"] += 1
    per_type_apcer = {name: {"n": d["n"], "apcer": d["false_accepts"] / d["n"] if d["n"] else None}
                       for name, d in by_type.items()}

    print(f"\n=== Per-spoof-type APCER (masked, detected subset) ===")
    for name in sorted(per_type_apcer):
        d = per_type_apcer[name]
        print(f"  {name:<32} n={d['n']:<6} APCER={d['apcer']*100:.2f}%" if d['apcer'] is not None else f"  {name}: n/a")

    # ------------------------------------------------------------------
    # Gray-zone (v2's real band) applied to masked probabilities
    # ------------------------------------------------------------------
    band_low, band_high = load_gray_zone_band("v2")
    gz_decisions = []
    for r in detected_outcomes:
        label = classify_three_way(r["prob_spoof"], band_low, band_high, threshold=config.DECISION_THRESHOLD)
        gz_decisions.append({**r, "three_way_label": label})

    n_uncertain = sum(1 for d in gz_decisions if d["three_way_label"] == "UNCERTAIN")
    decided = [d for d in gz_decisions if d["three_way_label"] != "UNCERTAIN"]
    for d in decided:
        d["pred_label"] = config.LABEL_SPOOF if d["three_way_label"] == "SPOOF" else config.LABEL_LIVE
    gz_metrics = metrics_from_outcomes(decided)
    gz_coverage = len(decided) / len(gz_decisions) if gz_decisions else None

    print(f"\n=== Gray-zone (v2 band [{band_low:.4f}, {band_high:.4f}]) on masked images ===")
    print(f"Coverage: {gz_coverage*100:.2f}%  ({len(decided)}/{len(gz_decisions)} decided, {n_uncertain} UNCERTAIN)")
    if gz_metrics["n"]:
        print(f"Decided-set: accuracy={gz_metrics['accuracy']*100:.2f}%  APCER={gz_metrics['apcer']*100:.2f}%  "
              f"BPCER={gz_metrics['bpcer']*100:.2f}%  HTER={gz_metrics['hter']*100:.2f}%")

    # ------------------------------------------------------------------
    # v3 retraining-cost estimate (only ever an ESTIMATE, reported not acted on)
    # ------------------------------------------------------------------
    v3_estimate = None
    if bpcer_flag:
        v3_estimate = {
            "trigger": f"masked-live BPCER {bpcer_live_masked*100:.2f}% exceeds the 5% threshold",
            "what_it_would_involve": (
                "Augment the EXISTING cropped training set (data/celeba_spoof_cropped) with a synthetic-mask "
                "version of a fraction of live (and possibly spoof) training images, using this same "
                "synthetic_mask.py overlay, then retrain MobileNetV3-Small with the identical recipe used for "
                "v2 (15 epochs, same splits/seed/schedule) on the combined (masked + unmasked) set."
            ),
            "real_timing_basis": {
                "mask_synthesis_and_eval_on_3900_test_images": "see 'total_runtime_seconds' in this JSON",
                "v2_cropped_training_time_for_18200_images_15_epochs": "~31.2 min (see TASK G report)",
            },
            "rough_estimate_caveat": (
                "Not a measured number -- a rough estimate from the real timings above: building a masked "
                "augmented set over the 18,200-image training set at a similar per-image rate would add "
                "roughly 10-15 minutes of one-time preprocessing, and training itself would take about the "
                "same ~30 min as v2 (same recipe, same image count, masks don't change decode/augment cost "
                "materially) -- so order of 45min-1hr total, not counting evaluation. This is a rough estimate "
                "for planning only, not a committed number."
            ),
        }

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output = {
        "checkpoint": args.checkpoint,
        "mask_colors_cycled": colors,
        "n_total": n_total,
        "mask_synthesis_failures": n_synthesis_failed,
        "mask_synthesis_failure_rate": n_synthesis_failed / n_total,
        "stage1_on_masked": {
            "n_eligible": n_synthesis_ok,
            "n_detected": n_stage1_detected,
            "n_missed": n_stage1_missed,
            "detection_rate": n_stage1_detected / n_synthesis_ok if n_synthesis_ok else None,
        },
        "stage2_masked_detected_subset": masked_metrics,
        "stage2_unmasked_baseline": unmasked_baseline["metrics_on_detected_subset"] if unmasked_baseline else None,
        "bpcer_live_masked_flag": bpcer_flag,
        "v3_retraining_estimate_if_flagged": v3_estimate,
        "per_spoof_type_apcer_masked": per_type_apcer,
        "gray_zone_masked": {
            "band_low": band_low, "band_high": band_high,
            "coverage": gz_coverage, "n_uncertain": n_uncertain,
            "decided_metrics": gz_metrics,
        },
        "examples_saved": example_index,
        "total_runtime_seconds": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full results to: {args.out_json}")


if __name__ == "__main__":
    main()
