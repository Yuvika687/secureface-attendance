"""
evaluate_real_world.py
------------------------
Runs the REAL, full detect -> crop -> classify pipeline (the same
face_utils.FaceDetector + liveness_infer.LivenessPredictor used by
realtime_pipeline.py and the webapp) against your own recorded real-world
test videos, and reports real per-attack-type APCER/BPCER/HTER.

This exists because the CelebA-Spoof test set (99.15% accuracy) only
proves the model works on CelebA-Spoof's own capture distribution. It does
NOT prove the model works on your actual webcam, your actual room, your
actual phone/printer. This script measures that, honestly, whatever the
numbers turn out to be.

Expected folder layout -- one subfolder per attack type, containing short
video recordings (.mp4/.mov/.avi) or still photos (.jpg/.png) of that
condition:

    real_world_test/
        live_no_mask/        <- you, unmasked, genuine
        live_with_mask/      <- you, wearing a mask, genuine
        printed_photo/       <- a printed photo of a face, held up to camera
        printed_photo_masked/<- a printed photo of a masked face
        phone_photo_replay/  <- a photo displayed on a phone screen
        phone_video_replay/  <- a video played on a phone screen
        screen_replay/       <- a laptop/monitor screen showing a face

Folders are matched by name; "live" in the folder name = genuine (ground
truth LIVE), anything else = attack (ground truth SPOOF). Vary lighting
and distance across your recordings within each folder for a more
representative sample.

Usage:
    python evaluate_real_world.py --data_root ../real_world_test
    # uses config.ACTIVE_MODEL_PATH (v2 by default) and the real v2
    # gray-zone band (results/gray_zone_v2.json) for the three-way
    # LIVE/SPOOF/UNCERTAIN decision -- same as the live webapp. Pass
    # --checkpoint to evaluate a specific checkpoint instead.
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

import config
from face_utils import FaceDetector, crop_face
from liveness_infer import LivenessPredictor, classify_three_way, load_gray_zone_band

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_genuine_folder(folder_name: str) -> bool:
    """Ground truth: a folder is LIVE only if its name starts with 'live'."""
    return folder_name.lower().startswith("live")


def iter_frames_from_video(path: str, sample_every: int = 5):
    """Yields every Nth frame from a video file to avoid over-sampling
    near-identical consecutive frames."""
    cap = cv2.VideoCapture(path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            yield frame
        idx += 1
    cap.release()


def collect_samples(folder_path: str):
    """Yields (frame, source_filename) for every video/image in a folder."""
    for fname in sorted(os.listdir(folder_path)):
        full = os.path.join(folder_path, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            for frame in iter_frames_from_video(full):
                yield frame, fname
        elif ext in IMAGE_EXTENSIONS:
            frame = cv2.imread(full)
            if frame is not None:
                yield frame, fname


def main():
    parser = argparse.ArgumentParser(description="Evaluate the real pipeline on real-world recordings")
    parser.add_argument("--data_root", required=True,
                         help="Folder containing one subfolder per attack type (see module docstring)")
    parser.add_argument("--checkpoint", default=config.ACTIVE_MODEL_PATH,
                         help="Defaults to config.ACTIVE_MODEL_PATH (v2). Set LIVENESS_MODEL_VERSION=v1 "
                              "to default to v1 instead, or pass an explicit path to override either.")
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    parser.add_argument("--gray_zone_version", default=config.MODEL_VERSION,
                         help="Which model version's gray-zone band (results/gray_zone_<version>.json) "
                              "to use for the three-way decision. Defaults to the active model version.")
    parser.add_argument("--sample_every", type=int, default=5,
                         help="Sample every Nth frame from videos (avoids near-duplicate consecutive frames)")
    parser.add_argument("--out_csv", default=os.path.join(config.PREDICTIONS_DIR, "real_world_predictions.csv"))
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise SystemExit(f"--data_root not found: {args.data_root}")

    detector = FaceDetector()
    predictor = LivenessPredictor(checkpoint_path=args.checkpoint, threshold=args.threshold)

    try:
        band_low, band_high = load_gray_zone_band(args.gray_zone_version)
        print(f"Three-way decision ENABLED, gray-zone band for {args.gray_zone_version}: [{band_low:.4f}, {band_high:.4f}]")
    except FileNotFoundError as e:
        band_low = band_high = None
        print(f"WARNING: {e} -- three-way decision disabled, falling back to plain LIVE/SPOOF.")

    # per-attack-type: list of (predicted_label, true_label, three_way_label)
    results_by_type = defaultdict(list)
    rows = []
    n_no_face = 0

    attack_folders = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    ])
    if not attack_folders:
        raise SystemExit(f"No subfolders found under {args.data_root}. See the module docstring for expected layout.")

    print(f"Found {len(attack_folders)} attack-type folders: {attack_folders}\n")

    for folder_name in attack_folders:
        folder_path = os.path.join(args.data_root, folder_name)
        true_label = config.LABEL_LIVE if is_genuine_folder(folder_name) else config.LABEL_SPOOF
        true_label_name = "LIVE" if true_label == config.LABEL_LIVE else "SPOOF"

        n_processed = 0
        for frame, source_file in collect_samples(folder_path):
            detections = detector.detect(frame)
            if not detections:
                n_no_face += 1
                continue

            best = max(detections, key=lambda d: d.width * d.height)
            crop = crop_face(frame, best)
            if crop is None or crop.size == 0:
                n_no_face += 1
                continue

            prob_spoof = predictor.predict_prob(crop)
            pred_label = config.LABEL_SPOOF if prob_spoof >= predictor.threshold else config.LABEL_LIVE
            label = "SPOOF" if pred_label == config.LABEL_SPOOF else "LIVE"
            confidence = prob_spoof if label == "SPOOF" else 1.0 - prob_spoof

            if band_low is not None:
                three_way_label = classify_three_way(prob_spoof, band_low, band_high, threshold=predictor.threshold)
            else:
                three_way_label = label

            results_by_type[folder_name].append((pred_label, true_label, three_way_label))
            rows.append({
                "attack_type": folder_name,
                "source_file": source_file,
                "true_label": true_label_name,
                "pred_label": label,
                "three_way_label": three_way_label,
                "confidence": f"{confidence:.4f}",
                "prob_spoof": f"{prob_spoof:.4f}",
            })
            n_processed += 1

        print(f"[{folder_name}] ground truth = {true_label_name}, {n_processed} frames with a detected face processed")

    detector.close()

    if n_no_face:
        print(f"\n{n_no_face} frames had no detectable face and were skipped (not counted in metrics).")

    # Write per-sample predictions CSV.
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack_type", "source_file", "true_label", "pred_label",
                                                "three_way_label", "confidence", "prob_spoof"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample predictions saved to: {args.out_csv}")

    # Per-attack-type metrics: plain-threshold (all samples) alongside the
    # three-way decided-subset (excludes UNCERTAIN) plus coverage.
    print("\n" + "=" * 90)
    print("REAL-WORLD EVALUATION -- BY ATTACK TYPE")
    print("=" * 90)
    print(f"{'Attack type':<22}{'N':<6}{'True':<7}{'Accuracy':<11}{'Error rate':<16}{'Coverage':<11}{'Decided err rate'}")
    print("-" * 90)

    all_preds, all_trues, all_three_way = [], [], []
    for folder_name in attack_folders:
        triples = results_by_type.get(folder_name, [])
        if not triples:
            print(f"{folder_name:<22}{'0':<6}{'--':<7}{'no samples'}")
            continue
        preds = np.array([p for p, t, tw in triples])
        trues = np.array([t for p, t, tw in triples])
        three_way = [tw for p, t, tw in triples]
        all_preds.extend(preds.tolist())
        all_trues.extend(trues.tolist())
        all_three_way.extend(three_way)

        is_genuine = is_genuine_folder(folder_name)
        acc = float(np.mean(preds == trues))
        if is_genuine:
            error_rate = float(np.mean(preds == config.LABEL_SPOOF))  # BPCER contribution
            error_name = "BPCER"
        else:
            error_rate = float(np.mean(preds == config.LABEL_LIVE))  # APCER contribution
            error_name = "APCER"

        decided_mask = np.array([tw != "UNCERTAIN" for tw in three_way])
        coverage = float(decided_mask.mean())
        if decided_mask.sum():
            decided_preds, decided_trues = preds[decided_mask], trues[decided_mask]
            if is_genuine:
                decided_err = float(np.mean(decided_preds == config.LABEL_SPOOF))
            else:
                decided_err = float(np.mean(decided_preds == config.LABEL_LIVE))
            decided_err_str = f"{error_name}={decided_err*100:.2f}%"
        else:
            decided_err_str = "n/a (all UNCERTAIN)"

        print(f"{folder_name:<22}{len(triples):<6}{'LIVE' if is_genuine else 'SPOOF':<7}"
              f"{acc*100:>6.2f}%    {error_name}={error_rate*100:.2f}%     {coverage*100:>6.2f}%    {decided_err_str}")

    print("-" * 90)

    if all_preds:
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        spoof_mask = all_trues == config.LABEL_SPOOF
        live_mask = all_trues == config.LABEL_LIVE

        overall_apcer = float(np.mean(all_preds[spoof_mask] == config.LABEL_LIVE)) if spoof_mask.sum() else float("nan")
        overall_bpcer = float(np.mean(all_preds[live_mask] == config.LABEL_SPOOF)) if live_mask.sum() else float("nan")
        overall_hter = (overall_apcer + overall_bpcer) / 2
        overall_acc = float(np.mean(all_preds == all_trues))

        print(f"\nOVERALL, plain threshold (pooled across all real-world folders, N={len(all_preds)}):")
        print(f"  Accuracy : {overall_acc*100:.2f}%")
        print(f"  APCER    : {overall_apcer*100:.2f}%  (pooled across all attack-type folders)")
        print(f"  BPCER    : {overall_bpcer*100:.2f}%  (pooled across all live folders)")
        print(f"  HTER     : {overall_hter*100:.2f}%")

        decided_mask = np.array([tw != "UNCERTAIN" for tw in all_three_way])
        coverage = float(decided_mask.mean())
        n_uncertain = int((~decided_mask).sum())
        print(f"\nOVERALL, three-way decision (coverage={coverage*100:.2f}%, "
              f"{n_uncertain}/{len(all_preds)} UNCERTAIN):")
        if decided_mask.sum():
            dp, dt = all_preds[decided_mask], all_trues[decided_mask]
            d_spoof, d_live = dt == config.LABEL_SPOOF, dt == config.LABEL_LIVE
            d_apcer = float(np.mean(dp[d_spoof] == config.LABEL_LIVE)) if d_spoof.sum() else float("nan")
            d_bpcer = float(np.mean(dp[d_live] == config.LABEL_SPOOF)) if d_live.sum() else float("nan")
            d_hter = (d_apcer + d_bpcer) / 2
            d_acc = float(np.mean(dp == dt))
            print(f"  Accuracy : {d_acc*100:.2f}%  (decided subset only)")
            print(f"  APCER    : {d_apcer*100:.2f}%")
            print(f"  BPCER    : {d_bpcer*100:.2f}%")
            print(f"  HTER     : {d_hter*100:.2f}%")
        else:
            print("  (every sample fell in UNCERTAIN -- no decided-subset metrics)")

        print(f"\nCompare this honestly against v2's own real deployment-crop numbers on "
              f"CelebA-Spoof (accuracy 99.04%, HTER 0.97%, see results/deployment_crop_eval_v2.json) "
              f"-- a real gap here is expected and is exactly what this script exists to measure, not hide.")
    print("=" * 90)


if __name__ == "__main__":
    main()
