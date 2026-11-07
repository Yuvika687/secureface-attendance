"""
baseline_opencv.py
--------------------
Optional simple baseline for the report: classical (non-deep-learning)
anti-spoofing using OpenCV's Laplacian-variance "blurriness" heuristic.

Idea: printed-photo and screen-replay attacks are usually slightly blurrier
and have different high-frequency texture statistics than a real face
photographed directly, because they are a "photo of a photo/screen". We
compute the variance of the Laplacian of the grayscale face crop as a crude
texture-sharpness score and threshold it.

This is intentionally simple -- its purpose is to give the report a
concrete comparison point ("classical baseline vs. our CNN"), not to be a
competitive anti-spoofing method by itself.

Usage:
    python baseline_opencv.py --data_root ../data/celeba_spoof/test
"""

import argparse
import os

import cv2
import numpy as np
from sklearn.metrics import roc_curve

import config

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def laplacian_variance_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_scores(root: str):
    scores, labels = [], []
    for class_name, label in (("live", config.LABEL_LIVE), ("spoof", config.LABEL_SPOOF)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(VALID_EXTENSIONS):
                continue
            image = cv2.imread(os.path.join(class_dir, fname))
            if image is None:
                continue
            scores.append(laplacian_variance_score(image))
            labels.append(label)
    return np.array(scores), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Classical OpenCV blur-based baseline for anti-spoofing")
    parser.add_argument("--data_root", required=True, help="Folder with live/ and spoof/ subfolders")
    args = parser.parse_args()

    scores, labels = load_scores(args.data_root)
    if len(scores) == 0:
        print(f"No images found under {args.data_root}")
        return

    # Lower sharpness (Laplacian variance) => more likely SPOOF.
    # We search thresholds to find the best-separating point (best accuracy),
    # purely for reporting a baseline number.
    fpr, tpr, thresholds = roc_curve(labels, -scores)  # negate so higher = more "spoof-like"
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = -thresholds[best_idx]

    preds = (scores < best_threshold).astype(int)  # 1 = spoof if sharpness below threshold
    accuracy = float(np.mean(preds == labels))

    spoof_mask = labels == config.LABEL_SPOOF
    live_mask = labels == config.LABEL_LIVE
    apcer = float(np.mean(preds[spoof_mask] == config.LABEL_LIVE)) if spoof_mask.sum() else float("nan")
    bpcer = float(np.mean(preds[live_mask] == config.LABEL_SPOOF)) if live_mask.sum() else float("nan")

    print("=" * 50)
    print("CLASSICAL BASELINE: Laplacian-variance sharpness heuristic")
    print("=" * 50)
    print(f"Best threshold (Laplacian variance): {best_threshold:.2f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"APCER   : {apcer*100:.2f}%")
    print(f"BPCER   : {bpcer*100:.2f}%")
    print("=" * 50)
    print("Use this as a comparison row against the CNN model's metrics in your report.")


if __name__ == "__main__":
    main()
