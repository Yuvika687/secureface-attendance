"""
benchmark_onnx.py
--------------------
TASK O: measures real CPU latency under ONNX Runtime for all three v2
(cropped-data) backbones, and verifies each exported ONNX model's output
actually matches its source PyTorch checkpoint (max abs diff over 20 real
test crops) before trusting any latency number from it.

Does NOT change the deployed model -- MobileNetV3-Small (PyTorch) stays
active; this is a measurement only.

Usage:
    python export_onnx.py --checkpoint ../models/v2_cropped/best_liveness.pth --out ../models/v2_cropped/liveness.onnx
    python export_onnx.py --checkpoint ../models/v2_cropped/resnet18/best_liveness.pth --out ../models/v2_cropped/resnet18/liveness.onnx
    python export_onnx.py --checkpoint ../models/v2_cropped/efficientnet_b0/best_liveness.pth --out ../models/v2_cropped/efficientnet_b0/liveness.onnx
    python benchmark_onnx.py
"""

import csv
import json
import os
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

import config
from build_summary import load_json
from dataset import get_eval_transform
from model import build_model

N_WARMUP = 10
N_TIMED = 100
N_ACCURACY_CHECK_IMAGES = 20

BACKBONES = {
    "mobilenet_v3_small": {
        "pth": os.path.join(config.MODELS_DIR, "v2_cropped", "best_liveness.pth"),
        "onnx": os.path.join(config.MODELS_DIR, "v2_cropped", "liveness.onnx"),
    },
    "resnet18": {
        "pth": os.path.join(config.MODELS_DIR, "v2_cropped", "resnet18", "best_liveness.pth"),
        "onnx": os.path.join(config.MODELS_DIR, "v2_cropped", "resnet18", "liveness.onnx"),
    },
    "efficientnet_b0": {
        "pth": os.path.join(config.MODELS_DIR, "v2_cropped", "efficientnet_b0", "best_liveness.pth"),
        "onnx": os.path.join(config.MODELS_DIR, "v2_cropped", "efficientnet_b0", "liveness.onnx"),
    },
}


def load_check_images(n: int):
    """Real crops from the cropped test set, evenly spread across the
    file, for the ONNX-vs-PyTorch output parity check (not random noise
    -- real inputs the model actually sees)."""
    csv_path = os.path.join(config.PREDICTIONS_DIR, "v2_cropped", "predictions_v2_cropped_test.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    step = max(1, len(rows) // n)
    paths = [rows[i]["filepath"] for i in range(0, len(rows), step)][:n]
    return paths


def verify_onnx_matches_pytorch(backbone: str, pth_path: str, onnx_path: str, image_paths, image_size: int) -> float:
    checkpoint = torch.load(pth_path, map_location="cpu")
    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    transform = get_eval_transform(image_size)

    max_abs_diff = 0.0
    with torch.no_grad():
        for path in image_paths:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            from PIL import Image as PILImage
            pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            tensor = transform(pil).unsqueeze(0)

            pytorch_logit = model(tensor).item()
            onnx_logit = session.run(None, {"face_crop": tensor.numpy()})[0].item()
            max_abs_diff = max(max_abs_diff, abs(pytorch_logit - onnx_logit))

    return max_abs_diff


def median_onnx_latency_ms(onnx_path: str, image_size: int) -> float:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(N_WARMUP):
        session.run(None, {"face_crop": dummy})

    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        session.run(None, {"face_crop": dummy})
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    image_paths = load_check_images(N_ACCURACY_CHECK_IMAGES)
    print(f"Loaded {len(image_paths)} real test crops for the ONNX-vs-PyTorch parity check.\n")

    results = {}
    for backbone, paths in BACKBONES.items():
        if not os.path.isfile(paths["onnx"]):
            print(f"SKIP {backbone}: {paths['onnx']} not found -- run export_onnx.py for it first.")
            continue

        print(f"=== {backbone} ===")
        max_diff = verify_onnx_matches_pytorch(backbone, paths["pth"], paths["onnx"], image_paths, config.IMAGE_SIZE)
        print(f"  Max abs diff (ONNX vs PyTorch logit, n={len(image_paths)} real crops): {max_diff:.6f}")

        latency = median_onnx_latency_ms(paths["onnx"], config.IMAGE_SIZE)
        print(f"  ONNX Runtime CPU latency: {latency:.2f}ms (median of {N_TIMED})")

        results[backbone] = {
            "onnx_path": paths["onnx"],
            "max_abs_diff_vs_pytorch": max_diff,
            "n_parity_check_images": len(image_paths),
            "cpu_onnx_ms_median": latency,
            "n_timed_runs": N_TIMED,
            "batch_size": 1,
            "image_size": config.IMAGE_SIZE,
        }

    if results:
        fastest = min(results, key=lambda b: results[b]["cpu_onnx_ms_median"])
        print(f"\nFastest on CPU under ONNX Runtime: {fastest} ({results[fastest]['cpu_onnx_ms_median']:.2f}ms)")
        print(f"MobileNetV3-Small is fastest: {fastest == 'mobilenet_v3_small'}")

    # Merge into results/backbone_latency.json (produced by benchmark_backbones.py,
    # TASK L) rather than overwriting it, so summary.json's existing
    # latency_and_size section just gains a "cpu_onnx_ms" field per backbone.
    latency_path = os.path.join(config.PROJECT_ROOT, "results", "backbone_latency.json")
    existing = load_json(latency_path) or {}
    for backbone, r in results.items():
        existing.setdefault(backbone, {})
        existing[backbone]["cpu_onnx_ms"] = r["cpu_onnx_ms_median"]
        existing[backbone]["onnx_max_abs_diff_vs_pytorch"] = r["max_abs_diff_vs_pytorch"]
        existing[backbone]["onnx_parity_check_n"] = r["n_parity_check_images"]
    with open(latency_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nMerged ONNX results into: {latency_path}")
    print("Run build_summary.py again to refresh results/summary.json with these numbers.")


if __name__ == "__main__":
    main()
