"""
benchmark_backbones.py
------------------------
TASK L: real latency/size benchmark for all three cropped-data (v2)
backbones -- parameter count, checkpoint file size, per-crop inference
latency on MPS and CPU-only (warm-up, then median of 100 single-sample
forward passes), and Grad-CAM time on MPS.

MobileNetV3-Small stays the deployed model regardless of these numbers --
this script only measures and reports, doesn't change anything.

Usage:
    python benchmark_backbones.py
"""

import json
import os
import statistics
import time

import torch

import config
from gradcam import GradCAM
from model import build_model

BACKBONES = {
    "mobilenet_v3_small": os.path.join(config.MODELS_DIR, "v2_cropped", "best_liveness.pth"),
    "resnet18": os.path.join(config.MODELS_DIR, "v2_cropped", "resnet18", "best_liveness.pth"),
    "efficientnet_b0": os.path.join(config.MODELS_DIR, "v2_cropped", "efficientnet_b0", "best_liveness.pth"),
}

N_WARMUP = 10
N_TIMED = 100
N_GRADCAM_WARMUP = 3
N_GRADCAM_TIMED = 20

# features[index] used by GradCAM, per backbone -- the last conv block
# before the classifier head, same index-selection logic used for
# mobilenet_v3_small throughout this project.
GRADCAM_LAYER_GETTER = {
    "mobilenet_v3_small": lambda m: m.feature_extractor.features[-1],
    "resnet18": lambda m: m.feature_extractor.layer4[-1],
    "efficientnet_b0": lambda m: m.feature_extractor.features[-1],
}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model(checkpoint_path: str, backbone: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def median_latency_ms(model, device: torch.device, image_size: int = 224) -> float:
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def median_gradcam_ms(model, target_layer, device: torch.device, image_size: int = 224) -> float:
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with GradCAM(model, target_layer) as cam:
        for _ in range(N_GRADCAM_WARMUP):
            cam(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        times = []
        for _ in range(N_GRADCAM_TIMED):
            t0 = time.perf_counter()
            cam(dummy)
            if device.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    mps_available = torch.backends.mps.is_available()
    results = {}

    for backbone, checkpoint_path in BACKBONES.items():
        if not os.path.isfile(checkpoint_path):
            print(f"SKIP {backbone}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n=== {backbone} ===")
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

        cpu_device = torch.device("cpu")
        model_cpu = load_model(checkpoint_path, backbone, cpu_device)
        n_params = count_parameters(model_cpu)
        latency_cpu = median_latency_ms(model_cpu, cpu_device)
        print(f"  params={n_params:,}  file_size={file_size_mb:.2f}MB  CPU latency={latency_cpu:.2f}ms (median of {N_TIMED})")

        latency_mps = None
        gradcam_mps = None
        if mps_available:
            mps_device = torch.device("mps")
            model_mps = load_model(checkpoint_path, backbone, mps_device)
            latency_mps = median_latency_ms(model_mps, mps_device)
            target_layer = GRADCAM_LAYER_GETTER[backbone](model_mps)
            gradcam_mps = median_gradcam_ms(model_mps, target_layer, mps_device)
            print(f"  MPS latency={latency_mps:.2f}ms (median of {N_TIMED})  "
                  f"Grad-CAM MPS={gradcam_mps:.2f}ms (median of {N_GRADCAM_TIMED})")
        else:
            print("  MPS not available on this machine -- skipping MPS/Grad-CAM timing.")

        results[backbone] = {
            "checkpoint": checkpoint_path,
            "param_count": n_params,
            "model_file_size_mb": file_size_mb,
            "latency_cpu_ms_median": latency_cpu,
            "latency_mps_ms_median": latency_mps,
            "gradcam_mps_ms_median": gradcam_mps,
            "n_timed_runs": N_TIMED,
            "n_gradcam_timed_runs": N_GRADCAM_TIMED,
            "batch_size": 1,
            "image_size": 224,
        }

    out_path = os.path.join(config.PROJECT_ROOT, "results", "backbone_latency.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
