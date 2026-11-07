"""
evaluate_v1_vs_v2.py
----------------------
TASK G step 5: compares v1 (trained on raw full frames, the original
mismatched setup) against v2 (trained on face crops built by
build_face_cropped_dataset.py), both evaluated on the SAME
deployment-style crops of the test set (data/celeba_spoof_cropped/test --
built with the exact crop_face/BlazeFace function the live pipeline uses),
so the comparison isolates the effect of what each model was TRAINED on,
not what it's being tested on.

Also reports v1's own raw-frame test accuracy for context (the number
that was reported before TASK F/G -- what v1 looks like evaluated the way
it always was, i.e. not an apples-to-apples deployment comparison).

Usage:
    python evaluate_v1_vs_v2.py
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import config
from dataset import LivenessDataset, get_eval_transform
from evaluate import collect_predictions, compute_anti_spoof_metrics
from evaluate_breakdown import SPOOF_TYPE_NAMES, load_official_labels, resolve_test_set_annotations
from model import build_model


def load_checkpoint(path: str, device):
    checkpoint = torch.load(path, map_location=device)
    backbone = checkpoint.get("backbone", config.BACKBONE)
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)
    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, backbone, image_size


def evaluate_on_cropped_test(model, image_size, data_root, device, batch_size=32):
    ds = LivenessDataset(os.path.join(data_root, "test"), transform=get_eval_transform(image_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    y_true, y_pred, y_prob = collect_predictions(model, loader, device, config.DECISION_THRESHOLD)
    metrics = compute_anti_spoof_metrics(y_true, y_pred)
    return metrics, ds, y_true, y_pred


def per_attack_type_apcer(ds, test_root, y_true, y_pred, raw_root, annotation_lookup_root):
    """Builds filepath->prediction lookup, then reuses
    resolve_test_set_annotations (with the cropped-dataset-aware
    annotation_lookup_root) to compute per-spoof-type APCER."""
    official_labels = load_official_labels(raw_root)
    records = resolve_test_set_annotations(
        test_root, raw_root, official_labels, annotation_lookup_root=annotation_lookup_root,
    )
    pred_by_path = {path: pred for (path, _), pred in zip(ds.samples, y_pred)}

    by_type = {}
    for rec in records:
        if rec["folder_label"] != config.LABEL_SPOOF:
            continue
        pred = pred_by_path.get(rec["filepath"])
        if pred is None:
            continue
        type_name = SPOOF_TYPE_NAMES.get(rec["spoof_type"], f"unknown({rec['spoof_type']})")
        by_type.setdefault(type_name, {"n": 0, "false_accepts": 0})
        by_type[type_name]["n"] += 1
        if pred == config.LABEL_LIVE:
            by_type[type_name]["false_accepts"] += 1

    return {
        name: {"n": d["n"], "apcer": d["false_accepts"] / d["n"] if d["n"] else None}
        for name, d in by_type.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Compare v1 (raw-frame) vs v2 (face-cropped) training, both on deployment-style test crops")
    parser.add_argument("--v1_checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--v2_checkpoint", default=os.path.join(config.MODELS_DIR, "v2_cropped", "best_liveness.pth"))
    parser.add_argument("--cropped_data_root", default=os.path.join(config.PROJECT_ROOT, "data", "celeba_spoof_cropped"))
    parser.add_argument("--raw_root", default=os.path.join(config.PROJECT_ROOT, "raw_downloads", "CelebA_Spoof", "CelebA_Spoof"))
    parser.add_argument("--annotation_lookup_root", default=os.path.join(config.PROJECT_ROOT, "data", "celeba_spoof", "test"))
    parser.add_argument("--out_json", default=os.path.join(config.PROJECT_ROOT, "results", "v1_vs_v2.json"))
    args = parser.parse_args()

    for path in (args.v1_checkpoint, args.v2_checkpoint):
        if not os.path.isfile(path):
            raise SystemExit(f"Checkpoint not found: {path}")

    device = config.DEVICE
    print(f"Using device: {device}")

    print(f"\nLoading v1 (raw-frame-trained): {args.v1_checkpoint}")
    v1_model, v1_backbone, v1_image_size = load_checkpoint(args.v1_checkpoint, device)
    print(f"Loading v2 (face-crop-trained): {args.v2_checkpoint}")
    v2_model, v2_backbone, v2_image_size = load_checkpoint(args.v2_checkpoint, device)

    print(f"\nEvaluating v1 on deployment-style test crops ({args.cropped_data_root}/test) ...")
    v1_metrics, v1_ds, v1_true, v1_pred = evaluate_on_cropped_test(v1_model, v1_image_size, args.cropped_data_root, device)
    print(f"Evaluating v2 on the SAME deployment-style test crops ...")
    v2_metrics, v2_ds, v2_true, v2_pred = evaluate_on_cropped_test(v2_model, v2_image_size, args.cropped_data_root, device)

    print(f"\nComputing per-attack-type APCER for both ...")
    test_root = os.path.join(args.cropped_data_root, "test")
    v1_by_type = per_attack_type_apcer(v1_ds, test_root, v1_true, v1_pred, args.raw_root, args.annotation_lookup_root)
    v2_by_type = per_attack_type_apcer(v2_ds, test_root, v2_true, v2_pred, args.raw_root, args.annotation_lookup_root)

    print("\n=== v1 (raw-frame-trained) vs v2 (face-crop-trained) ===")
    print("Both evaluated on the SAME deployment-style test crops:\n")
    print(f"{'Metric':<12}{'v1':<12}{'v2':<12}")
    for key in ("accuracy", "apcer", "bpcer", "hter"):
        print(f"{key:<12}{v1_metrics[key]*100:<12.2f}{v2_metrics[key]*100:<12.2f}")

    print(f"\n{'Attack type':<32}{'v1 APCER':<12}{'v2 APCER':<12}{'n':<6}")
    for name in sorted(set(v1_by_type) | set(v2_by_type)):
        v1a = v1_by_type.get(name, {}).get("apcer")
        v2a = v2_by_type.get(name, {}).get("apcer")
        n = v1_by_type.get(name, {}).get("n") or v2_by_type.get(name, {}).get("n")
        v1_str = f"{v1a*100:.2f}%" if v1a is not None else "n/a"
        v2_str = f"{v2a*100:.2f}%" if v2a is not None else "n/a"
        print(f"{name:<32}{v1_str:<12}{v2_str:<12}{n:<6}")

    output = {
        "v1_checkpoint": args.v1_checkpoint,
        "v2_checkpoint": args.v2_checkpoint,
        "evaluation_data": f"{args.cropped_data_root}/test (both models, same deployment-style crops)",
        "v1_metrics_on_deployment_crops": v1_metrics,
        "v2_metrics_on_deployment_crops": v2_metrics,
        "v1_per_attack_type_apcer": v1_by_type,
        "v2_per_attack_type_apcer": v2_by_type,
        "note_v1_raw_frame_baseline_accuracy": 0.9915384615384616,
        "note_v1_on_deployment_crops_prior_measurement": {
            "accuracy": 0.9594, "apcer": 0.0610, "bpcer": 0.0206, "hter": 0.0408,
            "source": "results/deployment_crop_eval.json (TASK F, live BlazeFace detection on raw frames)",
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved comparison to: {args.out_json}")


if __name__ == "__main__":
    main()
