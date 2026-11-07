"""
evaluate.py
-----------
Evaluates a trained liveness checkpoint on a held-out test set using proper
anti-spoofing metrics (not just accuracy):

    - Accuracy
    - Confusion matrix
    - APCER (Attack Presentation Classification Error Rate)
        = fraction of SPOOF samples wrongly classified as LIVE
        (this is the dangerous error: an attacker gets in)
    - BPCER (Bona Fide Presentation Classification Error Rate)
        = fraction of LIVE samples wrongly classified as SPOOF
        (this is the annoying error: a real user gets rejected)
    - HTER  = (APCER + BPCER) / 2
    - Per-class precision / recall / F1

Usage:
    # Evaluate on the CelebA-Spoof held-out test split
    python evaluate.py --data_root ../data/celeba_spoof/test --checkpoint ../models/best_liveness.pth

    # Evaluate generalization on CASIA-FASD
    python evaluate.py --data_root ../data/casia_fasd/test --checkpoint ../models/best_liveness.pth --tag casia_fasd
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import LivenessDataset, get_eval_transform
from model import build_model


@torch.no_grad()
def collect_predictions(model, loader, device, threshold: float):
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="evaluate"):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()  # P(spoof)
        preds = (probs >= threshold).astype(int)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_anti_spoof_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true / y_pred use config.LABEL_LIVE (0) / config.LABEL_SPOOF (1).

    APCER: among actual SPOOF (1) samples, fraction predicted LIVE (0).
    BPCER: among actual LIVE (0) samples, fraction predicted SPOOF (1).
    """
    spoof_mask = y_true == config.LABEL_SPOOF
    live_mask = y_true == config.LABEL_LIVE

    n_spoof = spoof_mask.sum()
    n_live = live_mask.sum()

    apcer = float(np.mean(y_pred[spoof_mask] == config.LABEL_LIVE)) if n_spoof > 0 else float("nan")
    bpcer = float(np.mean(y_pred[live_mask] == config.LABEL_SPOOF)) if n_live > 0 else float("nan")
    hter = (apcer + bpcer) / 2 if n_spoof > 0 and n_live > 0 else float("nan")

    accuracy = float(np.mean(y_true == y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[config.LABEL_LIVE, config.LABEL_SPOOF], zero_division=0
    )

    return {
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "hter": hter,
        "n_live": int(n_live),
        "n_spoof": int(n_spoof),
        "precision_live": float(precision[0]),
        "recall_live": float(recall[0]),
        "f1_live": float(f1[0]),
        "precision_spoof": float(precision[1]),
        "recall_spoof": float(recall[1]),
        "f1_spoof": float(f1[1]),
    }


def plot_confusion_matrix(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=[config.LABEL_LIVE, config.LABEL_SPOOF])
    labels = ["LIVE", "SPOOF"]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def print_report(metrics: dict, cm: np.ndarray):
    print("\n" + "=" * 50)
    print("LIVENESS EVALUATION REPORT")
    print("=" * 50)
    print(f"Samples: LIVE={metrics['n_live']}  SPOOF={metrics['n_spoof']}")
    print("-" * 50)
    print(f"Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"APCER    : {metrics['apcer']*100:.2f}%  (spoof wrongly accepted as live)")
    print(f"BPCER    : {metrics['bpcer']*100:.2f}%  (live wrongly rejected as spoof)")
    print(f"HTER     : {metrics['hter']*100:.2f}%")
    print("-" * 50)
    print(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    print(f"{'LIVE':<8}{metrics['precision_live']:<12.4f}{metrics['recall_live']:<12.4f}{metrics['f1_live']:<10.4f}")
    print(f"{'SPOOF':<8}{metrics['precision_spoof']:<12.4f}{metrics['recall_spoof']:<12.4f}{metrics['f1_spoof']:<10.4f}")
    print("-" * 50)
    print("Confusion Matrix (rows=actual, cols=predicted) [LIVE, SPOOF]:")
    print(cm)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the liveness classifier with anti-spoofing metrics")
    parser.add_argument("--data_root", required=True, help="Folder with live/ and spoof/ subfolders")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD,
                         help="Decision threshold on P(spoof); >= threshold -> SPOOF")
    parser.add_argument("--tag", default="test", help="Name used for output plot/report filenames")
    parser.add_argument("--plots_dir", default=config.PLOTS_DIR)
    parser.add_argument("--predictions_dir", default=config.PREDICTIONS_DIR)
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("backbone", config.BACKBONE)
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)

    model = build_model(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (backbone={backbone}, image_size={image_size}, "
          f"trained_val_acc={checkpoint.get('val_acc', 'n/a')})")

    ds = LivenessDataset(args.data_root, transform=get_eval_transform(image_size))
    print(f"Test samples: {len(ds)}  {ds.class_counts()}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    y_true, y_pred, y_prob = collect_predictions(model, loader, device, args.threshold)
    metrics = compute_anti_spoof_metrics(y_true, y_pred)

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.predictions_dir, exist_ok=True)

    cm_path = os.path.join(args.plots_dir, f"confusion_matrix_{args.tag}.png")
    cm = plot_confusion_matrix(y_true, y_pred, cm_path)

    print_report(metrics, cm)
    print(f"\nConfusion matrix plot saved to: {cm_path}")

    # Save raw predictions + summary metrics for the report.
    metrics_path = os.path.join(args.predictions_dir, f"metrics_{args.tag}.json")
    with open(metrics_path, "w") as f:
        json.dump({"threshold": args.threshold, **metrics}, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_path}")

    preds_path = os.path.join(args.predictions_dir, f"predictions_{args.tag}.csv")
    with open(preds_path, "w") as f:
        f.write("filepath,true_label,pred_label,prob_spoof\n")
        for (path, true_label), pred, prob in zip(ds.samples, y_pred, y_prob):
            f.write(f"{path},{true_label},{pred},{prob:.4f}\n")
    print(f"Per-sample predictions saved to: {preds_path}")


if __name__ == "__main__":
    main()
