"""
train.py
--------
Trains the LivenessCNN (Live vs Spoof) on the CelebA-Spoof subset.

Example:
    python train.py \\
        --data_root ../data/celeba_spoof \\
        --backbone mobilenet_v3_small \\
        --epochs 15 --batch_size 32 --lr 3e-4 --image_size 224 \\
        --freeze_epochs 2

Expects --data_root to contain train/ and val/ subfolders, each with
live/ and spoof/ image folders (see dataset.py / prepare_celeba_subset.py).
"""

import argparse
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving plots
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import LivenessDataset, get_train_transform, get_eval_transform
from model import build_model


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def plot_history(history: dict, out_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the liveness (Live vs Spoof) classifier")
    parser.add_argument("--data_root", default=config.CELEBA_SPOOF_DIR,
                         help="Root folder containing train/ and val/ subfolders")
    parser.add_argument("--backbone", default=config.BACKBONE,
                         choices=["mobilenet_v3_small", "efficientnet_b0", "resnet18"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--freeze_epochs", type=int, default=2,
                         help="Number of initial epochs to train with a frozen backbone "
                              "(head-only warm-up) before full fine-tuning")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out_dir", default=None,
                         help="Checkpoint output dir. Default: models/ for the project's "
                              "default backbone (mobilenet_v3_small), or models/<backbone>/ "
                              "for any other backbone, so a comparison run can never "
                              "overwrite the existing default checkpoint.")
    parser.add_argument("--plots_dir", default=None,
                         help="Plots/history output dir. Same backbone-namespacing default "
                              "behavior as --out_dir.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    # Only the project's default backbone keeps writing to the original flat
    # models/ and outputs/plots/ paths (so every other script that hardcodes
    # config.BEST_MODEL_PATH etc. keeps working unchanged). Any other
    # backbone is automatically namespaced into its own subdirectory, so a
    # comparison run can never silently overwrite the default checkpoint.
    is_default_backbone = args.backbone == config.BACKBONE
    if args.out_dir is None:
        args.out_dir = config.MODELS_DIR if is_default_backbone else os.path.join(config.MODELS_DIR, args.backbone)
    if args.plots_dir is None:
        args.plots_dir = config.PLOTS_DIR if is_default_backbone else os.path.join(config.PLOTS_DIR, args.backbone)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    print(f"Backbone: {args.backbone}  |  checkpoints -> {args.out_dir}  |  plots -> {args.plots_dir}")

    device = config.DEVICE
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_ds = LivenessDataset(train_dir, transform=get_train_transform(args.image_size))
    val_ds = LivenessDataset(val_dir, transform=get_eval_transform(args.image_size))

    print(f"Train samples: {len(train_ds)}  {train_ds.class_counts()}")
    print(f"Val samples:   {len(val_ds)}  {val_ds.class_counts()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = build_model(backbone=args.backbone, pretrained=True).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed weights from {args.resume}")

    # Handle mild class imbalance with pos_weight (weights the SPOOF/positive class).
    counts = train_ds.class_counts()
    pos_weight = torch.tensor(
        [counts["live"] / max(1, counts["spoof"])], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Freeze/unfreeze schedule: warm up the head first, then fine-tune everything.
        if epoch == 1 and args.freeze_epochs > 0:
            model.freeze_backbone()
            print(f"Epoch {epoch}: backbone FROZEN (head-only warm-up)")
        elif epoch == args.freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)
            print(f"Epoch {epoch}: backbone UNFROZEN (full fine-tuning, lower LR)")

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        # Save "last" checkpoint every epoch.
        torch.save({
            "model_state_dict": model.state_dict(),
            "backbone": args.backbone,
            "image_size": args.image_size,
            "epoch": epoch,
            "val_acc": val_acc,
        }, os.path.join(args.out_dir, "last_liveness.pth"))

        # Track best checkpoint by validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "backbone": args.backbone,
                "image_size": args.image_size,
                "epoch": epoch,
                "val_acc": val_acc,
            }, os.path.join(args.out_dir, "best_liveness.pth"))
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    # Save training curves + history.
    plot_path = os.path.join(args.plots_dir, "training_curves.png")
    plot_history(history, plot_path)
    with open(os.path.join(args.plots_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
    print(f"Best model:    {os.path.join(args.out_dir, 'best_liveness.pth')}")
    print(f"Training plot: {plot_path}")


if __name__ == "__main__":
    main()
