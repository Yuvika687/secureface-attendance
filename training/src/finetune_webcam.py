"""
Fine-tune the deployed liveness model (v2) on webcam crops from Collection mode.

    ./venv/bin/python training/src/finetune_webcam.py              # real run
    ./venv/bin/python training/src/finetune_webcam.py --epochs 1 --dry_run   # pipeline check only

Rules (docs/domain_adaptation_plan.md):
  * Trains ONLY on session-1 crops. Session-2 crops are never loaded here.
  * Epoch selection uses session-1 people held out from training (person-grouped,
    never splitting one person's crops) plus the original CelebA-Spoof *validation*
    split -- never session 2 and never the 3,900-image test set.
  * Starts from liveness/weights/best_liveness.pth with a low learning rate, BatchNorm
    statistics frozen, and each epoch mixes in an equal number of original training
    crops (balanced live/spoof) so the model doesn't forget the dataset.
  * Saves to liveness/weights/v3_webcam/best_liveness_v3_webcam.pth. The app keeps
    using v2 (liveness/weights/best_liveness.pth) until the owner approves a switch.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from webcam_data import (REPO, MANIFEST, V2_CHECKPOINT, V3_CHECKPOINT, V3_DIR, ImageList,  # noqa: E402
                         dataset_items, get_train_transform, metrics, predict, read_manifest)

import torch  # noqa: E402
from liveness.config import DEVICE  # noqa: E402
from liveness.model import build_model  # noqa: E402


def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dataset_mix", type=float, default=1.0,
                    help="original training crops per webcam crop, each epoch")
    ap.add_argument("--val_people", default="",
                    help="comma-separated session-1 person IDs held out for epoch selection (default: ~20%% of people)")
    ap.add_argument("--max_dataset_val_drop", type=float, default=0.01,
                    help="reject epochs whose CelebA-Spoof val HTER is worse than v2's by more than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=V3_CHECKPOINT)
    ap.add_argument("--dry_run", action="store_true", help="check the pipeline on a few crops; saves nothing")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    rows = [r for r in read_manifest() if r["session"] == "1"]          # session 2 is never read here
    if not rows:
        raise SystemExit("No session-1 crops in the manifest yet.")
    people = sorted({r["person_id"] for r in rows})
    if args.val_people:
        val_people = set(args.val_people.split(","))
    else:
        k = max(1, round(len(people) * 0.2)) if len(people) > 1 else 0
        val_people = set(random.sample(people, k))
    train_rows = [r for r in rows if r["person_id"] not in val_people]
    val_rows = [r for r in rows if r["person_id"] in val_people]
    if not train_rows:
        raise SystemExit("All session-1 people are held out; record more people or pass --val_people.")
    print(f"session-1 people: {people} | held out for epoch selection: {sorted(val_people) or 'none'}")
    print(f"webcam crops: train {len(train_rows)}, held-out {len(val_rows)}")

    webcam_train = [(r["path"], r["y"]) for r in train_rows]
    # balance classes within the webcam data by oversampling the smaller class
    live = [it for it in webcam_train if it[1] == 0]; spoof = [it for it in webcam_train if it[1] == 1]
    if not live or not spoof:
        raise SystemExit("Session-1 training data needs both real faces and attacks.")
    n_side = max(len(live), len(spoof))
    ds_train = dataset_items("train")
    ds_val = dataset_items("val")
    if args.dry_run:
        n_side = min(n_side, 16); ds_val = random.sample(ds_val, 64); val_rows = val_rows[:32]

    ckpt = torch.load(V2_CHECKPOINT, map_location=DEVICE)
    model = build_model(backbone=ckpt.get("backbone", "mobilenet_v3_small"), pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW([
        {"params": [p for p in model.feature_extractor.parameters()], "lr": args.lr_backbone},
        {"params": [p for p in model.head.parameters()], "lr": args.lr_head},
    ], weight_decay=1e-4)

    def evaluate():
        ds_probs = predict(model, ds_val, DEVICE)
        res = {"celeba_val": metrics(ds_probs, [y for _, y in ds_val])}
        if val_rows:
            res["webcam_heldout"] = metrics(predict(model, [(r["path"], r["y"]) for r in val_rows], DEVICE),
                                            [r["y"] for r in val_rows])
        return res

    base = evaluate()
    print("v2 before fine-tuning:", json.dumps(base))
    history, best, best_key = [{"epoch": 0, **base}], None, None
    for epoch in range(1, args.epochs + 1):
        items = random.choices(live, k=n_side) + random.choices(spoof, k=n_side)
        n_ds = int(len(items) * args.dataset_mix) // 2
        ds_live = [it for it in ds_train if it[1] == 0]; ds_spoof = [it for it in ds_train if it[1] == 1]
        items += random.sample(ds_live, n_ds) + random.sample(ds_spoof, n_ds)
        loader = torch.utils.data.DataLoader(ImageList(items, get_train_transform()), batch_size=args.batch_size,
                                             shuffle=True, num_workers=0)
        model.train(); freeze_batchnorm(model)
        t0, total = time.time(), 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(DEVICE)).view(-1), y.to(DEVICE))
            loss.backward(); opt.step(); total += loss.item() * len(y)
        res = evaluate()
        res.update(epoch=epoch, train_loss=total / len(items), seconds=round(time.time() - t0))
        history.append(res)
        print(json.dumps(res))
        ok = res["celeba_val"]["hter"] <= base["celeba_val"]["hter"] + args.max_dataset_val_drop
        key = res["webcam_heldout"]["hter"] if "webcam_heldout" in res else res["celeba_val"]["hter"]
        if ok and (best_key is None or key < best_key):
            best_key, best = key, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    if args.dry_run:
        print("dry run finished; nothing saved"); return
    if best is None:
        raise SystemExit("No epoch kept CelebA-Spoof val HTER within the allowed drop; nothing saved.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest_sha = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    torch.save({"model_state_dict": best, "backbone": ckpt.get("backbone", "mobilenet_v3_small"),
                "image_size": ckpt.get("image_size", 224), "base_checkpoint": os.path.relpath(V2_CHECKPOINT, REPO),
                "best_epoch": best_epoch, "args": vars(args), "session1_people": people,
                "heldout_people": sorted(val_people), "manifest_sha256": manifest_sha}, args.out)
    with open(os.path.join(V3_DIR, "finetune_history.json"), "w") as fh:
        json.dump({"best_epoch": best_epoch, "history": history, "args": vars(args)}, fh, indent=2)
    print(f"saved {args.out} (best epoch {best_epoch}). The app still uses v2 until you switch it.")


if __name__ == "__main__":
    main()
