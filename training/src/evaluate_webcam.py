"""
Before/after report for webcam adaptation: v2 (deployed) vs v3_webcam.

    ./venv/bin/python training/src/evaluate_webcam.py            # both models, if v3 exists
    ./venv/bin/python training/src/evaluate_webcam.py --only_v2  # baseline / sanity check

Evaluates on
  1. Session-2 webcam crops (never used for training or for choosing anything), and
  2. The original 3,900-image CelebA-Spoof test set (data/celeba_spoof_cropped/test).
Both with the plain 0.5 threshold and with the v2 UNCERTAIN band, per label and per
person/condition for session 2. Writes training/results/webcam_v3_eval.json.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from webcam_data import (REPO, GRAY_ZONE_V2, V2_CHECKPOINT, V3_CHECKPOINT, dataset_items,  # noqa: E402
                         metrics, predict, read_manifest)

import torch  # noqa: E402
from liveness.config import DEVICE  # noqa: E402
from liveness.model import build_model  # noqa: E402


def load(path):
    ckpt = torch.load(path, map_location=DEVICE)
    m = build_model(backbone=ckpt.get("backbone", "mobilenet_v3_small"), pretrained=False).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_v2", action="store_true")
    ap.add_argument("--skip_dataset", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "training", "results", "webcam_v3_eval.json"))
    args = ap.parse_args()
    band = tuple(json.load(open(GRAY_ZONE_V2))[k] for k in ("band_low", "band_high"))

    models = {"v2": V2_CHECKPOINT}
    if not args.only_v2:
        if not os.path.isfile(V3_CHECKPOINT):
            raise SystemExit(f"{V3_CHECKPOINT} not found; run finetune_webcam.py first (or use --only_v2).")
        models["v3_webcam"] = V3_CHECKPOINT

    try:
        s2 = [r for r in read_manifest() if r["session"] == "2"]
    except SystemExit:
        s2 = []
    test = [] if args.skip_dataset else dataset_items("test")
    report = {"band_v2": band, "n_session2": len(s2), "n_dataset_test": len(test), "models": {}}

    for name, path in models.items():
        model, res = load(path), {}
        if s2:
            probs = predict(model, [(r["path"], r["y"]) for r in s2], DEVICE)
            ys = [r["y"] for r in s2]
            res["session2"] = {"threshold_0.5": metrics(probs, ys), "with_v2_band": metrics(probs, ys, band)}
            groups = collections.defaultdict(list)
            for r, p in zip(s2, probs):
                for key in (f"label={r['label']}", f"person={r['person_id']}",
                            f"light={r['lighting']}", f"glasses={r['glasses']}"):
                    groups[key].append((p, r["y"]))
            res["session2_groups"] = {k: metrics([p for p, _ in v], [y for _, y in v], band) for k, v in sorted(groups.items())}
        if test:
            probs = predict(model, test, DEVICE)
            ys = [y for _, y in test]
            res["dataset_test"] = {"threshold_0.5": metrics(probs, ys), "with_v2_band": metrics(probs, ys, band)}
        report["models"][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "session2_groups"}, indent=1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
