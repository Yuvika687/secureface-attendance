# Training & Evaluation — Liveness/Anti-Spoofing Model

This folder contains the training code, evaluation code, and real result files for the
liveness classifier used by the attendance app (`liveness/weights/best_liveness.pth`).
It is a focused copy pulled from the original research project
(`anti_spoof_attendance`, archived at
[`archive/liveness-v1`](https://github.com/Yuvika687/anti-spoof-attendance/tree/archive/liveness-v1)
in this same GitHub account) — not the dataset, not the comparison-backbone
checkpoints, not raw face images. It exists so the model can be retrained or
re-evaluated without needing the original project folder.

## What the model does

A MobileNetV3-Small binary classifier that takes a cropped face image and outputs a
single spoof probability (`p_spoof`, 0–1). The deployed app turns that probability into
a three-way decision — `LIVE` / `SPOOF` / `UNCERTAIN` — using a gray-zone band (see
below), not a plain 0.5 cutoff.

## Dataset

[CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) — a public face
anti-spoofing dataset with 10 spoof attack types (photo, poster, A4 paper, PC/phone/pad
replay, 3D mask, and several paper-mask variants) plus live images. Not included here —
regenerate it with `src/prepare_celeba_subset.py` (and `src/prepare_casia_fasd.py` if
you also want the CASIA-FASD cross-dataset check referenced in the original project).

Real split sizes used for training (from `results/crop_dataset_build_stats.json`):

| Split | Images |
|---|---|
| train | 18,200 |
| val | 3,900 |
| test | 3,900 |

## The v1 → v2 crop fix (the most important thing to know before retraining)

The **first trained model (v1)** was trained on raw, full CelebA-Spoof photo frames.
But the real deployment pipeline crops tightly to the detected face first — the face
occupies a median of only ~16.8% of a raw frame. Evaluating v1 through the actual
deployment crop (instead of the raw frame it was trained on) dropped its accuracy from
99.15% (raw-frame eval) to a real **94.54%** (HTER 5.46%) — see `results/v1_vs_v2.json`.

**v2** was retrained on face crops built with the exact same BlazeFace-crop function the
live pipeline uses (`src/build_face_cropped_dataset.py`, which calls `crop_face()` /
`FaceDetector` — the same crop logic reused verbatim in the deployed app's
`liveness/face_utils.py`). v2 is the checkpoint actually shipped
(`liveness/weights/best_liveness.pth`, md5-verified identical to
`anti_spoof_attendance/models/v2_cropped/best_liveness.pth`).

Real numbers, same 3,900-image deployment-crop test set (`results/summary.json`,
`v1_vs_v2` section):

| Metric | v1 (raw-frame-trained) | v2 (deployed) |
|---|---|---|
| Accuracy | 94.54% | **98.69%** |
| APCER (false accept) | 7.74% | **1.38%** |
| BPCER (false reject) | 3.18% | **1.23%** |
| HTER | 5.46% | **1.31%** |

If you retrain from scratch, train on **face crops**, not raw frames — that mismatch is
the single biggest lever on real accuracy in this project.

## Backbone comparison

Three backbones were trained and compared on identical crops/recipe
(`results/summary.json`, `backbone_comparison`; also in `results/breakdown*.json` for
per-condition detail):

| Backbone | Accuracy | HTER | Params | Checkpoint size | CPU latency (median, ONNX) |
|---|---|---|---|---|---|
| **mobilenet_v3_small (deployed)** | 98.69% | 1.31% | 1.00M | 3.95 MB | 2.76 ms |
| resnet18 | 98.90% | 1.10% | 11.24M | 42.96 MB | 17.27 ms |
| efficientnet_b0 | 99.26% | 0.74% | 4.17M | 16.21 MB | 7.70 ms |

MobileNetV3-Small was chosen for deployment despite not having the top accuracy — its
latency and file size are far smaller, and the accuracy gap (98.69% vs 99.26%) was
judged not worth the ~3x larger, ~3x slower model for a real-time webcam check.

## The gray-zone band (three-way decision)

A plain `p_spoof >= 0.5` threshold gives a binary LIVE/SPOOF call with no way to flag
"not sure." `src/evaluate_gray_zone.py` instead selects a symmetric band around 0.5 —
calibrated **only on the validation set, never the test set** — sized so that ~5% of
validation predictions fall inside it. That band is then applied to hold out test
cases:

Real band (from `liveness/weights/gray_zone_v2.json`, also copied to
`training/results/gray_zone_v2.json`):

```
band_low  = 0.024217108554277167
band_high = 0.975782891445723
```

Real effect on the test set (`results/gray_zone_v2.json`):

| | Plain 0.5 threshold | Gray-zone band |
|---|---|---|
| Coverage (fraction decided) | 100% | 95.13% |
| Accuracy (on decided cases) | 98.69% | 99.78% |
| HTER (on decided cases) | 1.31% | 0.22% |

The remaining ~4.87% of test images land `UNCERTAIN` — mostly the images the plain
threshold was getting wrong anyway (24 of 27 false accepts and 19 of 24 false rejects on
the plain threshold are rescued into `UNCERTAIN` rather than resolved wrong).

## Masked-face evaluation

A synthetic mask overlay was tested against the same v2 model
(`src/evaluate_masked.py`, real result in `results/masked_eval.json`): accuracy
98.42%, HTER 1.57% on 3,676 successfully-masked+detected images. This confirms the
model stays reasonably accurate with a mask on, but **no dedicated mask-detection
feature exists** in this model or the deployed app — it is a binary
live-vs-spoof classifier only.

## How to reproduce training

```bash
# from this training/ folder
pip install -r requirements.txt

python src/prepare_celeba_subset.py         # downloads/prepares CelebA-Spoof into data/
python src/build_face_cropped_dataset.py    # builds the face-cropped training set (the v2 fix)

python src/train.py \
    --backbone mobilenet_v3_small \
    --epochs 15 --batch_size 32 --lr 3e-4 --image_size 224 \
    --freeze_epochs 2
# AdamW optimizer, cosine LR schedule, 2 epochs with a frozen pretrained backbone
# (head-only warm-up) then full fine-tuning at a lower LR. See src/train.py for exact
# behavior and src/config.py for all defaults.

python src/evaluate.py                      # test-set metrics
python src/evaluate_gray_zone.py             # recomputes the gray-zone band from val predictions
python src/evaluate_breakdown.py             # per-attack-type / illumination / environment APCER
python src/evaluate_masked.py                # synthetic-mask evaluation
```

`src/export_onnx.py` converts the trained `.pth` checkpoint to ONNX (used for the
latency numbers above); the deployed app loads the PyTorch checkpoint directly, not the
ONNX export.

## Files in this folder

- `src/` — training script (`train.py`), model definition (`model.py`), dataset
  loading (`dataset.py`), the face-crop builder used for the v1→v2 fix
  (`build_face_cropped_dataset.py`), all evaluation scripts, config (`config.py`), and
  supporting utilities (ONNX export, backbone/latency benchmarking, Grad-CAM, synthetic
  mask generation).
- `results/` — real evaluation output: `summary.json` (headline numbers + full
  v1-vs-v2 + backbone comparison + breakdowns), `gray_zone_v2.json` /
  `gray_zone.json` (band calibration + coverage/error tradeoff curve),
  `masked_eval.json`, `deployment_crop_eval*.json`, `backbone_latency.json`,
  `crop_dataset_build_stats.json`, and the corresponding chart images under
  `results/plots/`.
- `outputs/` — raw training history (loss/accuracy per epoch) and confusion-matrix
  data/plots for each backbone, under `outputs/plots/` and `outputs/predictions/`.
- `requirements.txt` — Python packages needed to run the scripts in `src/`.

Not included (regenerate instead of expecting them here): the CelebA-Spoof/CASIA-FASD
datasets, raw downloads, per-sample prediction CSVs, Grad-CAM/crop-check/masked-example
image galleries, and the comparison-backbone checkpoints (`resnet18`, `efficientnet_b0`
`.pth`/`.onnx` files) — all excluded for size and/or because they contain real face
images from the dataset. The `.pth` checkpoints themselves also aren't duplicated here;
the one actually used by the app lives at `../liveness/weights/best_liveness.pth`.
