# Face Attendance System with Liveness Detection

A Flask attendance app that checks a user in by face recognition, with an anti-spoofing
("liveness") check in front of it so a printed photo, phone screen, or similar replay
attack can't be used to fake someone's attendance.

This repo is the final merged project: a Flask face-recognition attendance app
(originally a student project, `Sofzenix/Smart-Attendance-System`) with a self-trained
liveness/anti-spoofing model merged into its scan pipeline. The model's own training
project (dataset prep, training script, and full evaluation results) is included under
[`training/`](training/) so the model can be retrained or audited. The original,
pre-merge training project is archived in full at
[`archive/liveness-v1`](https://github.com/Yuvika687/secureface-attendance/tree/archive/liveness-v1)
in this same repo.

## What it does

1. A student/employee registers a face (webcam captures several real embeddings).
2. At check-in, they look at the camera. The scan pipeline runs:
   - Face detection (is there a face, and only one?)
   - Liveness check (is it a real face, not a photo/screen/mask?) — a silent wait for
     one natural blink while classified `LIVE`, with an 8-second timeout
   - Face recognition (whose face is it?) — matched against registered embeddings
   - Attendance gets marked (`Present` / `Late`, based on the configured cutoff time)
3. Every scan's outcome is logged (dev-only review page) so a spoof attempt, an
   unrecognized face, or a successful check-in can all be traced back to which pipeline
   stage decided it.

## Pipeline

```
Webcam frame (browser)
   → BlazeFace detection (MediaPipe FaceDetector) — reject if 0 or >1 faces
   → crop_face() — the same crop function used to build the liveness model's training data
   → Liveness classifier (MobileNetV3-Small, trained in this repo — see training/)
        → p_spoof probability → three-way decision via a calibrated gray-zone band:
          LIVE / SPOOF / UNCERTAIN (not a plain 0.5 cutoff — see training/README.md)
   → Silent-blink challenge (only once classified LIVE) — natural blink detected via
     MediaPipe FaceLandmarker (478-point mesh) eye-aspect-ratio, 8s timeout
   → Face recognition — ArcFace (w600k_r50) ONNX, 512-D embedding, cosine similarity
     against the user's stored (Fernet-encrypted) embeddings
   → Attendance marked (SQLite locally, or Postgres/Neon in production)
```

## Models used

| Model | Trained by this project? | Purpose |
|---|---|---|
| MobileNetV3-Small liveness classifier (`liveness/weights/best_liveness.pth`) | **Yes** — see [`training/`](training/) for the full training/evaluation code and real results | Binary real-vs-spoof probability per face crop |
| ArcFace `w600k_r50` (ONNX, auto-downloaded on first run) | No — a pretrained, publicly available face-recognition model | 512-D face embedding for recognition/matching |
| MediaPipe BlazeFace / FaceLandmarker (auto-downloaded on first run) | No — Google's pretrained MediaPipe models | Face detection and 478-point landmark mesh (for the blink check) |

## Real results (the liveness model)

Full details, methodology, and every underlying result file are in
[`training/README.md`](training/README.md) and [`training/results/`](training/results/).
Headline numbers, from a held-out 3,900-image CelebA-Spoof test set the model was never
trained on (`training/results/summary.json`):

| | Accuracy | HTER (half total error rate) |
|---|---|---|
| Plain 0.5 threshold | 98.69% | 1.31% |
| With the calibrated gray-zone band (95.1% of cases decided, rest flagged `UNCERTAIN`) | 99.78% (on decided cases) | 0.22% (on decided cases) |

The model was retrained once already after discovering a real bug: the first version
(v1) was trained on uncropped photos but evaluated on cropped faces in production,
which alone cost ~4 points of accuracy. The current model (v2) is trained on the same
crop the live pipeline actually uses. See `training/README.md` for the full before/after
numbers and why MobileNetV3-Small was chosen over two larger backbones that scored
slightly higher.

Full-pipeline latency (face detection → liveness → recognition, one real round trip,
45 requests): **median 80.6 ms** (`docs/pipeline_latency_2026-11-07.txt`).

## How to run it

### Requirements
- Python 3.11 (pinned via `.python-version` / `runtime.txt`; `mediapipe==0.10.30` is
  pinned exactly — newer versions crash on Apple M-series + Metal 4, see
  `requirements.txt`)
- A webcam, for face registration and check-in

### Setup

```bash
git clone https://github.com/Yuvika687/secureface-attendance.git
cd secureface-attendance

python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env: set SECRET_KEY, FACE_ENCRYPTION_KEY (generate with the command
# shown in .env.example), and DATABASE_URL only if using Postgres/Neon —
# omit DATABASE_URL to use local SQLite instead.
```

On first run, `app.py` auto-downloads the ArcFace ONNX model and the MediaPipe
BlazeFace/FaceLandmarker model files (not committed to this repo — see
`.gitignore`). The liveness model (`liveness/weights/best_liveness.pth`) **is**
committed and needs no download.

### Create accounts

```bash
python scripts/create_admin.py          # interactive — creates/resets an admin login
python scripts/create_test_student.py   # interactive — creates a student login
                                         # (face still needs to be registered for real,
                                         # via the webcam, after first login)
```

### Run

```bash
python app.py                 # production-like: debug off
FLASK_DEBUG=1 python app.py   # local dev: enables the dev-only scan log, test-label
                               # dropdown on the scanner page, and the "clear today's
                               # attendance" admin action
```

Then open `http://localhost:5000`.

### Run the tests

```bash
pytest
```

## Repository layout

- `app.py`, `templates/`, `static/`, `database/`, `utils/` — the Flask attendance app
- `liveness/` — the liveness model's inference code (`predictor.py`, `face_utils.py`,
  `model.py`) and the trained weights actually used at runtime
  (`liveness/weights/best_liveness.pth`, `gray_zone_v2.json`)
- `training/` — the model's own training/evaluation project: training script,
  evaluation scripts, the real result files behind every number above, and
  instructions to reproduce training (see `training/README.md`)
- `scripts/` — account-creation and latency-measurement scripts
- `docs/` — handover document and dated, real verification logs (latency, test output,
  auth checks)

## Known issues

See [`docs/HANDOVER.md`](docs/HANDOVER.md) for the full, honest list (what's verified,
what's unverified, and what's still risky — including an existing `/admin/wipe_model`
route that wipes all users' face data, and an orphaned unused template).
