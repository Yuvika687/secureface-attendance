# SecureFace Attendance

A Flask attendance app that marks students present by face recognition, with a
**self-trained liveness (anti-spoofing) model** in front of it, so a printed photo, a
phone or a screen can't be used to fake someone's attendance.

It merges two of the owner's earlier projects: a Flask face-recognition attendance app
(originally `Sofzenix/Smart-Attendance-System`) and a MobileNetV3-Small liveness
classifier trained from scratch for this purpose. The model's full training and
evaluation project is in [`training/`](training/). The original pre-merge training
project is archived at
[`archive/liveness-v1`](https://github.com/Yuvika687/secureface-attendance/tree/archive/liveness-v1).

## The three parts

| Part | Who | What |
|---|---|---|
| **Common scanner** (`/`) | Anyone, but only on a device an admin approved | Camera → liveness → recognition → natural blink → attendance. Shows name + student ID with "Attendance marked" or "Already marked today". |
| **Student pages** | Logged-in students | Create an account, register their face once, view their own attendance. |
| **Admin pages** | Logged-in admins | All attendance, students (with **Reset face**), settings, **scanner devices**, scan log, spoof log, **Results**, **Test Mode**, **Pipeline View** with Grad-CAM. |

A public **How it works** page (`/how_it_works`) explains the pipeline.

## Scan pipeline

```
Webcam frame (approved scanner device only; others see "This device is not an approved scanner.")
  → BlazeFace face detection: reject if 0 or more than 1 face
  → Same-face tracking: IoU of this face box vs. the previous frame's; IoU < 0.25 ⇒ new person, reset
  → crop_face(): face + 25% margin, 224×224, ImageNet normalization (identical to training)
  → MobileNetV3-Small liveness model → P(spoof); averaged over the last 5 frames
  → Three-way decision with the calibrated UNCERTAIN band (liveness/weights/gray_zone_v2.json):
       LIVE / SPOOF / UNCERTAIN; SPOOF and UNCERTAIN stop here (recognition never runs)
  → Face Mesh alignment + ArcFace (w600k_r50) embedding, cosine match vs. registered students
  → Silent natural blink, verified by the server: open → closed → open eyes (Face Mesh EAR), same user, 8 s window
  → Attendance: Present / Late (admin cut-off), once per day
```

Every stage reached is logged in `scan_events` with the frame's P(spoof), the 5-frame
average and the IoU (Admin → Scan Log / Spoof Log).

## Models

| Model | Trained here? | Job |
|---|---|---|
| MobileNetV3-Small liveness classifier (`liveness/weights/best_liveness.pth`) | **Yes** (see [`training/`](training/)) | Real face or spoof |
| BlazeFace short-range (MediaPipe, auto-downloaded) | No | Face detection |
| MediaPipe Face Landmarker, 478 points (auto-downloaded) | No | Alignment, eye openness for the blink |
| ArcFace `w600k_r50` ONNX (InsightFace, auto-downloaded) | No | 512-D identity embedding |

## Results (liveness model)

From the held-out 3,900-image CelebA-Spoof test set (`training/results/summary.json`):

| | Accuracy | APCER | BPCER | HTER |
|---|---|---|---|---|
| Deployed v2 model | 98.69% | 1.38% | 1.23% | 1.31% |

**The v1 → v2 crop fix:** v1 was trained on face crops framed differently from what the
live pipeline produces. Retraining on crops made exactly like the live ones raised test
accuracy from **94.54% to 98.69%**.

**Measured pipeline latency** (45 real requests against the running app):
median round trip **80.6 ms** (`docs/pipeline_latency_2026-11-07.txt`).

These are dataset numbers. On a real webcam the model sees a different domain (see
[`docs/domain_adaptation_plan.md`](docs/domain_adaptation_plan.md)). Admin → **Results**
shows the full set read from `summary.json`, and live Test Mode results in a separate,
labeled section.

## Run it (macOS)

Requirements: **Python 3.13** (`.python-version`, `runtime.txt`) and a webcam.
`mediapipe==0.10.30` is pinned exactly because newer versions crash on Apple Silicon
with Metal (see `requirements.txt`).

```bash
git clone https://github.com/Yuvika687/secureface-attendance.git
cd secureface-attendance
python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env     # set SECRET_KEY and FACE_ENCRYPTION_KEY; omit DATABASE_URL for local SQLite

./venv/bin/python scripts/create_admin.py           # create an admin login (interactive)
./venv/bin/python app.py                            # http://localhost:5000
```

The first run downloads the BlazeFace, Face Landmarker and ArcFace model files (not in
git). The liveness weights are committed.

**First-time setup in the browser:**

1. Open `http://localhost:5000/login`, choose **Admin**, and log in.
2. On the computer that will be the scanner, go to **Settings → Scanner Devices** and
   click **Set this device as scanner**. Until you do, `/` shows "This device is not
   an approved scanner."
3. Students create an account (`/register`), log in, and register their face from their
   dashboard.

Port 5000 can clash with macOS AirPlay Receiver: turn it off in System Settings, or free
the port with `lsof -ti:5000 | xargs kill -9`. To reach the app from another device
(e.g. a phone) see [`docs/ngrok.md`](docs/ngrok.md).

`FLASK_DEBUG=1 ./venv/bin/python app.py` adds dev tools: the scanner's test-label
dropdown and "clear today's attendance".

### Tests

```bash
./venv/bin/pytest -v
```

The tests cover the crop-identity regression test, login and roles (including a wrong
Student/Admin toggle), students blocked from admin pages, CSRF, scanner device approval,
attendance marking and "Already marked today", wipe confirmation, registration owner
checks, and Test Mode never marking attendance. They use a temporary database, never
`database/smartface.db`, and read sample images from the sibling
`anti_spoof_attendance` project.

### Other scripts

```bash
./venv/bin/python scripts/create_test_student.py            # student login (face registered later via webcam)
./venv/bin/python scripts/reset_password.py you@example.com  # local password reset
SCANNER_DEVICE_TOKEN=<scanner_device cookie value> \
  ./venv/bin/python scripts/measure_pipeline_latency.py      # real latency against a running server
./venv/bin/python generate_report.py                         # regenerate SecureFace_Attendance_Report.pdf
```

## Repository layout

- `app.py`: all routes and the scan pipeline (`api_recognize_face`)
- `liveness/`: liveness model inference (detector + crop, model, predictor, weights)
- `utils/face_utils.py`: Face Mesh + ArcFace recognition, registration with owner checks, Fernet-encrypted embeddings
- `utils/gradcam.py`: Grad-CAM for Pipeline View (copied from `training/src/gradcam.py`)
- `database/`: SQLite/Postgres layer and schema
- `templates/`, `static/`: UI (one dark theme; status colors: green LIVE, red SPOOF, blue UNCERTAIN, orange no face)
- `training/`: the model's training/evaluation code and real result files
- `scripts/`, `tests/`, `docs/`

## Known limits

- The model was trained on dataset photos. On a webcam, real faces often land in the
  UNCERTAIN band and a phone photo was once judged LIVE. A data-collection tool is ready
  (Test Mode → Collection mode); see `docs/domain_adaptation_plan.md`. No recalibration has been done yet.
- The silent blink is not a defense against video replay (a video blinks too). The
  liveness model is the main defense.
- The Postgres path has never been run against a live database.

Full detail: [`docs/HANDOVER.md`](docs/HANDOVER.md).
