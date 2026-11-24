# SecureFace Attendance

A Flask attendance app that marks students present by face recognition, with a
**self-trained liveness (anti-spoofing) model** in front of it, to screen for printed photos and screen replays.

It merges two of the owner's earlier projects: a Flask face-recognition attendance app
(originally `Sofzenix/Smart-Attendance-System`) and a MobileNetV3-Small liveness
classifier trained from scratch for this purpose. The model's full training and
evaluation project is in [`training/`](training/). The original pre-merge training
project is archived at
[`archive/liveness-v1`](https://github.com/Yuvika687/secureface-attendance/tree/archive/liveness-v1).

## The three parts

| Part | Who | What |
|---|---|---|
| **Common scanner** (`/`) | Anyone (optionally only on admin-approved devices) | Camera → liveness → recognition → natural blink → attendance. Shows name + student ID with "Attendance marked" or "Already marked today". |
| **Student pages** | Logged-in students | Create an account, register their face once, view their own attendance. |
| **Admin pages** | Logged-in admins | Dashboard, Students (with **Reset face**), Attendance (summary, chart, filters, CSV), **Pipeline View** with Grad-CAM, **Test Mode**, Spoof Log (with a Scan Log tab), **Results**, Settings. |

A public **How it works** page (`/how_it_works`) explains the pipeline.

## Scan pipeline

```
Webcam frame (if "Require approved scanner device" is on: approved devices only)
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
average and the IoU (Admin → Spoof Log → Scan Log tab).

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

## Run on your laptop

Download the repository ZIP and extract it to a writable folder, or clone it with Git.
On Windows install 64-bit Python 3.13 with the Python launcher. Open that folder and
run **setup.cmd** once. It installs dependencies in a private virtual environment and
asks you to create your own admin. Then run **start.cmd** and open
[SecureFace](http://localhost:5000). Keep the server window open.
First startup requires internet for the pretrained vision models. On macOS/Linux use
Python 3.13 to create a venv, install requirements.txt, then run
`python scripts/create_admin.py` and `python run_local.py` in that environment.

1. Students create accounts and use **Register face** on their dashboard. Allow camera
   access; the popup takes 5 samples with only one face visible.
2. Open the scanner (`/`) on the attendance laptop. Look at the camera and blink once.
   The scanner shows the name and student ID with "Attendance marked" or "Already marked
   today". The record appears on the admin Dashboard/Attendance and the student's dashboard.
3. Optional: to allow only specific laptops to scan, turn on Settings → Scanner →
   **Require approved scanner device**, then on each scanner laptop log in as admin and
   click **Set this device as scanner**.

The scanner shows one short status at a time: "Step into the frame.", "Only one person at
a time.", "Hold still — checking again.", "Spoof detected — please use your real face.",
"Face not recognized…" or "Blink once to check in.". The camera view is zoomed to the
face for display only; the full camera frame is what the server analyses.

Each installation creates its own empty SQLite database in `database/smartface.db`.
Every friend creates their own accounts, scanner approvals and enrollments. Back up the
database after stopping the app, together with the private `.face_key`. Keep the
database, keys, uploads and virtual environment out of Git. Share the repository via
GitHub rather than zipping your used working directory.

### Optional Cloudflare Tunnel

After the local app works, install cloudflared separately following the
[Cloudflare installation guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
In a second terminal run `cloudflared tunnel --url http://127.0.0.1:5000`.
Open its printed HTTPS address on the scanner, sign in as admin on that address,
approve the browser, and log out. A new temporary hostname needs a new approval.
Keep your laptop awake and both processes running. Ctrl+C in the tunnel terminal
stops public access. The HTTPS link reaches the database of the laptop hosting the tunnel.

The scanner also evaluates face masks with a separate MobileNetV2 classifier.
A mask prediction asks for an uncovered face and blocks check-in. Spoof rejection
takes priority. These checks depend on a visible face and can make mistakes.
See [mask model and license](docs/MASK_MODEL.md).

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
- This local edition uses SQLite on the host laptop.

Full detail: [`docs/HANDOVER.md`](docs/HANDOVER.md).
