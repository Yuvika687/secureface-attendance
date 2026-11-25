# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Flask face-recognition attendance app ("SecureFace Attendance") with a self-trained
liveness/anti-spoofing model (MobileNetV3-Small, trained in `training/`) in front of
ArcFace recognition. It has three parts: a **common scanner** (`/`; optionally limited to admin-approved devices), **student pages** (register, register face once, view own attendance) and **admin
pages** (attendance, students, settings, scanner devices, logs, Results, Test Mode,
Pipeline View). Read `docs/HANDOVER.md` before assuming anything about *why* something is
built a certain way.

## Commands

```bash
# Setup (Python 3.13; the venv runs 3.13.15)
python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # SECRET_KEY, FACE_ENCRYPTION_KEY

# Run
./venv/bin/python app.py                 # http://localhost:5000
FLASK_DEBUG=1 ./venv/bin/python app.py   # + scanner test-label dropdown, /admin/dev/clear_today_attendance

# Tests: 78; each test uses a fresh temp DB, never database/smartface.db.
# Sample images are read from the sibling anti_spoof_attendance project.
./venv/bin/pytest -v                     # pytest.ini limits collection to tests/

# Accounts (no hardcoded credentials)
./venv/bin/python scripts/create_admin.py
./venv/bin/python scripts/create_test_student.py
./venv/bin/python scripts/reset_password.py you@example.com

# Real pipeline latency against a running server (if "Require approved scanner
# device" is on, pass an approved browser's scanner_device cookie value)
SCANNER_DEVICE_TOKEN=... ./venv/bin/python scripts/measure_pipeline_latency.py [base_url]

# PDF report (numbers read from training/results/summary.json + docs/pipeline_latency_*.txt)
./venv/bin/python generate_report.py
```

Scanner-device approval is a setting, **off by default** (Settings → Scanner →
"Require approved scanner device"): when off, any browser can open `/` and mark
attendance. When on, only browsers approved in Settings → Approved scanner devices can;
others see "This device is not an approved scanner." and the API returns 403.

Port 5000 can be taken by macOS AirPlay Receiver (`lsof -ti:5000 | xargs kill -9`).
The camera works on `http://localhost` without HTTPS; for other devices use ngrok
(`docs/ngrok.md`). The venv's scripts have absolute shebangs; if the folder moves,
recreate the venv.

**Verifying changes without touching real data:** copy the project (minus `venv/`, `.git`)
to a scratch dir, `sqlite3 database/smartface.db ".backup <copy>/database/smartface.db"`,
and run the copy with `<repo>/venv/bin/python -c "import app; app.app.run(port=...)"` from
inside it (`Config.BASE_DIR` follows the copy).

## Architecture

### Scan pipeline (`app.py` → `api_recognize_face()`)

1. Device check `_scanner_allowed()`: skipped unless the `require_scanner_approval` setting is `1`; then the SHA-256 of the `scanner_device` cookie must be in `scanner_devices`.
2. BlazeFace detect (0 → `NO_FACE`, >1 → `MULTI_FACE`).
3. Same-face tracking by **IoU** of consecutive face boxes; `< SAME_FACE_MIN_IOU` (0.25) → `FACE_MOVED`, session reset.
4. `crop_face()` (`liveness/face_utils.py`, **must stay byte-identical** to training: 25% margin, 224×224, ImageNet norm; guarded by `tests/test_crop_identical.py`).
5. MobileNetV3-Small → raw P(spoof); rolling 5-frame average **per Flask session** (never global).
6. Gray-zone three-way decision (`liveness/weights/gray_zone_v2.json` band + `liveness_threshold` setting). Only SPOOF clears an in-progress challenge; UNCERTAIN does not.
7. Face Mesh + ArcFace **only on LIVE**.
8. Logged-in student on the scanner → only their own face may proceed.
9. **Random challenge, verified server-side** (`_new_challenge` / `_advance_challenge`, second check after the liveness model): a random `_pick_turn()` left/right instruction, then one blink. From Face Mesh: `head_yaw` straight (|yaw| < 0.08) then turned past 0.12 the asked way (positive yaw = the person's own left, on the unmirrored frame), then `eyes_closed` open → closed → open (`_advance_blink`), same user, within `CHALLENGE_TIMEOUT_SECONDS` (15 s). Wrong-way turn or timeout → new random challenge. Nothing from the browser is trusted.
10. Attendance (Present/Late), once per day. Each stage → `scan_events` with `p_spoof_frame`, `p_spoof_avg`, `iou`.

### Admin-only tools (none of these mark attendance)
- **Pipeline View** (`/admin/pipeline_view`): per-frame box/crop/224 input/score; Grad-CAM on demand via `utils/gradcam.py` (copied from `training/src/gradcam.py`) on a deep copy of the model, layer `feature_extractor.features[12]`. Saves nothing to disk.
- **Test Mode** (`/admin/test_mode`): records real decisions per category into `test_mode_results`; APCER/BPCER/HTER from those counts. Collection mode saves crops to `data/domain_adaptation/` (gitignored) for `docs/domain_adaptation_plan.md`.
- **Spoof Log** (CSV export) with **Scan Log** as its second tab, **Results** (reads `training/results/summary.json` at request time; separate live-test section).

### Config is admin-tunable, not hardcoded
`liveness_threshold`, `face_tolerance`, `min_face_samples`, late cut-off and email settings
live in the `settings` table (`/settings`). **Never hand-edit them to make a test or face
pass.**

### Dual DB backend
SQLite locally (`database/smartface.db`, gitignored), Postgres if `DATABASE_URL` is set.
`database/db.py` (inline SQLite schema) and `database/schema.sql` (Postgres) must stay in
sync, and new tables/columns also need an entry in `app.py`'s startup migration block for
existing SQLite DBs. The Postgres path has never been run against a live instance.

### `training/` is separate from the live app
Training/eval code and real result files, not imported at runtime. Only quote numbers from
`training/results/`: 98.69% accuracy, HTER 1.31% (APCER 1.38%, BPCER 1.23%), v1 → v2
94.54% → 98.69%, and measured latency (`docs/pipeline_latency_2026-11-07.txt`). **Never** use
99.15%, 0.85%, 26.7 FPS, 61.2% or 99.8% (fabricated by a design mockup).

## Known landmines

- **`/admin/wipe_model`** deletes every user's face data and all attendance. It needs JSON
  `{"confirm": "WIPE"}` and is audit-logged, but there is still no UI button for it.
- **Domain shift.** Real webcam faces often land in the UNCERTAIN band, and a phone photo was
  judged LIVE once. Don't retune thresholds ad hoc; follow `docs/domain_adaptation_plan.md`
  (session 2 is test-only).
- No mask detector exists. The "remove your mask" text is shown on every non-match, not
  detected.
- `mediapipe==0.10.30` is pinned exactly (newer crashes on Apple Silicon + Metal).

## Project rules (standing instructions from the project owner)

1. No fake data, placeholder logic or invented numbers anywhere.
2. Never change detection thresholds to make a test or a specific face pass.
3. Report findings and wait for approval before risky/ambiguous changes (thresholds, security checks, anything destructive).
4. Never modify `anti_spoof_attendance` or `Smart-Attendance-System-main`.
5. Don't modify `liveness/` or the ArcFace matching logic unless the task is about them; if `liveness/face_utils.py` is touched, re-run `tests/test_crop_identical.py`.
6. Show real, reproducible verification after every change; test on a scratch DB copy, never the real DB.
7. Before each task: commit + tag `pre-<task>-YYYYMMDD`. After: pytest, start the app, check the pipeline on sample images, commit, push.
8. Do only what the task asks: no extra refactoring, renaming or deletions.
9. No co-author lines in commits.
