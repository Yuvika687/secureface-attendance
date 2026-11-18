# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Flask face-recognition attendance app with a self-trained liveness/anti-spoofing model
in front of the recognition step. It's a merge of two separate prior projects: a Flask
attendance app (originally "SmartFace") and a trained MobileNetV3-Small liveness
classifier (trained in `training/`, originally a separate project called
`anti_spoof_attendance`). See `README.md` for the full pipeline/architecture explanation
and `docs/HANDOVER.md` for a detailed, fact-checked project history (decisions made and
why, known issues, pending tasks) — read `docs/HANDOVER.md` before assuming anything
about "why" something is built a certain way.

## Commands

```bash
# Setup
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in SECRET_KEY, FACE_ENCRYPTION_KEY

# Run
./venv/bin/python app.py                 # production-like
FLASK_DEBUG=1 ./venv/bin/python app.py   # dev tools: scanner test-label dropdown,
                                          # /admin/dev/clear_today_attendance, /admin/scan_log styling

# Tests (one regression test: crop_face() output must stay byte-identical to the
# original training project's crop function, or the liveness model sees differently
# shaped input than it was trained on)
./venv/bin/pytest -v

# Account creation (no hardcoded credentials anywhere)
./venv/bin/python scripts/create_admin.py
./venv/bin/python scripts/create_test_student.py   # creates login only; face must be
                                                     # registered via a real webcam after
./venv/bin/python scripts/reset_password.py you@example.com   # local-only password reset

# Real pipeline latency measurement against a running server
./venv/bin/python scripts/measure_pipeline_latency.py

# Regenerate the PDF project report
./venv/bin/python generate_report.py
```

The venv's scripts (`pip`, `pytest`) have the absolute path baked into their shebang
line — if this folder is ever moved/renamed again, recreate the venv rather than
debugging "command not found" errors.

Default port is 5000; macOS's AirPlay Receiver also binds 5000 by default and can block
the dev server from starting — `lsof -ti:5000 | xargs kill -9` or disable AirPlay
Receiver in System Settings.

## Architecture

### The scan pipeline (the core of the app, in `app.py`'s `api_recognize_face()`)

One POST to `/api/recognize_face` per webcam frame, each stage able to short-circuit:

1. Decode base64 JPEG → BlazeFace detect (0 faces → `NO_FACE`; >1 → `MULTI_FACE`)
2. Same-face consistency check (face moved >60% of its own size → treat as a new subject,
   reset session state)
3. `crop_face()` (`liveness/face_utils.py`, **must stay byte-identical** to the original
   training project's crop — 25% margin, 224×224, ImageNet normalization; this is what
   `tests/test_crop_identical.py` guards)
4. MobileNetV3-Small liveness classifier → raw `p_spoof` for this frame
5. Rolling 5-frame average, kept **per Flask session cookie** (not global — a global
   list used to leak liveness state between concurrent users; don't reintroduce that)
6. Three-way decision via a **gray-zone band**, not a plain 0.5 cutoff:
   `band_low <= p_spoof <= band_high` → `UNCERTAIN`; else `SPOOF` if `>= threshold`
   else `LIVE`. Band is `liveness/weights/gray_zone_v2.json` (calibrated on a validation
   set only, never test). `threshold` is the admin-configurable `liveness_threshold`
   setting. **Only `SPOOF` clears the in-progress blink challenge; `UNCERTAIN` does
   not** — this was a real bug fix (motion-blur UNCERTAIN blips used to force a fresh
   challenge).
7. Face Mesh + ArcFace recognition runs **only on `LIVE`**, never on SPOOF/UNCERTAIN —
   confirmed by how the code branches, don't move recognition earlier in the pipeline.
8. Silent single-blink check (eyes-closed→eyes-open via real EAR from Face Mesh), 8s
   timeout. No visible challenge text (there used to be a randomized
   blink/turn-left/turn-right prompt; removed in favor of this, which is a known,
   deliberate tradeoff against video-replay — see `docs/HANDOVER.md` Section 5).
9. Attendance marked (Present/Late by a configurable cutoff), every stage reached logged
   to the `scan_events` table.

### Models (table in `README.md` / `docs/HANDOVER.md` Section 3)

Only the **MobileNetV3-Small liveness classifier** (`liveness/weights/best_liveness.pth`)
is self-trained — training code, eval scripts, and real result files are in `training/`.
BlazeFace, MediaPipe FaceLandmarker, and ArcFace (`w600k_r50`) are all pretrained,
auto-downloaded on first run (gitignored, not committed).

### Config/thresholds are admin-tunable at runtime, not hardcoded

`liveness_threshold`, `face_tolerance`, `min_face_samples`, late-cutoff time, and email
settings all live in the `settings` DB table, read/written via `/settings`. **Never
hand-edit these to make a specific test or face "pass"** — if a real threshold looks
wrong (see the gray-zone band note in Known Issues below), report it with real data
instead of retuning it unilaterally.

### Dual DB backend

SQLite locally (`database/smartface.db`, gitignored), Postgres (Neon) in production —
switched automatically by whether `DATABASE_URL` is set (`config.py`). `database/db.py`
wraps both behind the same interface; `database/schema.sql` mirrors the inline SQLite
schema for Postgres. The Postgres path has never been run against a live instance —
treat it as implemented-but-unverified.

### `training/` is a separate concern from the live app

It's the model's own training/eval project (dataset prep, `train.py`, eval scripts, real
result JSONs), kept for reproducibility — not imported or run by `app.py` at runtime.
See `training/README.md` for the v1→v2 crop-mismatch bug/fix story and real
accuracy/HTER numbers; don't quote numbers from anywhere else (there's a landing-page
mockup in the original training project with several fabricated numbers — listed
explicitly in `docs/HANDOVER.md` Section 4 as "numbers that must NOT be used").

## Known landmines

- **`/admin/wipe_model`** (in `app.py`) deletes every user's face data and all
  attendance rows, with no confirmation gate in the route itself. Pre-existing, not
  something this merge added — be deliberate before touching it.
- `generate_report.py`'s PDF **content** (not just branding) still describes a deleted
  rule-based anti-spoofing system and an unverified accuracy claim — don't treat it as
  a source of truth about the current system.
- `templates/attendance.html` is orphaned (zero references anywhere) — confirmed by
  grep, not yet explained or removed.
- No mask-detection feature exists. Don't add mask-specific UI/messages without first
  evaluating whether a reliable signal exists (see `docs/HANDOVER.md` Section 5) and
  reporting that before building anything.

## Project rules (standing instructions from the project owner)

1. No fake data, placeholder logic, or invented numbers anywhere — in code, UI, or
   docs. Every number must trace to a real file or measurement.
2. Never change detection thresholds to make a test or a specific face pass.
3. Report findings and wait for approval before risky/ambiguous changes (thresholds,
   security checks, anything destructive).
4. `anti_spoof_attendance` and the original `Smart-Attendance-System-main` folders are
   historical sources only — never modify them.
5. If you ever touch `liveness/face_utils.py`, re-run `tests/test_crop_identical.py`
   and confirm it still passes before committing.
6. Show real, reproducible verification (curl/pytest/sqlite output, etc.) after every
   change — not just a description of what was done.
