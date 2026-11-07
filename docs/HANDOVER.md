# HANDOVER — SmartFace + Liveness Attendance System

Written 2026-11-07. Everything in this document was verified against the actual code, git
history, result files, or a live run of the app on 2026-11-07 (see cited file paths and
command outputs). Anything I could not verify this way is explicitly marked **"unverified"**.
Do not treat unmarked claims as self-evidently true either — re-check anything load-bearing
before you rely on it; code changes after this date are not reflected here.

---

## 1. PROJECT SUMMARY

**What it does (plain language):** A face-recognition attendance system for students. A
student stands in front of a webcam; the system (a) checks the face is a real live person,
not a photo/screen/video, using a trained neural network, (b) recognizes who they are using
ArcFace face embeddings, (c) waits for them to naturally blink once, then (d) marks them
present (or late) for the day in a database. Admins get a dashboard, analytics, CSV export,
and settings. It was built by merging two of the repo owner's own earlier projects:

- `anti_spoof_attendance` — a trained MobileNetV3-Small liveness/anti-spoofing classifier
  (the "brain" that decides LIVE vs SPOOF).
- The original `Smart-Attendance-System-main` ("SmartFace") — a Flask attendance app with
  ArcFace face recognition, which used to have its own weaker rule-based anti-spoofing.

The merge replaced SmartFace's old rule-based spoof checks with the trained model from
`anti_spoof_attendance`, and fixed a number of pre-existing security/correctness bugs in
SmartFace along the way (see Section 5).

**Folder:** `/Users/yuvikamalhotra/Documents/project/secureface-attendance` (renamed from
`liveness_smartface_merged` on 2026-11-07, after this document was first written; the
project was also renamed to "SecureFace Attendance" and its GitHub repo to
`Yuvika687/secureface-attendance`, renamed from `anti-spoof-attendance`)
**Git branch:** `main` (renamed from `master` in the same pass)
**Latest commit:** `5fc0663c91a5deda9f74a929d8adfcdd92746352` (2026-11-02 15:18:10 +0530)

**Recent commits** (`git log --oneline -5`, oldest to newest):
```
4f9d93e Initial import: SmartFace Attendance System (unmodified copy)
f74217b Integrate the trained liveness model into SmartFace; fix security/correctness issues (Steps 2-3)
399200d Add dev-only test tooling: clear-attendance action, scan event log, test-student script
23e46f1 Fix challenge reset on transient UNCERTAIN; add dev-only test-label tagging
5fc0663 Silent-blink attendance flow, message cleanup, real face box, UI bug fixes
```
`git status` was clean (no uncommitted changes) as of this writing.

**Two sibling projects exist and must never be modified** (standing rule from day one of
this merge):
- `/Users/yuvikamalhotra/Documents/project/anti_spoof_attendance` — source of the trained
  liveness model, the real benchmark results, and the crop/detector code reused verbatim.
- `/Users/yuvikamalhotra/Documents/Projects/Smart-Attendance-System-main` — the original
  unmodified SmartFace app this project was copied from.

---

## 2. HOW TO RUN

### Setup
```bash
cd /Users/yuvikamalhotra/Documents/project/secureface-attendance
python3 -m venv venv          # already exists; only needed if rebuilding
./venv/bin/pip install -r requirements.txt
```
**Verified fact:** the existing `venv/` runs **Python 3.13.15** (`./venv/bin/python --version`).
This contradicts `.python-version` and `runtime.txt`, which both say `python-3.11.11` —
those two files are leftovers from the original SmartFace project's deployment target and
have not been reconciled with what's actually installed. See Known Issues.

### Model files and where they live
- `liveness/weights/best_liveness.pth` (4.1MB, **committed to git**) — the trained
  MobileNetV3-Small checkpoint. Verified byte-identical (`md5`) to
  `anti_spoof_attendance/models/v2_cropped/best_liveness.pth`, i.e. this is the real "v2"
  model whose numbers are in `results/summary.json` (Section 4), not a different/untested copy.
- `liveness/weights/gray_zone_v2.json` (committed) — the real computed UNCERTAIN-band
  thresholds from the original project's `evaluate_gray_zone.py`.
- `liveness/weights/blaze_face_short_range.tflite` — BlazeFace detector, **gitignored**,
  auto-downloaded on first run by `liveness/face_utils.py`.
- `models/` directory (gitignored, auto-downloaded on first run):
  - `models/face_landmarker.task` — MediaPipe FaceLandmarker (478-point mesh), downloaded by
    `utils/face_utils.py`.
  - `models/w600k_r50.onnx` — ArcFace recognition model, downloaded by `utils/face_utils.py`
    (extracted from the InsightFace `buffalo_l.zip` bundle).
- If you're on a machine with no internet access or with the SSL issue described in Known
  Issues, these three files must be placed manually at those exact paths.

### Exact commands
```bash
# Create an admin account (interactive, no hardcoded password):
./venv/bin/python scripts/create_admin.py

# Create a student login (interactive; does NOT register a face — that needs a real webcam):
./venv/bin/python scripts/create_test_student.py

# Run normally (production-like, no dev tools):
./venv/bin/python app.py

# Run with dev tools enabled (scanner-page test-label dropdown, /admin/dev/clear_today_attendance):
FLASK_DEBUG=1 ./venv/bin/python app.py

# Alternative one-liner that also auto-creates an admin from env vars on first run of a fresh DB:
ADMIN_EMAIL=you@example.com ADMIN_PASSWORD='YourPass#123' ./venv/bin/python app.py
```

### Port, URLs, known issues
- Default port: **5000** (Flask dev server). **Known macOS issue, hit directly during this
  project**: macOS's "AirPlay Receiver" also listens on port 5000 by default; if the server
  won't bind, Flask itself prints `Address already in use ... On macOS, try searching for
  and disabling 'AirPlay Receiver' in System Settings.` Alternative: kill whatever holds the
  port with `lsof -ti:5000 | xargs kill -9`, or disable AirPlay Receiver in System Settings →
  General → AirDrop & Handoff.
- `http://127.0.0.1:5000/` — the public scanner (home page, no login).
- `http://127.0.0.1:5000/login` — student login. `/admin_login` — admin login (**two separate
  pages today**, not yet merged into one with a role toggle — see Section 6c).
- `http://127.0.0.1:5000/admin` — admin dashboard (admin-only).
- `http://127.0.0.1:5000/employee` — student dashboard (student-only).
- `http://127.0.0.1:5000/admin/scan_log` — admin-only review of recent scan events.

---

## 3. ARCHITECTURE

### Folder structure (one line per important file)

```
app.py                       Flask app: all routes, the recognize_face pipeline, session helpers
config.py                    Config class: SECRET_KEY, DB URL auto-detect, mail, Fernet key, uploads
database/db.py                DB connection (SQLite + a Postgres wrapper), schema, settings helpers
database/schema.sql            Postgres schema (mirrors the inline SQLite schema in db.py)
database/smartface.db          The actual SQLite DB file (gitignored; real local test data lives here) --
                                filename unchanged by the project rename (purely internal, not user-facing)
email_service.py               Absentee-email sending, used by an APScheduler job in app.py
generate_report.py             Generates SecureFace_Attendance_Report.pdf (renamed from
                                SmartFace_Project_Report.pdf on 2026-11-07, along with all product-name
                                strings inside it -- "SmartFace AI" -> "SecureFace Attendance", the
                                "Developed By: Sofzenix Technologies" line removed). Its TECHNICAL CONTENT
                                was not rewritten and still describes the deleted rule-based engine (LBP
                                variance, Moiré/OLED screen detection, the removed randomized
                                turn_left/turn_right challenge) and the unverified 99.8% LFW claim. See
                                Known Issues -- this still needs a content rewrite, not just a rename.
liveness/__init__.py           Marks liveness/ as a subpackage
liveness/config.py              Constants copied from anti_spoof_attendance/src/config.py (thresholds,
                                image size, device auto-detect)
liveness/face_utils.py          FaceDetector + crop_face(), copied verbatim from anti_spoof_attendance
liveness/model.py               LivenessCNN / build_model() (MobileNetV3-Small + custom head)
liveness/predictor.py           LivenessPredictor, classify_three_way(), load_gray_zone_band()
liveness/transform.py           get_eval_transform() -- the exact inference-time preprocessing
liveness/weights/*              Trained checkpoint + gray-zone band (committed); BlazeFace .tflite
                                (gitignored, auto-downloaded)
scripts/create_admin.py         Interactive admin account creation, no hardcoded password
scripts/create_test_student.py  Interactive student login creation (face NOT pre-registered)
scripts/measure_pipeline_latency.py  Real latency measurement tool against a running server
static/css/style.css            Shared dark theme for all base.html-extending pages (1785 lines)
static/js/camera.js             Webcam capture (640px max width, JPEG q=0.85)
static/js/csrf.js               Global fetch() wrapper injecting X-CSRFToken from a <meta> tag
templates/base.html             Shared layout (sidebar, toast container) for admin/student pages
templates/scanner.html          Standalone full-screen scanner page (does NOT extend base.html)
templates/scan_log.html          Admin-only table of recent scan_events rows
templates/admin_dashboard.html, admin_login.html, analytics.html, attendance_history.html,
  employees.html, employee_dashboard.html, login.html, profile.html, register.html,
  settings.html                 Admin/student pages, all extend base.html
templates/attendance.html        ORPHANED -- grepped for every reference in app.py/templates/js;
                                  zero hits. No route renders it. Unverified why it still exists;
                                  flagged in Known Issues, not deleted (out of scope for this doc).
tests/test_crop_identical.py     Regression test: this project's crop output == anti_spoof_attendance's
utils/face_utils.py              ArcFace recognition engine (register_face, recognize_face_with_liveness,
                                  Face Mesh via MediaPipe Tasks API, Fernet encryption of embeddings)
utils/validators.py              Input validation (email w/ DNS MX check, password rules, employee_id, etc.)
.env.example                    Template for SECRET_KEY/DATABASE_URL/MAIL_*/FACE_ENCRYPTION_KEY
Procfile                        gunicorn launch command (Heroku/Render-style deployment)
runtime.txt / .python-version    Both say python-3.11.11 -- see mismatch note in Section 2
cloudflare_tunnel.md             Deployment doc for exposing the app via Cloudflare Tunnel (product-name
                                  references updated to SecureFace Attendance on 2026-11-07; still a
                                  Windows-specific guide, otherwise unverified/unchanged)
README.md                       Rewritten on 2026-11-07 to describe the real pipeline, the real trained
                                  model and its real results, and working run instructions -- no longer
                                  describes the deleted rule-based engine or the 99.8% claim.
SecureFace_Attendance_Report.pdf Static PDF, regenerated 2026-11-07 (renamed from
                                  SmartFace_Project_Report.pdf). generate_report.py's own technical
                                  content is still stale (see above) -- the PDF's product name is now
                                  correct but its architecture description is not.
docs/HANDOVER.md                 This file.
docs/pipeline_latency_2026-11-07.txt   Real latency measurement output (Section 4)
docs/pytest_result_2026-11-07_rename.txt      Real test-suite output (Section 4/6)
docs/auth_verification_2026-11-07.txt  Real curl output proving CSRF/auth gating (Section 6)
```

### The complete scan pipeline (verified by reading `app.py`'s `api_recognize_face()`, current
as of commit `5fc0663`)

1. **Decode** the incoming base64 JPEG frame (`data_uri_to_cv2_img`).
2. **BlazeFace detect** (`liveness/face_utils.py`'s `FaceDetector`, MediaPipe Tasks API,
   `min_confidence=0.5`). Zero faces → `NO_FACE`, pipeline stops. More than one face →
   `MULTI_FACE`, pipeline stops (session reset either way).
3. **Same-face consistency check**: if the largest face's bounding-box center moved more
   than 60% of its own size since the last frame (`FACE_MOVE_REJECT_RATIO = 0.6`), the
   session (smoothing history + any in-progress blink attempt) is reset — treated as a
   different physical subject.
4. **`crop_face()`** (`liveness/face_utils.py`, copied verbatim from `anti_spoof_attendance`):
   25% margin padding around the bbox (`FACE_CROP_MARGIN = 0.25`), then resized to **224×224**
   and normalized with ImageNet mean/std (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`) in
   `liveness/transform.py`'s `get_eval_transform()`. `tests/test_crop_identical.py` asserts
   this crop is byte-identical to `anti_spoof_attendance`'s own `crop_face()` output for the
   same input frame — passing as of this writing (Section 4/6).
5. **MobileNetV3-Small liveness classifier** (`liveness/predictor.py`'s `LivenessPredictor`):
   outputs a raw `P(spoof)` in [0,1] for this one frame.
6. **Per-session temporal smoothing**: `_smoothed_prob_spoof()` in `app.py` keeps a rolling
   average of the last 5 raw probabilities (`RECENT_PROBS_WINDOW = 5`) in the Flask session
   cookie — replaces the old global `_spoof_history` list that used to leak state between
   concurrent users.
7. **Three-way decision** (`classify_three_way()` in `liveness/predictor.py`):
   ```python
   if band_low <= prob_spoof <= band_high: return "UNCERTAIN"
   return "SPOOF" if prob_spoof >= threshold else "LIVE"
   ```
   `band_low = 0.024217108554277167`, `band_high = 0.9757828914457228` (from
   `liveness/weights/gray_zone_v2.json`, computed on the *validation* set only, never test —
   see `results/summary.json`'s `"gray_zone"` section). `threshold` is the admin-configurable
   `liveness_threshold` setting (DB `settings` table, default `'0.5'`). **So "LIVE" requires
   the smoothed P(spoof) to be below ~0.024 — a strict bar.** See Section 6 for why this
   matters (the UNCERTAIN investigation).
   - `SPOOF` → the in-progress blink attempt is cleared (session reset for `challenge_type`/
     `challenge_user`/`challenge_assigned_at`); pipeline stops here. Face Mesh/ArcFace never run.
   - `UNCERTAIN` → rejected for this frame only; the in-progress blink attempt is deliberately
     **not** cleared (fixed bug, see Section 5); pipeline stops here too.
   - `LIVE` → continue to step 8.
8. **Face Mesh + ArcFace** (`utils/face_utils.py`'s `recognize_face_with_liveness(frame_bgr,
   match_threshold=0.40)`): MediaPipe FaceLandmarker (478 landmarks) aligns the face via a
   5-point similarity transform to 112×112, ArcFace ONNX (`w600k_r50`) produces a 512-D
   embedding, matched against stored embeddings via cosine similarity. `match_threshold` is
   the admin-configurable `face_tolerance` setting (default `'0.45'`). Also returns real EAR
   (eye-aspect-ratio) and head-yaw from the mesh, and a `multi_face` flag if Face Mesh itself
   finds more than one face at this later stage (pipeline stops, `MULTI_FACE`).
9. **Silent blink check** (no longer a random multi-type challenge — see Section 5): if no
   match → `UNRECOGNIZED`. If matched, waits for one natural eyes-closed→eyes-open transition
   (`liveness_verified` flag, computed client-side in `scanner.html` from the real
   server-reported `eyes_closed` value each frame). An 8-second timeout
   (`CHALLENGE_TIMEOUT_SECONDS = 8`) silently restarts the attempt clock if no blink occurs.
10. **Attendance marking**: on a passed blink, inserts into `attendance` (status `Late` or
    `Present` depending on the admin-configurable late-cutoff setting) if not already marked
    today for that user; otherwise reports "already checked in today." Every stage reached is
    logged to `scan_events` (Section 3, DB tables) via `_log_scan_event()`.

### Every model used

| Model | Purpose | Source | File path |
|---|---|---|---|
| MobileNetV3-Small (custom head, 1 logit) | Liveness/spoof classification | **Trained** by the repo owner, in `anti_spoof_attendance` (training code not in this repo) | `liveness/weights/best_liveness.pth` |
| BlazeFace (short-range) | Face detection for the liveness stage | Pretrained (Google/MediaPipe) | `liveness/weights/blaze_face_short_range.tflite` (gitignored, auto-downloaded) |
| MediaPipe FaceLandmarker (478 pts + iris) | Face alignment + EAR/head-yaw for ArcFace stage | Pretrained (Google/MediaPipe) | `models/face_landmarker.task` (gitignored, auto-downloaded) |
| ArcFace `w600k_r50` (ONNX) | 512-D face recognition embeddings | Pretrained (InsightFace `buffalo_l` bundle) | `models/w600k_r50.onnx` (gitignored, auto-downloaded) |

### Database tables and columns (SQLite inline schema in `database/db.py`, mirrored in
`database/schema.sql` for Postgres)

- **`users`**: `id, employee_id (UNIQUE), name, email (UNIQUE), phone, department,
  password_hash, role CHECK IN ('admin','employee'), profile_photo, face_encoding (unused
  legacy BLOB), face_encodings (unused legacy TEXT), face_embedding (TEXT, Fernet-encrypted
  JSON), face_registered (INTEGER 0/1), created_at`.
- **`attendance`**: `id, user_id (FK), date, time, status CHECK IN ('Present','Absent','Late',
  'Leave'), method`.
- **`settings`**: `key (PK), value`. Defaults: `late_cutoff_hour='9'`, `late_cutoff_minute='0'`,
  `company_name='Sofzenix Technologies'`, `face_tolerance='0.45'`, `min_face_samples='5'`,
  `liveness_threshold='0.5'`, `email_enabled='0'`, `email_trigger_hour='18'`,
  `email_trigger_minute='0'`, `hr_email=''`.
- **`liveness_events`**: `id, session_id, decision CHECK IN ('LIVE','SPOOF','UNCERTAIN'),
  p_spoof (REAL), user_id, created_at`. One row per decision **change** per browser session
  (deduped), not per frame.
- **`scan_events`**: `id, session_id, stage CHECK IN ('NO_FACE','MULTI_FACE','FACE_MOVED',
  'SPOOF','UNCERTAIN','RECOGNIZED','UNRECOGNIZED','CHALLENGE_PASSED','ATTENDANCE_MARKED',
  'ALREADY_MARKED'), user_id, detail (TEXT), test_label (TEXT, dev-only), created_at`. One row
  per deepest pipeline stage reached per request (deduped for the continuous states).

### All routes and APIs with access level

| Route | Methods | Access | Purpose |
|---|---|---|---|
| `/` | GET | public | Scanner page (home) |
| `/login` | GET/POST | public (redirects if already a logged-in student) | Student login |
| `/admin_login` | GET/POST | public (redirects if already a logged-in admin) | Admin login |
| `/register` | GET/POST | public | Student self-registration |
| `/logout` | GET | public | Clears session |
| `/profile` | GET/POST | student (any logged-in user) | Profile editing |
| `/admin` | GET | admin | Admin dashboard |
| `/admin/dev/clear_today_attendance` | POST | admin **and** `FLASK_DEBUG=1` (hard-gated both ways) | Deletes today's attendance rows, for repeat live testing |
| `/admin/scan_log` | GET | admin | Lists most recent 300 `scan_events` rows |
| `/manage_employees` | GET/POST | admin | Student directory, add/delete |
| `/attendance_history` | GET | admin | Filterable attendance history |
| `/analytics` | GET | admin | Charts/stats |
| `/settings` | GET/POST | admin | Reads/writes the `settings` table |
| `/export_csv` | GET | admin | CSV export of attendance |
| `/employee` | GET | student | Student dashboard |
| `/scanner` | GET | public | Redirects to `/` |
| `/admin/wipe_model` | POST | admin | **Destructive**: wipes `face_registered`/`face_embedding` for **every** user and deletes **all** attendance rows for **everyone**. Pre-existing SmartFace route, not added by this merge. See Known Issues. |
| `/api/register_face` | POST | any logged-in user | Registers one face sample for the current user |
| `/api/recognize_face` | POST | public (no login — this is the scanner's own endpoint) | The full pipeline above |
| `/api/edit_attendance` | POST | admin | Edit a single attendance record |
| `/api/system_status` | GET | admin | Reports real liveness/ArcFace engine state |
| `/api/manual_attendance` | POST | admin | Manually mark attendance for a user |
| `/api/admin_add_employee` | POST | admin | Admin-created student account |
| `/api/search_employees` | GET | admin | Search box backing `/manage_employees` |
| `/api/attendance_feed` | GET | admin | Live feed for the dashboard |
| `/api/upload_photo` | POST | any logged-in user | Profile photo upload |
| `/api/send_test_email` | POST | admin | SMTP test |

CSRF protection (`Flask-WTF`'s `CSRFProtect(app)`) wraps **every** state-changing route above
by default — verified live on 2026-11-07 (`docs/auth_verification_2026-11-07.txt`): a POST to
`/login` without a token returned `400 "The CSRF token is missing."`, and the dev-only
clear-attendance route is rejected at the CSRF layer *before* its own `FLASK_DEBUG` check even
runs (double-gated). The admin-only JSON APIs (`search_employees`, `attendance_feed`,
`system_status`) all returned `401 {"error":"Unauthorized"}` with no session cookie, verified
in the same run.

---

## 4. REAL RESULTS (with source file for each number)

All numbers below are read directly from
`anti_spoof_attendance/results/summary.json` unless otherwise noted. That file documents its
own protocol differences explicitly (see its `"consistency_notes"` field) — read it before
quoting any number out of context.

- **Headline dataset result** (`summary.json` → `"headline"`, canonical protocol, n=3900,
  the deployed v2 MobileNetV3-Small model): **accuracy 98.69%, APCER 1.38%, BPCER 1.23%,
  HTER 1.31%**.
- **v1 → v2 crop-mismatch fix** (`summary.json` → `"v1_vs_v2"`, n=3900 both): v1 accuracy
  **94.54%**, HTER **5.46%** → v2 accuracy **98.69%**, HTER **1.31%**. v1 was trained on
  non-deployment-style crops; v2 was retrained on crops produced the *same* way the live
  pipeline crops faces, which is why it generalizes much better to the real crop shape.
- **Gray-zone band** (`summary.json` → `"gray_zone"`): selected on the *validation* set only
  (never test), `band_low=0.024217108554277167`, `band_high=0.9757828914457228`, targeting a
  5% uncertain fraction. Applying it to the test set drops HTER to 0.22% at 95.1% coverage.
- **Masked-face evaluation** (`summary.json` → `"masked_eval"`, synthetic surgical-mask
  overlay on real images, BlazeFace-only protocol): masked accuracy **98.42%**, HTER
  **1.57%**, vs. an unmasked-baseline HTER of 0.97% on the same detection protocol.
- **Measured pipeline speed**: re-measured live on 2026-11-07 against the actual running
  server with `scripts/measure_pipeline_latency.py` (45 real requests, 30 LIVE + 15 SPOOF
  sample images) — raw output saved to `docs/pipeline_latency_2026-11-07.txt`:
  - Full round-trip (wall clock): **median 80.6ms**, mean 75.2ms, min 23.8ms, max 443.7ms
    (the max is almost certainly a cold-start/first-request outlier, not a steady-state value).
  - Server-reported stages: BlazeFace detect median 3.9ms; liveness classify median 13.1ms;
    Face Mesh+ArcFace recognition (LIVE frames only) median 58.0ms.
  - **Note:** an earlier measurement in this same chat (not saved to a file at the time)
    reported ~50ms median under different machine load — treat the file above as the
    authoritative, re-verifiable number; run-to-run variance on this hardware is real and
    not yet characterized beyond these two data points.
- **Per-stage model latency/size** (`summary.json` → `"backbone_comparison"."latency_and_size"`):
  MobileNetV3-Small — 1,000,993 params, 3.95MB file, **CPU ONNX inference 2.76ms** (median,
  `onnx_max_abs_diff_vs_pytorch = 0.000187`, 20-run parity check).
- **Live test results recorded so far** (from the real `database/smartface.db` on this
  machine, confirmed 2026-11-07): 2 real users (`Yuvika`, admin; `Yogita`, student,
  `face_registered=1`), 1 real attendance row (`user_id=2`, `2026-11-02 14:50:30`, status
  `Late`), 65 rows in `scan_events` and 40 in `liveness_events` from the live camera test
  session on 2026-11-02. Stage breakdown from that session: `UNCERTAIN` ×20, `RECOGNIZED`
  ×19, `MULTI_FACE` ×15, `UNRECOGNIZED` ×4, `CHALLENGE_PASSED` ×3, `ALREADY_MARKED` ×2,
  `SPOOF` ×1, `ATTENDANCE_MARKED` ×1. Real p_spoof values recorded during that session for
  the real user's own face (glasses, normal room light) ranged roughly 0.024–0.17 — see
  Section 5/6 for why this matters.
- **Regression test**: `tests/test_crop_identical.py`, re-run 2026-11-07, **2 passed** (output
  in `docs/pytest_result_2026-11-07_rename.txt`).

### Numbers that must NOT be used, and why

These came from a Google-Stitch-generated landing-page **mockup** for an earlier phase of this
project (`anti_spoof_attendance/design/stitch/overview.html`) and were never real
measurements — they are placeholder numbers the design tool invented to make the mockup look
populated:

- **99.15% accuracy** (`overview.html` lines 155, 342) — real number is **98.69%**.
- **0.85% HTER** (line 349) — real number is **1.31%** (canonical) or 1.57% (masked).
- **26.7 FPS / ~37ms latency** (lines 40, 187) — real measured number is in
  `docs/pipeline_latency_2026-11-07.txt` (see above); the real per-model ONNX inference time
  is 2.76ms (not full-pipeline FPS).
- **61.2% baseline** (lines 325, 328) — there is no real LBP+SVM (or similar) baseline model
  in this project; this was a fabricated comparison bar. The real, legitimate comparison is
  the v1-vs-v2 crop-mismatch finding above (94.54% → 98.69%).
- **"FastAPI"** (line 368) — the actual framework is **Flask** (confirmed throughout `app.py`).

---

## 5. DECISIONS MADE AND WHY

All verified against the current code/commit history unless flagged otherwise.

- **Rule-based anti-spoofing removed, replaced by the trained model.** Confirmed:
  `utils/face_utils.py` has no `detect_screen_display()`, `compute_anti_spoof_score()`, or
  `analyze_3d_depth()` functions (commit `f74217b`'s message documents their removal). The
  trained MobileNetV3-Small is the sole spoof detector now.
- **Face Mesh and ArcFace run only after LIVE.** Confirmed in `app.py`: `recognize_face_with_
  liveness()` is only called inside the `if decision == "LIVE"` continuation, never for
  SPOOF/UNCERTAIN/NO_FACE/MULTI_FACE.
- **UNCERTAIN kept as a third state; only SPOOF clears the challenge.** Fixed in commit
  `23e46f1` after live-camera testing showed a real bug: UNCERTAIN frames (caused by natural
  motion blur) used to clear the in-progress blink attempt the same way SPOOF did, forcing a
  fresh random challenge on recovery. Now only a confirmed SPOOF clears it. No security
  regression: an UNCERTAIN frame never reaches recognition or advances the challenge anyway.
- **Same-face consistency and multi-face rejection.** Confirmed: `_face_moved_too_much()`
  (60% of face-size displacement threshold) resets the session; `len(detections) > 1` at the
  BlazeFace stage and `multi_face` from the Face Mesh stage both reject with `MULTI_FACE`.
- **"Employee" renamed to "student" in UI only; DB role stays `'employee'`.** Confirmed by a
  full-repo grep: every remaining `employee` string in code is a route name, Python/Jinja
  variable, DB/JSON field name, or function name — never user-visible text. The `role` column
  CHECK constraint still only allows `('admin', 'employee')`.
- **Mask handling — CORRECTION to the assumption in this document's own request.** No mask
  classifier exists, confirmed. **However**, the claim "low-match message asks to remove
  mask" is **not currently implemented** — I checked the actual UNRECOGNIZED-path message in
  `app.py` and it is the generic `"Face not recognized — please register first."` for *any*
  non-match (mask, bad lighting, or truly unregistered — all produce identical text). A
  mask-aware message was proposed in chat (2026-11-02) but explicitly **not built**, because
  MediaPipe Face Mesh is trained to always estimate full landmark positions even under
  occlusion (it doesn't flag "this region is covered"), so a reliable signal would need
  evaluation first — proposed next step: test per-landmark visibility/presence scores or a
  texture heuristic against the existing synthetic-mask images in `anti_spoof_attendance`
  (`results/gradcam/`, `masked_eval` in `summary.json`) before building anything. Do not
  write mask-detection code or mask-specific messages without doing that evaluation first and
  reporting it, per the project rules in Section 8.
- **Spoof messages never claim a specific attack type.** Confirmed: the SPOOF message is
  `"Spoof detected — please use your real face."` — binary classifier, can't distinguish
  photo/phone/screen, so it doesn't claim to.
- **Silent blink instead of a random blink/turn-left/turn-right challenge** (decided
  2026-11-02, after live testing). The UI used to show "Blink your eyes" / "Turn your head
  LEFT" etc. and randomized which one was required (defeats simple video replay, since replay
  footage can't react to an unpredictable live prompt). This was replaced with a silent wait
  for one natural blink, no visible instruction, 8-second timeout. **Honest tradeoff,
  recorded in chat and in the commit message (`5fc0663`)**: this removes that specific
  replay-attack defense — any video of the real person will blink on its own within a few
  seconds. The trained liveness model's LIVE/SPOOF classification is the primary defense
  regardless and is unaffected by this change.
- **Per-session (not global) liveness state.** Confirmed: all smoothing/challenge state lives
  in the signed Flask session cookie (`_get_liveness_session_id()`, `_smoothed_prob_spoof()`,
  etc.), replacing a deleted global `_spoof_history` list that leaked state between
  concurrent users.
- **Real Fernet encryption of face embeddings**, not a placeholder. Confirmed:
  `utils/face_utils.py` has `_fernet = Fernet(Config.FACE_ENCRYPTION_KEY)` and
  `_encrypt_embeddings`/`_decrypt_embeddings` (with legacy-plaintext fallback for pre-existing
  rows). `Config.FACE_ENCRYPTION_KEY` auto-generates and persists to `.face_key` if not set
  via environment.
- **`min_face_samples` and `face_tolerance` are real, used settings**, not dead config.
  Confirmed in `utils/face_utils.py`'s `register_face()` and `recognize_face_with_liveness()`.
- **"Days Present" counts Present + Late** (fixed 2026-11-02, live-test feedback). A Late
  check-in is still a day the student showed up; the student dashboard used to show "Days
  Present: 0" for a student with only a Late day. Verified against the real `Yogita` account:
  0 Present + 1 Late → dashboard now shows 1.
- **Admin creation has no hardcoded credentials.** Confirmed: `database/db.py`'s
  `_insert_defaults`/`_insert_defaults_pg` only create an admin if `ADMIN_EMAIL`/
  `ADMIN_PASSWORD` env vars are set; otherwise `scripts/create_admin.py` is the documented path.

---

## 6. CURRENT STATUS

### Done and verified (with how it was verified)
- Core pipeline integration (Steps 2-3 of the original merge plan) — commit `f74217b`.
  Verified via: `tests/test_crop_identical.py` (re-run 2026-11-07, 2 passed), live curl calls
  confirming SPOOF frames never reach Face Mesh/ArcFace, CSRF/auth gating (re-verified
  2026-11-07, `docs/auth_verification_2026-11-07.txt`), real Fernet-encrypted embeddings
  observed in the DB, `min_face_samples`/`face_tolerance` behavior observed directly.
- MediaPipe Tasks-API migration (the old `mp.solutions.face_mesh` API was removed in
  `mediapipe==0.10.30`, breaking the original SmartFace recognition engine) — fixed, server
  starts and recognizes faces correctly as of 2026-11-07.
- Dev test tooling (`scan_events` log, dev-only clear-attendance, test-student script,
  dev-only test-label dropdown) — commit `399200d`, re-verified live 2026-11-07.
- UNCERTAIN-challenge-reset bug fix — commit `23e46f1`, verified live by reproducing the exact
  scenario (LIVE+challenge → injected UNCERTAIN blip → recovery with the same challenge).
- Live-test-feedback round: silent blink + 8s timeout, 5 exact message strings, duplicate
  heading/subtitle fix, real face-box-from-detection-coordinates (backend data verified
  correct; **on-screen visual alignment not verified by me — no camera/browser in this
  environment, needs human confirmation**), "Days Present" fix (verified against the real
  Yogita account), test-label dropdown resize/reposition — commit `5fc0663`.

### Done but untested
- **Postgres duplicate-registration handling.** `app.py`'s `register()` route catches
  `(sqlite3.IntegrityError, psycopg2.IntegrityError)` with `conn.rollback()` — implemented,
  but **never run against a live Postgres database** (no Postgres instance available in any
  session so far). Explicitly flagged as untested per the project's own instruction.

### In progress / where it stopped
Nothing is mid-edit — every change described above is committed and the working tree is
clean. The **next** piece of work (not started) is the mask-detection evaluation proposed in
Section 5, pending the project owner's go-ahead.

### Pending tasks, in order (as given by the project owner)
a. ~~Live test feedback fixes~~ — **done**, see above (silent blink, messages, UNCERTAIN
   investigation, meters, face box, duplicate heading, Days-Present, test-label dropdown).
b. **Mask live test and report** — not started.
c. **Step 4 UI**: consistent design system across all pages (note: `base.html`/`style.css`
   already share one dark theme across admin/student pages — confirmed by reading both; the
   real gap is `scanner.html` being a separate standalone page, and no pages yet built for the
   items below), a single login page with a Student/Admin toggle (role checked server-side;
   today `/login` and `/admin_login` are two separate pages), an admin Spoof Log (today's
   `/admin/scan_log` is a plain table, not yet styled/featured as described), a Results page
   (to surface Section 4's real dataset numbers in-app), a Test Mode (must not mark real
   attendance — not built), the `liveness_threshold`/`face_tolerance` settings exposed in the
   settings UI (confirmed: `/settings` route reads/writes both already — verify the HTML form
   fields are present and wire any gaps), and a How It Works page (not built). **Not started.**
d. **Step 5 final verification** — not started (depends on c).
e. **Wrap-up**: project and folder renamed to SecureFace Attendance / `secureface-attendance`
   (done 2026-11-07); `README.md` rewritten to describe the real pipeline and results (done);
   `generate_report.py` renamed its output and all product-name strings, but its actual
   technical narrative (rule-based engine, randomized challenge, 99.8% claim) still needs a
   content rewrite, not just a rename. **Partially done.**

---

## 7. KNOWN ISSUES AND RISKS

- **The gray-zone band may be too strict for real webcam conditions.** Real p_spoof values
  recorded during the live test for the project owner's own face (glasses, normal indoor
  light) were mostly in the 0.03–0.17 range — genuinely low, but above the strict
  `band_low=0.024` needed for a LIVE classification, so the UI flickers between LIVE and
  UNCERTAIN frame to frame. This is a plausible domain-shift issue (the band was calibrated
  on CelebA-Spoof dataset photos, not real webcam frames with glasses glare) but **has not
  been fixed or re-calibrated** — the project owner explicitly said not to change thresholds
  just to make a test pass, and asked for findings first. Whoever picks this up should not
  silently retune `band_low` without discussing it.
- **`/admin/wipe_model` is highly destructive and pre-existing (not added by this merge).**
  It deletes `face_embedding`/`face_registered` for **every** user and **all** attendance
  rows for **everyone**, with no confirmation dialog visible in the route code itself (any
  confirmation would have to be client-side JS, not verified here). Treat as a live landmine
  during any further testing.
- **Python version mismatch**: the real venv runs 3.13.15; `runtime.txt`/`.python-version`
  declare 3.11.11. Unreconciled.
- **`mediapipe` is pinned exactly to `0.10.30`** (not `>=`) — `requirements.txt` documents why
  (versions `>0.10.30` crash on Apple M-series + Metal with "Service is unavailable"). Do not
  bump this without re-testing on the actual target hardware.
- **SSL/model-download issue seen during setup** (not a code bug): Python's `urllib`
  (Homebrew python) failed with `SELF_SIGNED_CERT_IN_CHAIN` downloading model files on this
  machine at one point; `curl` worked fine. If a fresh machine hits this, pre-download the
  three gitignored model files listed in Section 2 via `curl` instead of relying on the app's
  own auto-download.
- **`templates/attendance.html` is orphaned** — zero references anywhere in `app.py`,
  templates, or JS (verified by grep). Not deleted; unclear if intentional.
- **Face-box visual alignment is backend-verified only.** The `face_box` field returned by
  `/api/recognize_face` has been confirmed to contain correct pixel coordinates, but whether
  `scanner.html`'s on-screen box actually lands on the real face (accounting for the mirrored,
  letterboxed video preview) has **not** been visually confirmed — no camera or browser
  available in any session so far. Get a human to look at it before trusting it.
- **Mask detection does not exist.** Any UI or doc claiming mask-aware behavior beyond "no
  mask classifier" is wrong — see Section 5's correction.
- **README.md and generate_report.py are stale**, describing the deleted rule-based engine
  and at least one unverified accuracy claim ("99.8% LFW accuracy" — not sourced from any
  result file found in this repo or `anti_spoof_attendance`). Don't trust either document for
  current-system claims until Section 6e is done.
- **Postgres path is entirely untested** (see Section 6).
- **No camera/browser available in any AI coding session so far** — anything requiring visual
  confirmation (face-box alignment, actual blink detection UX, general look-and-feel) needs a
  human with a webcam, not just curl/pytest.

---

## 8. PROJECT RULES (the next AI must follow these)

These are standing instructions from the project owner across this entire project, verified
consistent with everything found in the commit history and chat record:

1. **No fake data, placeholder logic, mock results, or invented numbers.** Every number shown
   in the UI or written in docs must come from a real file or a real measurement you can
   point to. If you can't verify it, say so explicitly — do not round up, estimate, or guess
   "for now."
2. **Never change thresholds (gray-zone band, liveness_threshold, face_tolerance, etc.) just
   to make your own test pass or to make a specific face "work."** If a real threshold looks
   wrong, report what you found with real data and wait for a decision.
3. **Report findings and wait for approval before risky/ambiguous changes** — especially
   anything affecting detection thresholds, security checks (CSRF, auth, encryption), or data
   that could be destructive (see `/admin/wipe_model`).
4. **Don't rebuild working parts from scratch; restyle and fix instead.** The existing
   admin/student page structure and `base.html`/`style.css` theme already work — Step 4 is
   about consistency and new pages, not a rewrite.
5. **Never modify the two original project folders** (`anti_spoof_attendance`,
   `Smart-Attendance-System-main`). Work only in this merged copy.
6. **Reuse `crop_face()` and `FaceDetector` verbatim from `anti_spoof_attendance` — the crop
   must match training exactly** (25% margin, 224×224, ImageNet normalization). If you ever
   touch `liveness/face_utils.py`, re-run `tests/test_crop_identical.py` and confirm it still
   passes before committing.
7. **Test and show proof after every change.** This project's history is full of "implemented
   X, verified via curl/pytest/sqlite3 output: ..." — keep that pattern. A change without a
   real, reproducible verification step is not considered done.
8. **The scanner's spoof message must never claim a specific attack type** (photo, phone,
   screen, etc.) — the classifier is binary and genuinely cannot tell them apart. Don't add
   that claim back in for a "nicer" message.
