# HANDOVER: SecureFace Attendance

Updated 2026-11-19 to the final state after the Task 1–4 and Phase 1–5 work. Facts here
were checked against the code, the git history, result files, or a real run (cited).
Anything that could not be checked that way is marked **unverified**. Re-check anything
load-bearing; code changes after this date are not reflected.

---

## 1. Summary

A face-recognition attendance system for students with a **self-trained liveness
(anti-spoofing) model** in front of recognition. A student stands in front of the
**common scanner** (an admin-approved device). The system:

1. Checks the face is a real, live person, not a photo, phone or screen (trained MobileNetV3-Small).
2. Recognizes who it is (ArcFace).
3. Waits for one natural blink.
4. Marks them Present or Late, once per day.

Merged from two of the owner's own projects; both originals must never be modified:
- `/Users/yuvikamalhotra/Documents/project/anti_spoof_attendance`: source of the trained
  liveness model, its real results, and the crop/detector code reused verbatim.
- `/Users/yuvikamalhotra/Documents/Projects/Smart-Attendance-System-main`: the original
  "SmartFace" Flask app.

**Folder:** `/Users/yuvikamalhotra/Documents/project/secureface-attendance`
**Repo:** `github.com/Yuvika687/secureface-attendance`, branch `main`
**Python:** 3.13 (venv runs 3.13.15; `.python-version` and `runtime.txt` now say 3.13.15)

### Three parts

| Part | Access | Contents |
|---|---|---|
| Common scanner `/` | No login; **approved scanner device only** | Face → liveness → recognition → silent blink → attendance. Success shows name, student ID and "Attendance marked" / "Already marked today". |
| Student pages | Student login | `/register` (create account), `/employee` (dashboard: own attendance, face registration popup, "Present: X, of which Late: Y"), `/profile` |
| Admin pages | Admin login | Dashboard, Students (Reset face), Attendance, Analytics, Settings (incl. Scanner Devices), Pipeline View, Test Mode, Spoof Log, Scan Log, Results |

Public: `/login` (single page, Student/Admin toggle), `/how_it_works`.

---

## 2. How to run

```bash
cd /Users/yuvikamalhotra/Documents/project/secureface-attendance
./venv/bin/pip install -r requirements.txt        # venv already exists
./venv/bin/python scripts/create_admin.py           # interactive, no hardcoded password
./venv/bin/python app.py                            # http://localhost:5000
FLASK_DEBUG=1 ./venv/bin/python app.py              # + dev tools (scanner test-label dropdown,
                                                    #   /admin/dev/clear_today_attendance)
./venv/bin/pytest -v                                # 67 tests (pytest.ini: testpaths = tests)
```

**After the first start, approve a scanner.** Log in as admin on the scanner computer →
Settings → Scanner Devices → **Set this device as scanner**. Until then `/` returns 403
"This device is not an approved scanner." The approval is a cookie for one host name,
so `localhost` and an ngrok URL each need their own approval.

- Port 5000 can collide with macOS AirPlay Receiver (`lsof -ti:5000 | xargs kill -9`).
- Camera access works on `http://localhost` without HTTPS. For other devices use ngrok
  (`docs/ngrok.md`). The old Cloudflare Tunnel guide was removed.
- Model files: `liveness/weights/best_liveness.pth` and `gray_zone_v2.json` are committed.
  BlazeFace (`liveness/weights/blaze_face_short_range.tflite`), `models/face_landmarker.task`
  and `models/w600k_r50.onnx` are auto-downloaded (gitignored). If urllib hits
  `SELF_SIGNED_CERT_IN_CHAIN`, download them with `curl` to those paths.
- `scripts/measure_pipeline_latency.py` needs `SCANNER_DEVICE_TOKEN=<scanner_device cookie value>`.
- `./venv/bin/python generate_report.py` regenerates `SecureFace_Attendance_Report.pdf`
  (numbers read from `training/results/summary.json` and the latency file at build time).

**Existing databases are migrated at startup** (`app.py`, the block after "Initialize
Database"): it adds the new tables (`scanner_devices`, `admin_audit_log`,
`test_mode_results`) and columns (`scan_events.p_spoof_frame/p_spoof_avg/iou`,
collection columns on `test_mode_results`). All of this work was verified on scratch
copies of the real DB. The real `database/smartface.db` gets these on its next start.

---

## 3. Architecture

### Scan pipeline (`app.py` → `api_recognize_face()`)

1. **Device check:** `_current_scanner_device()` looks up the SHA-256 of the
   `scanner_device` cookie in `scanner_devices` (not revoked). Otherwise 403
   `DEVICE_NOT_APPROVED`.
2. Decode the frame. **BlazeFace** (`liveness/face_utils.py`): 0 faces → `NO_FACE`,
   more than 1 → `MULTI_FACE` (session reset).
3. **Same-face tracking by IoU:** `_face_box_iou(prev, cur)`. IoU < `SAME_FACE_MIN_IOU` (0.25)
   → `FACE_MOVED`, session reset. The 0.25 equals the old rule's cutoff (center moved >
   0.6 × face size) for a same-size box moving sideways. It also catches sudden size
   changes the old rule ignored. Both real logged jumps are still caught (IoU 0.182 and 0.034).
4. **`crop_face()`:** 25% margin, then Resize 224×224 + ImageNet normalization. Byte-identical to
   the training crop (`tests/test_crop_identical.py`).
5. **MobileNetV3-Small** → raw P(spoof) per frame; 5-frame average per Flask session.
6. **Three-way decision** (`classify_three_way`): band `[0.024217…, 0.975782…]` from
   `liveness/weights/gray_zone_v2.json` → UNCERTAIN; otherwise SPOOF if ≥ `liveness_threshold`
   (setting, default 0.5), else LIVE. Only SPOOF clears an in-progress blink attempt.
7. **Face Mesh + ArcFace** (`utils/face_utils.py`), only on LIVE. Cosine match with
   `face_tolerance` (default 0.45).
8. If a **student is logged in** in this browser, only their own face may proceed; another
   registered face → "This face doesn't match this account". (Kept from Task 1; attendance
   normally happens on the logged-out common scanner.)
9. **Silent blink, verified by the server** (`_advance_blink`, per session): the server's own
   Face Mesh `eyes_closed` reading (EAR below the existing 0.21 cut-off) must go open → closed →
   open for the same matched user within 8 s (`CHALLENGE_TIMEOUT_SECONDS`). A closed-eye frame
   that fails recognition still counts for the same tracked face; only a recognized open frame
   completes it. Any `liveness_verified` flag from the browser is ignored (tested in
   `tests/test_server_blink.py`).
10. **Attendance:** Present/Late by the cut-off setting, once per user per day.

Messages: "Attendance marked", "Already marked today", "Face not recognized — if you're
wearing a mask, please remove it and try again." (shown on every non-match; there is no
mask detector), "This face doesn't match this account", "Spoof detected — please use your
real face.", "Hold still — checking again.", "Only one person at a time.", "Step into the frame."

Every stage reached is logged to `scan_events` (deduplicated for repeating states) with
`p_spoof_frame`, `p_spoof_avg` (after the model ran) and `iou` (once a previous box exists).

### Models

| Model | Purpose | Source | File |
|---|---|---|---|
| MobileNetV3-Small + head (1 logit) | Liveness | **Trained** (`training/`) | `liveness/weights/best_liveness.pth` |
| BlazeFace short-range | Detection | Pretrained (MediaPipe) | `liveness/weights/blaze_face_short_range.tflite` |
| MediaPipe Face Landmarker (478 pts) | Alignment, EAR | Pretrained | `models/face_landmarker.task` |
| ArcFace `w600k_r50` ONNX | Identity | Pretrained (InsightFace) | `models/w600k_r50.onnx` |

### Files (main ones)

```
app.py                     all routes, scan pipeline, Pipeline View, Test Mode, Spoof Log, Results, device approval
config.py                  secret key, DB auto-detect, mail, Fernet key, uploads
database/db.py, schema.sql SQLite inline schema + Postgres schema (mirrors each other)
liveness/                  inference: face_utils.py (FaceDetector, crop_face), model.py, predictor.py, transform.py, weights/
utils/face_utils.py        Face Mesh + ArcFace, register_face() with owner checks, Fernet-encrypted embeddings
utils/gradcam.py           Grad-CAM (copied verbatim from training/src/gradcam.py, minus its __main__ demo)
utils/validators.py        input validation
templates/                 base.html (shared layout), login.html (single login), scanner.html (standalone),
                           scanner_not_approved.html, how_it_works.html (standalone public), pipeline_view.html,
                           test_mode.html, spoof_log.html, results.html, scan_log.html, settings.html, employees.html,
                           employee_dashboard.html, admin_dashboard.html, analytics.html, attendance_history.html,
                           profile.html, register.html
static/css/style.css       shared dark theme + status colors (--state-live/spoof/uncertain/noface)
scripts/                   create_admin, create_test_student, reset_password, measure_pipeline_latency
tests/                     conftest.py (temp DB per test), helpers.py, 7 test files incl. test_crop_identical.py
training/                  model training/eval code + real results (not imported at runtime)
docs/                      this file, domain_adaptation_plan.md, ngrok.md, dated evidence logs
generate_report.py         builds SecureFace_Attendance_Report.pdf from real files
```

Removed: `templates/attendance.html` (was orphaned), `templates/admin_login.html` (replaced by the
single login page), `cloudflare_tunnel.md` (replaced by `docs/ngrok.md`).

### Database tables

- `users`: `id, employee_id, name, email, phone, department, password_hash, role ('admin'|'employee'),
  profile_photo, face_encoding/face_encodings (unused legacy), face_embedding (Fernet-encrypted JSON),
  face_registered, created_at`. Students are role `'employee'` in the DB; the UI says "student".
- `attendance`: `id, user_id, date, time, status ('Present'|'Absent'|'Late'|'Leave'), method`.
- `settings`: key/value (`liveness_threshold`, `face_tolerance`, `min_face_samples`, late cut-off, email…).
- `liveness_events`: one row per LIVE/SPOOF/UNCERTAIN decision change per session.
- `scan_events`: `id, session_id, stage, user_id, detail, test_label, p_spoof_frame, p_spoof_avg, iou, created_at`.
- `scanner_devices`: `id, token_hash (SHA-256), label, approved_by, user_agent, created_at, revoked_at`.
- `admin_audit_log`: `id, admin_user_id, admin_email, action, detail, ip_address, created_at`
  (actions: `WIPE_MODEL`, `SCANNER_APPROVED`, `SCANNER_REVOKED`, `RESET_FACE`).
- `test_mode_results`: `id, test_session, category, decision, p_spoof_frame, p_spoof_avg, admin_user_id,
  person_id, glasses, lighting, collection_session, crop_path, created_at`.

`created_at` is SQLite `CURRENT_TIMESTAMP` (UTC); the log pages label times as UTC.

### Routes

| Route | Methods | Access |
|---|---|---|
| `/` | GET | approved scanner device (403 page otherwise) |
| `/api/recognize_face` | POST | approved scanner device |
| `/login` | GET, POST | public; Student/Admin toggle, real role checked → "Invalid credentials" on mismatch |
| `/admin_login` | GET, POST | GET redirects to `/login?role=admin`; POST logs in as admin |
| `/register`, `/how_it_works`, `/logout` | | public |
| `/scanner` | GET | redirects to `/` |
| `/employee`, `/profile`, `/api/register_face`, `/api/upload_photo` | | logged-in user |
| `/admin`, `/manage_employees`, `/attendance_history`, `/analytics`, `/settings`, `/export_csv` | | admin |
| `/admin/pipeline_view`, `/api/admin/pipeline_frame` | GET / POST | admin; never marks attendance or saves images |
| `/admin/test_mode`, `/api/admin/test_scan`, `/api/admin/test_mode/new_session`, `/api/admin/test_mode/metrics` | | admin; never runs recognition or marks attendance |
| `/admin/spoof_log` (`?export=csv`), `/admin/scan_log` | GET | admin |
| `/admin/results`, `/admin/results/plot/<name>` | GET | admin (plots: whitelisted files under `training/`) |
| `/admin/scanner_devices/approve`, `/admin/scanner_devices/<id>/revoke` | POST | admin, audit-logged |
| `/admin/students/<id>/reset_face` | POST | admin, audit-logged |
| `/admin/wipe_model` | POST | admin; needs JSON `{"confirm": "WIPE"}`, audit-logged |
| `/admin/dev/clear_today_attendance` | POST | admin **and** `FLASK_DEBUG=1` |
| `/api/edit_attendance`, `/api/manual_attendance`, `/api/admin_add_employee`, `/api/search_employees`, `/api/attendance_feed`, `/api/system_status`, `/api/send_test_email` | | admin |

CSRF (`Flask-WTF`) covers every state-changing route. This is tested (`tests/test_csrf.py`).

---

## 4. Real results

From `training/results/summary.json` (CelebA-Spoof test set, n = 3900, deployed v2 model):

- **Accuracy 98.69%, APCER 1.38%, BPCER 1.23%, HTER 1.31%.**
- **v1 → v2 crop fix:** accuracy 94.54% → 98.69% (HTER 5.46% → 1.31%).
- **Measured pipeline latency** (`docs/pipeline_latency_2026-11-07.txt`, 45 requests): median
  round trip 80.6 ms. Repeat runs during this work against scratch copies gave medians of
  77–87 ms (per-phase checks; not saved as files).
- `summary.json` also has the backbone comparison, the gray-zone results and the masked-face
  evaluation. Admin → Results and the PDF report read them from there.

**Numbers that must NOT be used** (invented by a design-mockup tool in
`anti_spoof_attendance/design/stitch/overview.html`): 99.15% accuracy, 0.85% HTER, 26.7 FPS /
~37 ms, 61.2% "baseline", 99.8% LFW accuracy, "FastAPI".

**Live webcam results:** none recorded as files yet. Test Mode stores real per-frame decisions in
`test_mode_results`, and Admin → Results shows them in a separate "Live test results" section.

---

## 5. Decisions and why

- **Trained model replaces rule-based anti-spoofing.** The old texture/screen/depth rules
  were deleted.
- **Recognition only after LIVE.** SPOOF/UNCERTAIN never reach Face Mesh/ArcFace.
- **UNCERTAIN doesn't clear a blink attempt; SPOOF does.** Fixes a real bug found in live testing.
- **Silent blink** instead of a random visible challenge. Deliberate tradeoff: a replayed
  video would blink too. The liveness model is the primary defense.
- **Attendance only at the common scanner, on approved devices.** Scan Face links were removed
  from student pages (Task 3), and only admin-approved browsers can run the scanner (Task 4).
- **IoU tracking replaces the center-distance rule** (threshold 0.25, reasoning in Section 3).
- **Registration owner checks** (`register_face`). A new sample must average ≥ `face_tolerance`
  similarity to the user's stored samples (skipped for the first one) and must not reach
  `face_tolerance` against any other fully registered user. Otherwise: "Only the account
  owner should be in front of the camera."
- **Single login page.** The toggle only states intent; the DB role decides. Unknown
  email, wrong password and wrong role all give the same "Invalid credentials".
- **Status colors carry meaning only:** green LIVE, red SPOOF, blue UNCERTAIN, orange no face / more than one face.
- **Test Mode metrics.** One row per frame with exactly one face, using the scanner's own
  5-frame-average decision. APCER = attack frames judged LIVE. BPCER = real-person frames
  judged SPOOF. UNCERTAIN is counted separately. Per-attack HTER = (that APCER + overall
  BPCER) / 2.
- **Grad-CAM** on `feature_extractor.features[12]` (last conv block, 576×7×7, confirmed by a
  forward hook). Computed only when Explain is pressed, on a private deep copy of the
  model, so hooks never touch the live scanner's model.
- **No thresholds were changed** in any of this work.

---

## 6. Status

### Done and verified (scratch-copy server + pytest)
- Task 1: wipe confirmation + audit log; per-account check; clear messages; orphan template
  deleted; Python version files.
- Task 2: Lavya's face samples cleared in the real DB (backup:
  `database/backups/smartface_pre-task2_20261118.db`); registration owner checks.
- Task 3: Scan Face removed from student pages; name + ID on success; "Register your face"
  banner.
- Task 4: per-frame scores in the scan log; scanner device approval; IoU tracking.
- Phase 1: Pipeline View + Grad-CAM.
- Phase 2: UI (single login, colors, scanner box label, meters removed, Reset face, Scan
  Log columns, Spoof Log, Results, Test Mode, How It Works).
- Phase 3: collection mode + `docs/domain_adaptation_plan.md`.
- Phase 4: 59 tests (now 67 with the server-side blink tests).
- Phase 5: report rewrite, docs.

### Not verified (needs a person with a camera and a browser)
- On-screen face-box alignment and label on the mirrored, letterboxed video (backend
  coordinates are verified).
- Real blink detection UX; Pipeline View / Test Mode camera loops; Grad-CAM overlay
  appearance.
- Mobile layout on a real phone (CSS breakpoints added; not seen on a device).
- Real phone/printed-photo attacks against the webcam (only dataset images were used here).

### Pending
- **Domain adaptation Part 2.** Collect sessions 1 and 2 first (`docs/domain_adaptation_plan.md`),
  then recalibrate on session 1 and report on session 2. Nothing has been recalibrated.
- Postgres path: implemented, never run against a live Postgres.

---

## 7. Known issues and risks

- **Domain shift.** On a webcam, real faces often land just inside the UNCERTAIN band (logged
  values about 0.024–0.17), and a phone photo was judged LIVE once (average 0.0238 at
  2026-11-18 11:22 UTC). Not fixed; see the plan.
- **Logged-in student on the scanner.** If a student is logged in on the scanner browser, only
  their face can mark attendance there. Log out on the kiosk.
- **`mediapipe==0.10.30` pinned exactly.** Newer versions crash on Apple M-series with Metal.
- **Free ngrok URLs change** on restart, so the scanner must be re-approved for each new URL.
- Collected face crops (`data/domain_adaptation/`) are real people's faces. They are
  gitignored; delete them when done.

---

## 8. Project rules (from the owner)

1. No fake data, placeholder logic or invented numbers. Every number must trace to a real file or measurement.
2. Never change thresholds to make a test or a specific face pass. Report findings with data instead.
3. Report and wait before risky or ambiguous changes (thresholds, security checks, anything destructive).
4. Never modify `anti_spoof_attendance` or `Smart-Attendance-System-main`.
5. Don't modify `liveness/` or the ArcFace matching logic unless the task is about them. If
   `liveness/face_utils.py` is touched, re-run `tests/test_crop_identical.py`.
6. Show real, reproducible verification after every change. Test on a scratch copy of the DB,
   never the real `database/smartface.db`.
7. Before each task, commit and tag `pre-<task>-YYYYMMDD`. After it: pytest, start the app,
   check the scan pipeline on the sample images, commit, push.
8. The spoof message never names an attack type (the classifier is binary).
9. No co-author lines in commits.
