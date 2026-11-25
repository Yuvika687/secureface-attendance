import os
import io
import csv
from calendar import monthrange
import uuid
import sqlite3
import time
import hashlib
import secrets
import math
import base64
import copy
import threading
import json
import re
import cv2
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, session, jsonify, flash, url_for, Response, send_file, abort
from flask_wtf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
from database.db import get_db_connection, init_db, get_setting, set_setting, _DEFAULT_SETTINGS
from utils.face_utils import register_face, recognize_face_with_liveness, data_uri_to_cv2_img, REGISTER_REJECTED_IDENTITY
from utils.validators import (
    validate_registration, validate_profile_update, validate_name,
    validate_employee_id, validate_email, validate_phone, validate_password,
    validate_department, validate_status, validate_date, validate_time,
    validate_profile_photo, generate_secure_password, ALLOWED_DEPARTMENTS
)
from liveness.face_utils import FaceDetector, crop_face
from liveness.mask import mask_probability
from liveness.predictor import LivenessPredictor, classify_three_way, load_gray_zone_band

try:
    import psycopg2
    _IntegrityErrors = (sqlite3.IntegrityError, psycopg2.IntegrityError)
except ImportError:
    _IntegrityErrors = (sqlite3.IntegrityError,)

app = Flask(__name__)
app.config.from_object(Config)

# Debug mode: only on when explicitly requested (local development), never
# the previous hardcoded `debug=True`. Set FLASK_DEBUG=1 for local dev.
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "0") == "1"

csrf = CSRFProtect(app)
# All <form method="POST"> templates carry a hidden csrf_token field, and
# every fetch() POST goes through a global wrapper (static/js/csrf.js) that
# attaches the X-CSRFToken header read from a <meta> tag -- Flask-WTF
# accepts the token from either place. Nothing is blanket-exempted: every
# state-changing endpoint, including the JSON ones, is actually protected.

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize Database
if not Config.USE_POSTGRES:
    if not os.path.exists(Config.DATABASE_URI):
        os.makedirs(os.path.dirname(Config.DATABASE_URI), exist_ok=True)
        init_db()
    else:
        # Migration for existing DBs: ensure newer columns/tables/settings exist.
        conn = get_db_connection()
        for col, coltype in [('profile_photo', 'TEXT'), ('face_embedding', 'TEXT')]:
            try:
                conn.execute(f"SELECT {col} FROM users LIMIT 1")
            except Exception:
                try:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col} {coltype}")
                    conn.commit()
                except Exception:
                    pass
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS liveness_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('LIVE', 'SPOOF', 'UNCERTAIN')),
                    p_spoof REAL NOT NULL,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scan_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    stage TEXT NOT NULL CHECK(stage IN (
                        'NO_FACE', 'MULTI_FACE', 'FACE_MOVED', 'SPOOF', 'UNCERTAIN',
                        'RECOGNIZED', 'UNRECOGNIZED', 'CHALLENGE_PASSED',
                        'ATTENDANCE_MARKED', 'ALREADY_MARKED'
                    )),
                    user_id INTEGER,
                    detail TEXT,
                    test_label TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS admin_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_user_id INTEGER NOT NULL,
                    admin_email TEXT,
                    action TEXT NOT NULL,
                    detail TEXT,
                    ip_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute("SELECT test_label FROM scan_events LIMIT 1")
        except Exception:
            try:
                conn.execute("ALTER TABLE scan_events ADD COLUMN test_label TEXT")
                conn.commit()
            except Exception:
                pass
        for col, coltype in (('person_id', 'TEXT'), ('glasses', 'TEXT'), ('lighting', 'TEXT'),
                             ('collection_session', 'INTEGER'), ('crop_path', 'TEXT')):
            try:
                conn.execute(f"SELECT {col} FROM test_mode_results LIMIT 1")
            except Exception:
                try:
                    conn.execute(f"ALTER TABLE test_mode_results ADD COLUMN {col} {coltype}")
                    conn.commit()
                except Exception:
                    pass
        for col in ('p_spoof_frame', 'p_spoof_avg', 'iou'):
            try:
                conn.execute(f"SELECT {col} FROM scan_events LIMIT 1")
            except Exception:
                try:
                    conn.execute(f"ALTER TABLE scan_events ADD COLUMN {col} REAL")
                    conn.commit()
                except Exception:
                    pass
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_mode_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_session TEXT NOT NULL,
                category TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('LIVE', 'SPOOF', 'UNCERTAIN', 'NO_FACE', 'MULTI_FACE')),
                p_spoof_frame REAL,
                p_spoof_avg REAL,
                admin_user_id INTEGER,
                person_id TEXT,
                glasses TEXT,
                lighting TEXT,
                collection_session INTEGER,
                crop_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
            conn.commit()
        except Exception:
            pass
        try:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS scanner_devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_hash TEXT NOT NULL UNIQUE,
                    label TEXT,
                    approved_by INTEGER,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    revoked_at TIMESTAMP
                )
            ''')
            conn.commit()
        except Exception:
            pass
        for key, value in _DEFAULT_SETTINGS.items():
            try:
                conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
            except Exception:
                pass
        conn.commit()
        conn.close()
else:
    init_db()

# --- Liveness model (loaded once at startup, not per-request) ---
# This is the trained MobileNetV3-Small classifier from anti_spoof_attendance,
# reused verbatim via the liveness/ subpackage -- it replaces SmartFace's
# deleted rule-based anti-spoofing engine as the sole spoof detector.
print("[Liveness] Loading trained liveness model + BlazeFace detector...")
_liveness_detector = FaceDetector()
_liveness_predictor = LivenessPredictor()
_gray_zone_band = load_gray_zone_band()
print(f"[Liveness] Ready. Gray-zone band: {_gray_zone_band}")

# --- APScheduler for auto email notifications ---
scheduler = None
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from email_service import send_absentee_emails

    email_enabled = get_setting('email_enabled', '0')
    trigger_hour = int(get_setting('email_trigger_hour', '18'))
    trigger_minute = int(get_setting('email_trigger_minute', '0'))

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        send_absentee_emails,
        'cron',
        hour=trigger_hour,
        minute=trigger_minute,
        id='absentee_emails',
        replace_existing=True,
        misfire_grace_time=3600
    )
    if email_enabled == '1':
        scheduler.start()
        print(f"[Scheduler] Email notifications enabled — triggers at {trigger_hour}:{trigger_minute:02d}")
    else:
        print("[Scheduler] Email notifications disabled (enable in Settings)")
except Exception as e:
    print(f"[Scheduler] Could not initialize: {e}")


def get_late_cutoff():
    hour = int(get_setting('late_cutoff_hour', '9'))
    minute = int(get_setting('late_cutoff_minute', '0'))
    return hour, minute


def is_late(time_str):
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        hour, minute = get_late_cutoff()
        return t.hour > hour or (t.hour == hour and t.minute > minute)
    except Exception:
        return False


# ============================================================
#  LIVENESS PIPELINE HELPERS (per-session, not global)
# ============================================================
# Everything here lives in Flask's signed session cookie, scoped to one
# browser -- not a module-level global like the deleted _spoof_history,
# which leaked state between concurrent users/sessions.

RECENT_PROBS_WINDOW = 5          # frames averaged for the smoothed P(spoof)
SAME_FACE_MIN_IOU = 0.25         # face-box IoU vs. the previous frame below this => treat as a different physical subject
# Word the admin must type to confirm /admin/wipe_model (sent as JSON "confirm").
WIPE_CONFIRM_WORD = "WIPE"

# Scanner result messages (one per outcome).
MSG_ATTENDANCE_MARKED = "Attendance marked"
MSG_ALREADY_MARKED = "Already marked today"
MSG_FACE_NOT_RECOGNIZED = "Face not recognized. Check your enrollment and try again."
MSG_ACCOUNT_MISMATCH = "This face doesn't match this account"
MSG_DEVICE_NOT_APPROVED = "This device is not an approved scanner."

# Mask check: block only when the mask model is at least this sure on this many frames in a row.
MASK_MIN_PROB = 0.90
MASK_STREAK_FRAMES = 3

# Scanner-device approval: an admin approves a browser; it keeps a random
# token in this cookie (only the token's SHA-256 is stored server-side).
SCANNER_COOKIE_NAME = "scanner_device"
SCANNER_COOKIE_MAX_AGE = 5 * 365 * 24 * 3600   # stays approved until an admin revokes it

# Random active challenge (second check after the liveness model): after LIVE +
# recognized, the server asks for one random head turn, then one natural blink, all by
# the same tracked face within CHALLENGE_TIMEOUT_SECONDS. Everything is measured by the
# server from Face Mesh (utils/face_utils.compute_liveness_metrics); nothing the browser
# sends is trusted. A flat photo can't produce a real yaw change and can't blink.
CHALLENGE_TIMEOUT_SECONDS = 15   # turn + blink must both finish inside this window
# head_yaw = (nose x - eye-centre x) / eye distance, on the unmirrored camera frame.
# Positive = the person turned to THEIR left (checked on real photos, 2026-11-25).
# 0.08 / 0.12 are the values the original SmartFace turn challenge used.
HEAD_STRAIGHT_MAX_YAW = 0.08     # |yaw| below this = looking straight
HEAD_TURN_MIN_YAW = 0.12         # |yaw| above this (after straight) = turned
TURN_INSTRUCTIONS = {"left": "Turn your head left", "right": "Turn your head right"}
BLINK_INSTRUCTION = "Now blink once"


def _get_liveness_session_id():
    """A stable per-browser-session id for the liveness_events log, distinct
    from the Flask auth session (the scanner page has no login)."""
    if 'liveness_session_id' not in session:
        session['liveness_session_id'] = uuid.uuid4().hex
    return session['liveness_session_id']


def _smoothed_prob_spoof(raw_prob):
    """Per-session rolling average of the last RECENT_PROBS_WINDOW raw
    P(spoof) values -- replaces the deleted global _spoof_history list."""
    recent = session.get('recent_probs', [])
    recent.append(raw_prob)
    recent = recent[-RECENT_PROBS_WINDOW:]
    session['recent_probs'] = recent
    return sum(recent) / len(recent)


def _reset_liveness_session(keep_log_dedup=False):
    """Clears all per-session liveness/challenge state -- called on no-face,
    multi-face, an abrupt face-position jump, or a logged-out SPOOF/UNCERTAIN
    decision, so a new physical subject always starts from a clean slate.

    keep_log_dedup: NO_FACE / MULTI_FACE keep the scan-log de-dup key, so an
    empty camera logs NO_FACE once instead of on every frame."""
    session.pop('recent_probs', None)
    session.pop('last_bbox', None)
    session.pop('last_logged_decision', None)
    if not keep_log_dedup:
        session.pop('last_logged_stage', None)
    session.pop('mask_streak', None)
    _clear_blink_attempt()


def _clear_blink_attempt():
    """Ends any in-progress challenge (head turn + blink) for this browser session."""
    session.pop('challenge_type', None)
    session.pop('challenge_user', None)
    session.pop('challenge_assigned_at', None)
    session.pop('challenge_turn', None)
    session.pop('turn_stage', None)
    session.pop('blink_stage', None)


def _pick_turn():
    """Random head-turn direction for a new challenge (unpredictable to the client)."""
    return secrets.choice(("left", "right"))


def _new_challenge(user_id):
    """Starts a fresh random challenge for this recognized user."""
    _clear_blink_attempt()
    session['challenge_type'] = 'turn_then_blink'
    session['challenge_turn'] = _pick_turn()
    session['challenge_user'] = user_id
    session['challenge_assigned_at'] = time.time()
    session['turn_stage'] = 'start'


def _challenge_instruction():
    if session.get('turn_stage') == 'turned':
        return BLINK_INSTRUCTION
    return TURN_INSTRUCTIONS.get(session.get('challenge_turn'), "")


def _advance_challenge(liveness_metrics, allow_complete=True):
    """Server-side random challenge, from this frame's Face Mesh metrics:

        turn:  start --straight--> straight --turned the asked way--> turned
        blink: (only after the turn) open -> closed -> open   (_advance_blink)

    Returns 'turn', 'blink', 'done', or 'wrong' (turned the other way: the
    caller starts a new random challenge). With allow_complete=False (an
    unrecognized frame of the same tracked face) progress is allowed but the
    challenge can't finish."""
    if not liveness_metrics or not liveness_metrics.get("face_mesh_detected"):
        return 'done' if session.get('blink_stage') == 'done' else (
            'blink' if session.get('turn_stage') == 'turned' else 'turn')
    stage = session.get('turn_stage', 'start')
    if stage != 'turned':
        yaw = float(liveness_metrics.get("head_yaw") or 0.0)
        wanted = 1 if session.get('challenge_turn') == 'left' else -1
        if stage == 'start' and abs(yaw) < HEAD_STRAIGHT_MAX_YAW:
            stage = 'straight'
        elif stage == 'straight':
            if yaw * wanted > HEAD_TURN_MIN_YAW:
                stage = 'turned'
            elif yaw * wanted < -HEAD_TURN_MIN_YAW:
                return 'wrong'
        session['turn_stage'] = stage
        return 'turn' if stage != 'turned' else 'blink'
    return 'done' if _advance_blink(liveness_metrics, allow_complete) else 'blink'


def _advance_blink(liveness_metrics, allow_complete=True):
    """Server-side blink check, per session, from the eye-aspect-ratio the
    server itself computed with Face Mesh on this LIVE frame
    (liveness_metrics["eyes_closed"] = EAR below utils/face_utils.py's
    existing cut-off). Requires open -> closed -> open:

        start --open--> open --closed--> closed --open--> done

    Frames where Face Mesh found no face don't count as "open". With
    allow_complete=False (an unrecognized LIVE frame of the same tracked
    face -- closed eyes can make recognition fail for a frame) the attempt
    may move open -> closed but can't finish; finishing needs a recognized
    open-eye frame of the same user. Returns True once done."""
    if not liveness_metrics or not liveness_metrics.get("face_mesh_detected"):
        return session.get('blink_stage') == 'done'
    closed = bool(liveness_metrics.get("eyes_closed"))
    stage = session.get('blink_stage', 'start')
    if stage == 'start' and not closed and allow_complete:
        stage = 'open'
    elif stage == 'open' and closed:
        stage = 'closed'
    elif stage == 'closed' and not closed and allow_complete:
        stage = 'done'
    session['blink_stage'] = stage
    return stage == 'done'


def _face_box_iou(a, b):
    """Intersection-over-union of two [x, y, width, height] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _is_different_face(prev_bbox, cur_bbox):
    """Same-face consistency check: True if the face box's IoU with the
    previous frame's box is below SAME_FACE_MIN_IOU, i.e. the face that
    passed liveness is no longer the same physical subject (swapped, or a
    phone/photo pushed in). Catches position jumps AND sudden size changes."""
    if prev_bbox is None:
        return False
    return _face_box_iou(prev_bbox, cur_bbox) < SAME_FACE_MIN_IOU


def _log_liveness_event_if_changed(decision, p_spoof, user_id=None):
    """Writes one liveness_events row ONLY when the decision differs from the
    last one logged for this session -- one row per decision CHANGE, not one
    row per frame (frames arrive every ~150-300ms; logging every one would
    flood the table)."""
    sid = _get_liveness_session_id()
    if session.get('last_logged_decision') == decision:
        return
    session['last_logged_decision'] = decision
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO liveness_events (session_id, decision, p_spoof, user_id) VALUES (?, ?, ?, ?)",
        (sid, decision, round(float(p_spoof), 4), user_id),
    )
    conn.commit()
    conn.close()


def _log_scan_event(stage, user_id=None, detail=None, dedup=True, update_dedup_key=True, test_label=None,
                    p_spoof_frame=None, p_spoof_avg=None, iou=None):
    """Writes one scan_events row for the pipeline STAGE reached this request
    -- for reviewing a test session afterward (what happened, in order).

    For the continuous per-frame states (NO_FACE/MULTI_FACE/SPOOF/UNCERTAIN/
    RECOGNIZED/UNRECOGNIZED) dedup=True so a stage that repeats across many
    frames logs once per CHANGE, not once per frame. One-shot transitions
    (FACE_MOVED/CHALLENGE_PASSED/ATTENDANCE_MARKED/ALREADY_MARKED) always
    log -- they're already rare by construction. A bare "LIVE" classification
    is deliberately not logged as its own row here: in this pipeline LIVE
    always leads straight into RECOGNIZED or UNRECOGNIZED (or MULTI_FACE) the
    same request, so the deeper stage already implies it -- logging both
    would double up every frame.

    FACE_MOVED passes update_dedup_key=False so it doesn't disturb the dedup
    baseline for the deeper stage logged right after it in the same request.

    p_spoof_frame / p_spoof_avg: this frame's raw liveness score and the
    5-frame smoothed score the decision used. Set on every row logged after
    the liveness model ran; NULL for stages before it (NO_FACE, BlazeFace
    MULTI_FACE, FACE_MOVED) -- no score exists yet for those.

    iou: face-box IoU vs. the previous frame's box (same-face check), for
    every row logged after a face box exists and a previous box was known.
    """
    sid = _get_liveness_session_id()
    if dedup and session.get('last_logged_stage') == stage:
        return
    if update_dedup_key:
        session['last_logged_stage'] = stage
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO scan_events (session_id, stage, user_id, detail, test_label, p_spoof_frame, p_spoof_avg, iou) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, stage, user_id, detail, test_label,
         None if p_spoof_frame is None else float(p_spoof_frame),
         None if p_spoof_avg is None else float(p_spoof_avg),
         None if iou is None else float(iou)),
    )
    conn.commit()
    conn.close()


# ============================================================
#  PUBLIC ROUTES
# ============================================================

def _hash_scanner_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _current_scanner_device():
    """The approved, non-revoked scanner_devices row for this browser's
    scanner cookie, or None."""
    token = request.cookies.get(SCANNER_COOKIE_NAME)
    if not token:
        return None
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, label FROM scanner_devices WHERE token_hash = ? AND revoked_at IS NULL",
        (_hash_scanner_token(token),),
    ).fetchone()
    conn.close()
    return row


def _scanner_approval_required():
    return get_setting('require_scanner_approval', '0') == '1'


def _scanner_allowed():
    """Any device may scan unless the admin turned on "Require approved
    scanner device"; then only approved devices (cookie) may."""
    return not _scanner_approval_required() or bool(_current_scanner_device())


@app.route("/")
def home():
    from contextlib import closing
    approved = _scanner_allowed()
    with closing(get_db_connection()) as conn:
        enrolled = bool(conn.execute("SELECT 1 FROM users WHERE face_registered=1 AND face_embedding IS NOT NULL LIMIT 1").fetchone())
    return render_template("scanner.html", approved=approved, enrolled=enrolled,
                           approval_message=MSG_DEVICE_NOT_APPROVED, debug_mode=DEBUG_MODE)


def _do_login(forced_role=None):
    """Shared login handler for the single login page. The Student/Admin
    toggle only says which kind of account the user expects; the account's
    real role in the DB is what's checked. Wrong password, unknown email, or
    a role that doesn't match the toggle all give the same message."""
    role = forced_role or (request.form.get("role") or "student")
    if role not in ("student", "admin"):
        role = "student"
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        flash("Please enter both email and password.", "danger")
        return render_template("login.html", role=role)

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    expected_db_role = "admin" if role == "admin" else "employee"
    if not user or not check_password_hash(user["password_hash"], password) or user["role"] != expected_db_role:
        flash("Invalid credentials", "danger")
        return render_template("login.html", role=role)

    session["user_id"] = user["id"]
    session["role"] = user["role"]
    session["name"] = user["name"]
    session["employee_id"] = user["employee_id"]
    session["profile_photo"] = user["profile_photo"] or ""
    if session.pop("open_enrollment", False) and user["role"] == "employee":
        return redirect(url_for("employee_dashboard", _anchor="register-face"))
    return redirect(url_for("admin_dashboard" if user["role"] == "admin" else "employee_dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.args.get("enroll") == "1":
        session["open_enrollment"] = True
    if "user_id" in session:
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        if session.get("role") == "employee":
            if session.pop("open_enrollment", False):
                return redirect(url_for("employee_dashboard", _anchor="register-face"))
            return redirect(url_for("employee_dashboard"))
    if request.method == "POST":
        return _do_login()
    role = request.args.get("role", "student")
    return render_template("login.html", role=role if role in ("student", "admin") else "student")


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    """Kept so existing links/redirects work: shows the single login page
    with the Admin toggle preselected."""
    if request.method == "POST":
        if "user_id" in session and session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return _do_login(forced_role="admin")
    if "user_id" in session and session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("login", role="admin"))


@app.route("/how_it_works")
def how_it_works():
    headline = None
    try:
        with open(os.path.join(Config.BASE_DIR, "training", "results", "summary.json")) as fh:
            headline = json.load(fh).get("headline")
    except Exception:
        headline = None
    return render_template("how_it_works.html", headline=headline)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.args.get("enroll") == "1":
        session["open_enrollment"] = True
    if request.method == "POST":
        cleaned, errors = validate_registration(request.form)

        if errors:
            # Return with first error
            first_error = list(errors.values())[0]
            flash(first_error, "danger")
            return render_template("register.html", departments=ALLOWED_DEPARTMENTS)

        conn = get_db_connection()
        try:
            hashed_pw = generate_password_hash(cleaned['password'])
            conn.execute('''
                INSERT INTO users (employee_id, name, email, phone, department, password_hash, role)
                VALUES (?, ?, ?, ?, ?, ?, 'employee')
            ''', (cleaned['employee_id'], cleaned['name'], cleaned['email'],
                  cleaned['phone'], cleaned['department'], hashed_pw))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except _IntegrityErrors:
            # Catches both sqlite3.IntegrityError and psycopg2.IntegrityError
            # (previously only caught sqlite3's -- a duplicate employee_id/email
            # under Postgres fell through to the generic except below instead
            # of this specific message). NOTE: not yet tested against a live
            # Postgres database -- confirm this path for real before relying
            # on it in production.
            conn.rollback()
            flash("Student ID or Email already exists.", "danger")
        except Exception as e:
            conn.rollback()
            flash("Registration failed. Please try again.", "danger")
        finally:
            conn.close()

    return render_template("register.html", departments=ALLOWED_DEPARTMENTS)


@app.route("/logout")
def logout():
    session.clear()
    if request.args.get("enroll") == "1":
        return redirect(url_for("login", enroll="1"))
    return redirect(url_for("home"))


# ============================================================
#  PROFILE MANAGEMENT
# ============================================================

@app.route("/profile", methods=["GET", "POST"])
def profile():
    """Employee profile management page."""
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_info":
            cleaned, errors = validate_profile_update(request.form)

            if errors:
                first_error = list(errors.values())[0]
                flash(first_error, "danger")
                conn.close()
                return redirect(url_for("profile"))

            conn.execute('''
                UPDATE users SET name = ?, phone = ?, department = ? WHERE id = ?
            ''', (cleaned['name'], cleaned['phone'], cleaned['department'], session["user_id"]))
            conn.commit()
            session["name"] = cleaned['name']
            flash("Profile updated successfully.", "success")

        elif action == "change_password":
            current_pw = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            confirm_pw = request.form.get("confirm_password", "")

            if not check_password_hash(user["password_hash"], current_pw):
                flash("Current password is incorrect.", "danger")
                conn.close()
                return redirect(url_for("profile"))

            if new_pw != confirm_pw:
                flash("New passwords do not match.", "danger")
                conn.close()
                return redirect(url_for("profile"))

            new_pw_clean, pw_err = validate_password(new_pw)
            if pw_err:
                flash(pw_err, "danger")
                conn.close()
                return redirect(url_for("profile"))

            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                         (generate_password_hash(new_pw_clean), session["user_id"]))
            conn.commit()
            flash("Password changed successfully.", "success")

        elif action == "upload_photo":
            if 'photo' not in request.files:
                flash("No file selected.", "danger")
                conn.close()
                return redirect(url_for("profile"))

            file = request.files['photo']
            is_valid, photo_err = validate_profile_photo(file)
            if not is_valid:
                flash(photo_err, "danger")
                conn.close()
                return redirect(url_for("profile"))

            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"profile_{session['user_id']}_{uuid.uuid4().hex[:8]}.{ext}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Delete old photo if exists
            if user["profile_photo"]:
                old_path = os.path.join(Config.UPLOAD_FOLDER, user["profile_photo"])
                if os.path.exists(old_path):
                    os.remove(old_path)

            conn.execute("UPDATE users SET profile_photo = ? WHERE id = ?", (filename, session["user_id"]))
            conn.commit()
            session["profile_photo"] = filename
            flash("Profile photo updated.", "success")

        conn.close()
        return redirect(url_for("profile"))

    conn.close()
    return render_template("profile.html", user=user, departments=ALLOWED_DEPARTMENTS)


# ============================================================
#  ADMIN ROUTES
# ============================================================

@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    total_emp = conn.execute("SELECT COUNT(*) FROM users WHERE role='employee'").fetchone()[0]

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    present_today = conn.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ? AND status IN ('Present', 'Late')", (today,)).fetchone()[0]
    present_yesterday = conn.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ? AND status IN ('Present', 'Late')", (yesterday,)).fetchone()[0]

    # Read the stored `status` column (set once, at check-in time, using
    # whatever the late cutoff was *then*) rather than recomputing from
    # is_late() against the *current* cutoff setting -- the two disagreed on
    # historical days whenever an admin changed the cutoff, since analytics()
    # already used the stored-status approach while this route recomputed
    # live. Both now use the same method everywhere.
    late_count = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Late'", (today,)
    ).fetchone()[0]
    late_yesterday = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Late'", (yesterday,)
    ).fetchone()[0]

    attendance_rate = round((present_today / total_emp * 100), 1) if total_emp > 0 else 0

    present_trend = 0
    if present_yesterday > 0:
        present_trend = round(((present_today - present_yesterday) / present_yesterday) * 100, 1)
    elif present_today > 0:
        present_trend = 100

    late_trend = 0
    if late_yesterday > 0:
        late_trend = round(((late_count - late_yesterday) / late_yesterday) * 100, 1)
    elif late_count > 0:
        late_trend = 100

    recent_attendance = conn.execute('''
        SELECT attendance.id, attendance.time, attendance.status, attendance.date,
               users.name, users.employee_id, users.department, users.profile_photo
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE attendance.date = ?
        ORDER BY attendance.time DESC LIMIT 20
    ''', (today,)).fetchall()

    weekly_data, weekly_late, weekly_labels = [], [], []
    for i in range(6, -1, -1):
        day = datetime.now() - timedelta(days=i)
        d = day.strftime("%Y-%m-%d")
        count = conn.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ? AND status IN ('Present', 'Late')", (d,)).fetchone()[0]
        late_n = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ? AND status = 'Late'", (d,)
        ).fetchone()[0]
        weekly_data.append(count)
        weekly_late.append(late_n)
        weekly_labels.append(day.strftime("%a %d"))

    admin_user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()

    employees_list = conn.execute("SELECT id, employee_id, name, department FROM users WHERE role = 'employee'").fetchall()

    conn.close()

    absent_today = total_emp - present_today

    return render_template("admin_dashboard.html",
                           total_emp=total_emp,
                           present_today=present_today,
                           absent_today=absent_today,
                           late_count=late_count,
                           attendance_rate=attendance_rate,
                           present_trend=present_trend,
                           late_trend=late_trend,
                           recent_attendance=recent_attendance,
                           weekly_data=weekly_data,
                           weekly_late=weekly_late,
                           weekly_labels=weekly_labels,
                           admin=admin_user,
                           employees_list=employees_list,
                           debug_mode=DEBUG_MODE)


@app.route("/admin/dev/clear_today_attendance", methods=["POST"])
def admin_dev_clear_today_attendance():
    """Dev-only: deletes today's attendance rows so a tester can repeat the
    scan flow without waiting for a new day (attendance is marked once per
    day per user). Hard-disabled unless FLASK_DEBUG=1 (same flag that gates
    Flask's own debug mode) -- never available outside local development,
    regardless of who is logged in."""
    if not DEBUG_MODE:
        return jsonify({"success": False, "msg": "Dev tools are disabled outside local development."}), 403
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()
    deleted = conn.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,)).fetchone()[0]
    conn.execute("DELETE FROM attendance WHERE date = ?", (today,))
    conn.commit()
    conn.close()
    flash(f"[DEV] Cleared {deleted} attendance record(s) for {today}.", "success")
    return redirect(url_for("admin_dashboard"))


# ============================================================
#  PIPELINE VIEW + GRAD-CAM (admin only; never marks attendance,
#  never writes scan logs, never saves images to disk)
# ============================================================

# Grad-CAM target: the last convolutional block of MobileNetV3-Small,
# feature_extractor.features[12] (Conv2dNormActivation 96->576, 7x7 output
# for a 224x224 input) -- the same layer training/src/gradcam.py uses.
GRADCAM_LAYER_NAME = "feature_extractor.features[12]"
_gradcam_lock = threading.Lock()
_gradcam_model = None   # private deep copy so Grad-CAM hooks never touch the live scanner's model


def _get_gradcam_model():
    global _gradcam_model
    if _gradcam_model is None:
        _gradcam_model = copy.deepcopy(_liveness_predictor.model)
        _gradcam_model.eval()
    return _gradcam_model


def _bgr_to_data_uri(img_bgr, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return None
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.route("/admin/pipeline_view")
def admin_pipeline_view():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    return render_template("pipeline_view.html", gradcam_layer=GRADCAM_LAYER_NAME)


@app.route("/api/admin/pipeline_frame", methods=["POST"])
def api_admin_pipeline_frame():
    """One frame through detection -> crop -> 224x224 model input -> liveness
    score/decision, optionally with Grad-CAM. Pure inspection: no session
    smoothing, no recognition, no attendance, no logging, nothing on disk."""
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    frame = data_uri_to_cv2_img(data.get("image") or "") if data.get("image") else None
    if frame is None:
        return jsonify({"success": False, "msg": "Could not decode image."}), 400

    frame_h, frame_w = frame.shape[:2]
    detections = _liveness_detector.detect(frame)
    boxes = [{"x": d.x, "y": d.y, "width": d.width, "height": d.height,
              "confidence": round(float(d.confidence), 4)} for d in detections]
    out = {"success": True, "frame_width": frame_w, "frame_height": frame_h,
           "boxes": boxes, "face_count": len(detections)}
    if not detections:
        out.update(state="NO_FACE", msg="No face detected.")
        return jsonify(out)

    best = max(detections, key=lambda d: d.width * d.height)
    crop = crop_face(frame, best)
    if crop is None or crop.size == 0:
        out.update(state="NO_FACE", msg="Face too close to the frame edge.")
        return jsonify(out)

    # 224x224 model input = the predictor's own Resize step (before ToTensor /
    # ImageNet normalization), so this is exactly what the network sees.
    from liveness.face_utils import bgr_crop_to_pil
    pil = bgr_crop_to_pil(crop)
    model_input_rgb = np.array(_liveness_predictor.transform.transforms[0](pil))
    model_input_bgr = cv2.cvtColor(model_input_rgb, cv2.COLOR_RGB2BGR)

    p_spoof = _liveness_predictor.predict_prob(crop)
    liveness_threshold = float(get_setting('liveness_threshold', '0.5'))
    decision = classify_three_way(p_spoof, *_gray_zone_band, threshold=liveness_threshold)
    out.update(
        state=decision, p_spoof=p_spoof, multi_face=len(detections) > 1,
        band_low=_gray_zone_band[0], band_high=_gray_zone_band[1], threshold=liveness_threshold,
        crop=_bgr_to_data_uri(crop), crop_size=[int(crop.shape[1]), int(crop.shape[0])],
        model_input=_bgr_to_data_uri(model_input_bgr, ".png"),
    )

    if data.get("explain"):
        from utils.gradcam import GradCAM, overlay_heatmap
        tensor = _liveness_predictor.transform(pil).unsqueeze(0).to(_liveness_predictor.device)
        with _gradcam_lock:
            model = _get_gradcam_model()
            with GradCAM(model, model.feature_extractor.features[12]) as cam:
                heatmap, cam_prob = cam(tensor)
        overlay = overlay_heatmap(model_input_bgr, heatmap)
        out.update(
            gradcam=_bgr_to_data_uri(overlay, ".png"),
            gradcam_layer=GRADCAM_LAYER_NAME,
            gradcam_explains="spoof evidence" if cam_prob >= 0.5 else "live evidence",
            gradcam_p_spoof=cam_prob,
        )
    return jsonify(out)


# ============================================================
#  SPOOF LOG / RESULTS / TEST MODE (admin only)
# ============================================================

@app.route("/admin/spoof_log")
def admin_spoof_log():
    """SPOOF/UNCERTAIN rows from scan_events, filterable, with CSV export.
    Times are the DB's CURRENT_TIMESTAMP (UTC)."""
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    stage = request.args.get("stage", "all")
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()
    label = (request.args.get("label") or "").strip()

    where, params = ["stage IN ('SPOOF', 'UNCERTAIN')"], []
    if stage in ("SPOOF", "UNCERTAIN"):
        where.append("stage = ?"); params.append(stage)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_from):
        where.append("date(created_at) >= ?"); params.append(date_from)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_to):
        where.append("date(created_at) <= ?"); params.append(date_to)
    if label:
        where.append("test_label = ?"); params.append(label[:40])
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, created_at, stage, p_spoof_frame, p_spoof_avg, iou, test_label, session_id, detail "
        f"FROM scan_events WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT 2000", params
    ).fetchall()
    labels = [r["test_label"] for r in conn.execute(
        "SELECT DISTINCT test_label FROM scan_events WHERE test_label IS NOT NULL AND test_label != '' ORDER BY test_label"
    ).fetchall()]
    conn.close()

    if request.args.get("export") == "csv":
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["id", "time_utc", "stage", "p_spoof_frame", "p_spoof_avg", "iou", "test_label", "session_id", "detail"])
        for r in rows:
            w.writerow([r["id"], r["created_at"], r["stage"], r["p_spoof_frame"], r["p_spoof_avg"], r["iou"],
                        r["test_label"] or "", r["session_id"], r["detail"] or ""])
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=spoof_log.csv"})
    return render_template("spoof_log.html", rows=rows, labels=labels,
                           f={"stage": stage, "from": date_from, "to": date_to, "label": label})


# Result files read at request time from training/ (never copied as numbers into code).
RESULTS_SUMMARY_PATH = os.path.join(Config.BASE_DIR, "training", "results", "summary.json")
LATENCY_FILE_PATH = os.path.join(Config.BASE_DIR, "docs", "pipeline_latency_2026-11-07.txt")
TRAINING_HISTORY_PATH = os.path.join(Config.BASE_DIR, "training", "outputs", "plots", "v2_cropped", "training_history.json")
RESULT_PLOTS = {
    "training_curves": "training/outputs/plots/v2_cropped/training_curves.png",
    "confusion_test": "training/outputs/plots/v2_cropped/confusion_matrix_v2_cropped_test.png",
    "confusion_val": "training/outputs/plots/v2_cropped/confusion_matrix_v2_cropped_val.png",
    "gray_zone_curve": "training/results/plots/v2/gray_zone_coverage_vs_error.png",
    "breakdown_spoof_type": "training/results/plots/v2/breakdown_spoof_type.png",
}


@app.route("/admin/results")
def admin_results():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    summary, error = None, None
    try:
        with open(RESULTS_SUMMARY_PATH) as fh:
            summary = json.load(fh)
    except Exception as e:
        error = f"Could not read training/results/summary.json: {e}"
    latency_text = None
    try:
        with open(LATENCY_FILE_PATH) as fh:
            latency_text = fh.read()
    except Exception:
        latency_text = None
    plots = {k: os.path.isfile(os.path.join(Config.BASE_DIR, v)) for k, v in RESULT_PLOTS.items()}
    history = None
    try:
        with open(TRAINING_HISTORY_PATH) as fh:
            history = json.load(fh)
    except Exception:
        history = None
    latency_median_ms = None
    m = re.search(r"Full round-trip[^\n]*?median=([0-9.]+)ms", latency_text or "")
    if m:
        latency_median_ms = float(m.group(1))
    live = _test_mode_metrics(None)
    return render_template("results.html", s=summary, error=error, plots=plots, history=history,
                           latency_median_ms=latency_median_ms,
                           latency_text=latency_text, latency_file=os.path.relpath(LATENCY_FILE_PATH, Config.BASE_DIR),
                           live=live)


@app.route("/admin/results/plot/<name>")
def admin_results_plot(name):
    if session.get("role") != "admin":
        abort(401)
    rel = RESULT_PLOTS.get(name)
    if not rel or not os.path.isfile(os.path.join(Config.BASE_DIR, rel)):
        abort(404)
    return send_file(os.path.join(Config.BASE_DIR, rel), mimetype="image/png")


# Test Mode: what the admin is showing the camera. Bona fide = real person.
TEST_CATEGORIES = {
    "real_face": ("Real face", True),
    "real_face_masked": ("Masked face", True),
    "printed_photo": ("Printed photo", False),
    "phone_photo": ("Phone photo", False),
    "phone_video": ("Phone video", False),
    "laptop_screen": ("Laptop screen", False),
}


def _test_mode_metrics(test_session):
    """Real counts from test_mode_results (one row per camera frame with a
    single detected face), for one test session or all (None).
      APCER (attack category) = frames decided LIVE / attack frames
      BPCER (bona fide)       = frames decided SPOOF / bona fide frames
      UNCERTAIN frames are counted and shown separately (neither accepted
      nor rejected as spoof -- the scanner just asks to retry).
      HTER = (APCER + BPCER) / 2; per attack category it uses that
      category's APCER with the overall bona fide BPCER."""
    conn = get_db_connection()
    q = ("SELECT category, decision, COUNT(*) AS n FROM test_mode_results "
         "WHERE decision IN ('LIVE','SPOOF','UNCERTAIN')")
    params = []
    if test_session:
        q += " AND test_session = ?"; params.append(test_session)
    q += " GROUP BY category, decision"
    counts = {}
    for r in conn.execute(q, params).fetchall():
        counts.setdefault(r["category"], {"LIVE": 0, "SPOOF": 0, "UNCERTAIN": 0})[r["decision"]] = r["n"]
    conn.close()

    def rate(num, den):
        return (num / den) if den else None

    bona = {"LIVE": 0, "SPOOF": 0, "UNCERTAIN": 0}
    attack = {"LIVE": 0, "SPOOF": 0, "UNCERTAIN": 0}
    for cat, c in counts.items():
        tgt = bona if TEST_CATEGORIES.get(cat, ("", False))[1] else attack
        for k in tgt:
            tgt[k] += c[k]
    bpcer_all = rate(bona["SPOOF"], sum(bona.values()))
    apcer_all = rate(attack["LIVE"], sum(attack.values()))

    rows = []
    for key, (label, is_bona) in TEST_CATEGORIES.items():
        c = counts.get(key, {"LIVE": 0, "SPOOF": 0, "UNCERTAIN": 0})
        n = sum(c.values())
        row = {"key": key, "label": label, "bona_fide": is_bona, "n": n, **c,
               "uncertain_rate": rate(c["UNCERTAIN"], n), "correct": None, "wrong": None,
               "apcer": None, "bpcer": None, "hter": None}
        if is_bona:
            row.update(correct=c["LIVE"], wrong=c["SPOOF"], bpcer=rate(c["SPOOF"], n))
        else:
            row.update(correct=c["SPOOF"], wrong=c["LIVE"], apcer=rate(c["LIVE"], n))
            if row["apcer"] is not None and bpcer_all is not None:
                row["hter"] = (row["apcer"] + bpcer_all) / 2
        rows.append(row)
    overall = {"n_bona": sum(bona.values()), "n_attack": sum(attack.values()),
               "apcer": apcer_all, "bpcer": bpcer_all,
               "hter": (apcer_all + bpcer_all) / 2 if apcer_all is not None and bpcer_all is not None else None,
               "uncertain_bona": bona["UNCERTAIN"], "uncertain_attack": attack["UNCERTAIN"]}
    return {"rows": rows, "overall": overall, "total": overall["n_bona"] + overall["n_attack"]}


# Domain-adaptation collection mode (Phase 3). Saves the raw 25%-margin face
# crop (what crop_face() returns, before resizing) as PNG, outside git.
COLLECTION_DIR = os.path.join(Config.BASE_DIR, "data", "domain_adaptation")
COLLECTION_MIN_INTERVAL_S = 1.0     # at most one saved crop per second (consecutive frames are near-duplicates)
COLLECTION_GLASSES = ("glasses", "no_glasses")
COLLECTION_LIGHTING = ("normal", "dim")
COLLECTION_SESSIONS = (1, 2)        # 1 = adaptation data, 2 = held-out test data only


def _parse_collection(data):
    """Validated collection settings from a Test Mode request, or (None, error)."""
    c = data.get("collect")
    if not c:
        return None, None
    person = str(c.get("person_id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", person):
        return None, "Person ID: 1-20 letters, digits, _ or - (use a code like P01, not a name)."
    if c.get("glasses") not in COLLECTION_GLASSES:
        return None, "Choose glasses / no glasses."
    if c.get("lighting") not in COLLECTION_LIGHTING:
        return None, "Choose normal / dim light."
    try:
        sess_no = int(c.get("session"))
    except (TypeError, ValueError):
        sess_no = None
    if sess_no not in COLLECTION_SESSIONS:
        return None, "Choose collection session 1 or 2."
    return {"person_id": person, "glasses": c["glasses"], "lighting": c["lighting"], "session": sess_no}, None


def _collection_counts(person_id, sess_no):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT category, glasses, lighting, COUNT(*) AS n FROM test_mode_results "
        "WHERE crop_path IS NOT NULL AND person_id = ? AND collection_session = ? "
        "GROUP BY category, glasses, lighting", (person_id, sess_no)
    ).fetchall()
    conn.close()
    return [{"category": r["category"], "glasses": r["glasses"], "lighting": r["lighting"], "n": r["n"]} for r in rows]


def _new_test_session():
    session["test_session"] = uuid.uuid4().hex[:12]
    session.pop("test_recent_probs", None)
    session.pop("test_last_category", None)
    return session["test_session"]


@app.route("/admin/test_mode")
def admin_test_mode():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    current = session.get("test_session") or _new_test_session()
    conn = get_db_connection()
    sessions = conn.execute(
        "SELECT test_session, MIN(created_at) AS started, COUNT(*) AS n FROM test_mode_results "
        "GROUP BY test_session ORDER BY started DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return render_template("test_mode.html", categories=TEST_CATEGORIES, current_session=current,
                           sessions=sessions, metrics=_test_mode_metrics(current))


@app.route("/api/admin/test_mode/new_session", methods=["POST"])
def api_admin_test_new_session():
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401
    sid = _new_test_session()
    return jsonify({"success": True, "test_session": sid, "metrics": _test_mode_metrics(sid)})


@app.route("/api/admin/test_mode/metrics")
def api_admin_test_metrics():
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401
    sid = request.args.get("session") or None
    if sid == "all":
        sid = None
    return jsonify({"success": True, "test_session": sid, "metrics": _test_mode_metrics(sid)})


@app.route("/api/admin/test_scan", methods=["POST"])
def api_admin_test_scan():
    """Test Mode frame: BlazeFace -> crop -> liveness model -> 5-frame
    average -> three-way decision (same functions and thresholds as the
    scanner). Records the result for Test Mode metrics. Never runs
    recognition, never touches attendance or the scanner's session state."""
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    category = data.get("category")
    if category not in TEST_CATEGORIES:
        return jsonify({"success": False, "msg": "Choose what you are showing the camera."}), 400
    frame = data_uri_to_cv2_img(data.get("image")) if data.get("image") else None
    if frame is None:
        return jsonify({"success": False, "msg": "Could not decode image."}), 400
    collect, collect_err = _parse_collection(data)
    if collect_err:
        return jsonify({"success": False, "msg": collect_err}), 400
    test_session = session.get("test_session") or _new_test_session()

    if session.get("test_last_category") != category:
        session["test_recent_probs"] = []
        session["test_last_category"] = category

    detections = _liveness_detector.detect(frame)
    face_box, p_frame, p_avg, crop = None, None, None, None
    if len(detections) == 0:
        decision = "NO_FACE"
    elif len(detections) > 1:
        decision = "MULTI_FACE"
    else:
        d = detections[0]
        h, w = frame.shape[:2]
        face_box = {"x": d.x, "y": d.y, "width": d.width, "height": d.height, "frame_width": w, "frame_height": h}
        crop = crop_face(frame, d)
        if crop is None or crop.size == 0:
            decision = "NO_FACE"
        else:
            p_frame = _liveness_predictor.predict_prob(crop)
            probs = (session.get("test_recent_probs") or [])[-(RECENT_PROBS_WINDOW - 1):] + [p_frame]
            session["test_recent_probs"] = probs
            p_avg = float(sum(probs) / len(probs))
            decision = classify_three_way(p_avg, *_gray_zone_band,
                                          threshold=float(get_setting('liveness_threshold', '0.5')))
    if decision in ("NO_FACE", "MULTI_FACE"):
        session["test_recent_probs"] = []

    crop_rel = None
    if collect and crop is not None and p_frame is not None \
            and time.time() - session.get("collect_last_saved", 0) >= COLLECTION_MIN_INTERVAL_S:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        rel = os.path.join(f"session_{collect['session']}", category,
                           f"{collect['person_id']}__{collect['glasses']}__{collect['lighting']}__{stamp}.png")
        path = os.path.join(COLLECTION_DIR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if cv2.imwrite(path, crop):
            crop_rel = rel
            session["collect_last_saved"] = time.time()

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO test_mode_results (test_session, category, decision, p_spoof_frame, p_spoof_avg, admin_user_id, "
        "person_id, glasses, lighting, collection_session, crop_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (test_session, category, decision, p_frame, p_avg, session["user_id"],
         collect["person_id"] if collect else None, collect["glasses"] if collect else None,
         collect["lighting"] if collect else None, collect["session"] if collect else None, crop_rel),
    )
    conn.commit()
    conn.close()

    if crop_rel:
        manifest = os.path.join(COLLECTION_DIR, "manifest.csv")
        new_file = not os.path.isfile(manifest)
        with open(manifest, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["crop_path", "label", "bona_fide", "person_id", "glasses", "lighting", "session",
                            "decision", "p_spoof_frame", "p_spoof_avg", "saved_utc"])
            w.writerow([crop_rel, category, int(TEST_CATEGORIES[category][1]), collect["person_id"],
                        collect["glasses"], collect["lighting"], collect["session"], decision,
                        p_frame, p_avg, datetime.utcnow().isoformat(timespec="seconds")])

    out = {"success": True, "decision": decision, "p_spoof_frame": p_frame, "p_spoof_avg": p_avg,
           "face_box": face_box, "test_session": test_session, "metrics": _test_mode_metrics(test_session)}
    if collect:
        out["saved"] = bool(crop_rel)
        out["collection_counts"] = _collection_counts(collect["person_id"], collect["session"])
    return jsonify(out)


@app.route("/admin/scan_log")
def admin_scan_log():
    """Admin-only review of the most recent scan-pipeline events, in order,
    with timestamps -- for reviewing a live test session afterward. Each row
    is the DEEPEST stage a given request reached (see _log_scan_event());
    FACE_MOVED is an extra marker row, not a terminal stage. Shows the most
    recent 300 rather than filtering by "today" -- created_at is SQLite's
    CURRENT_TIMESTAMP (UTC), and comparing that to a local-time date string
    would silently drop or duplicate rows near midnight."""
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    rows = conn.execute('''
        SELECT scan_events.id, scan_events.session_id, scan_events.stage,
               scan_events.detail, scan_events.test_label, scan_events.created_at,
               scan_events.p_spoof_frame, scan_events.p_spoof_avg, scan_events.iou,
               users.name AS user_name, users.employee_id
        FROM scan_events
        LEFT JOIN users ON scan_events.user_id = users.id
        ORDER BY scan_events.id DESC
        LIMIT 300
    ''').fetchall()
    conn.close()
    return render_template("scan_log.html", rows=rows)


@app.route("/manage_employees", methods=["GET", "POST"])
def manage_employees():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    conn = get_db_connection()

    if request.method == "POST":
        action = request.form.get("action")
        user_id = request.form.get("user_id")
        if action == "delete" and user_id:
            conn.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            flash("Student removed successfully.", "success")
            return redirect(url_for("manage_employees"))

    employees = conn.execute(
        "SELECT id, employee_id, name, email, phone, department, face_registered, profile_photo, created_at FROM users WHERE role='employee' ORDER BY name"
    ).fetchall()

    dept_stats = conn.execute(
        "SELECT department, COUNT(*) as count FROM users WHERE role='employee' GROUP BY department ORDER BY count DESC"
    ).fetchall()

    conn.close()

    return render_template("employees.html", employees=employees, dept_stats=dept_stats,
                           departments=ALLOWED_DEPARTMENTS)


@app.route("/admin/students/<int:user_id>/reset_face", methods=["POST"])
def admin_reset_face(user_id):
    """Clears one student's stored face samples so they can register again."""
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    conn = get_db_connection()
    student = conn.execute("SELECT name, employee_id FROM users WHERE id = ? AND role = 'employee'", (user_id,)).fetchone()
    if student:
        conn.execute("UPDATE users SET face_embedding = NULL, face_registered = 0 WHERE id = ?", (user_id,))
        conn.commit()
    conn.close()
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("manage_employees"))
    from utils.face_utils import invalidate_cache
    invalidate_cache()
    _log_admin_action("RESET_FACE", f"user_id={user_id} employee_id={student['employee_id']}")
    flash(f"Face samples cleared for {student['name']} ({student['employee_id']}). They can register again.", "success")
    return redirect(url_for("manage_employees"))


@app.route("/attendance_history")
def attendance_history():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    dept_filter = request.args.get('department', '')
    status_filter = request.args.get('status', '')
    try:
        page = max(1, int(request.args.get('page', 1)))
    except ValueError:
        page = 1
    per_page = 25

    conn = get_db_connection()

    query = '''
        SELECT attendance.id, attendance.time, attendance.date, attendance.status, attendance.method,
               users.name, users.employee_id, users.department, users.profile_photo
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE 1=1
    '''
    params = []

    if date_from:
        query += " AND attendance.date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND attendance.date <= ?"
        params.append(date_to)
    if dept_filter:
        query += " AND users.department = ?"
        params.append(dept_filter)
    if status_filter:
        query += " AND attendance.status = ?"
        params.append(status_filter)

    count_query = query.replace(
        "SELECT attendance.id, attendance.time, attendance.date, attendance.status, attendance.method,\n               users.name, users.employee_id, users.department, users.profile_photo",
        "SELECT COUNT(*)"
    )
    total_records = conn.execute(count_query, params).fetchone()[0]
    total_pages = max(1, (total_records + per_page - 1) // per_page)

    query += " ORDER BY attendance.date DESC, attendance.time DESC LIMIT ? OFFSET ?"
    params.extend([per_page, (page - 1) * per_page])

    records = conn.execute(query, params).fetchall()

    departments = conn.execute("SELECT DISTINCT department FROM users WHERE department IS NOT NULL ORDER BY department").fetchall()

    # Summary + one chart (merged from the old Analytics page): last 14 days.
    total_students = conn.execute("SELECT COUNT(*) FROM users WHERE role='employee'").fetchone()[0]
    chart_labels, chart_on_time, chart_late = [], [], []
    for i in range(13, -1, -1):
        day = datetime.now() - timedelta(days=i)
        d = day.strftime("%Y-%m-%d")
        present = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ? AND status IN ('Present', 'Late')", (d,)
        ).fetchone()[0]
        late = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ? AND status = 'Late'", (d,)
        ).fetchone()[0]
        chart_labels.append(day.strftime("%d %b"))
        chart_on_time.append(present - late)
        chart_late.append(late)
    month_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
    month_checkins = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date >= ? AND status IN ('Present', 'Late')", (month_start,)
    ).fetchone()[0]
    month_late = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date >= ? AND status = 'Late'", (month_start,)
    ).fetchone()[0]

    conn.close()

    return render_template("attendance_history.html",
                           total_students=total_students,
                           present_today=chart_on_time[-1] + chart_late[-1],
                           month_checkins=month_checkins,
                           month_late=month_late,
                           chart_labels=chart_labels,
                           chart_on_time=chart_on_time,
                           chart_late=chart_late,
                           records=records,
                           departments=departments,
                           date_from=date_from,
                           date_to=date_to,
                           dept_filter=dept_filter,
                           status_filter=status_filter,
                           page=page,
                           total_pages=total_pages,
                           total_records=total_records)


@app.route("/analytics")
def analytics():
    """Merged into the Attendance page (summary + chart on top)."""
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    return redirect(url_for("attendance_history"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        late_hour = request.form.get("late_cutoff_hour", "9")
        late_minute = request.form.get("late_cutoff_minute", "0")
        company_name = request.form.get("company_name", "")
        face_tolerance = request.form.get("face_tolerance", "0.45")
        min_face_samples = request.form.get("min_face_samples", "5")
        liveness_threshold = request.form.get("liveness_threshold", "0.5")

        # Email settings
        email_enabled = request.form.get("email_enabled", "0")
        email_hour = request.form.get("email_trigger_hour", "18")
        email_minute = request.form.get("email_trigger_minute", "0")
        hr_email_val = (request.form.get("hr_email", "") or "").strip().lower()

        try:
            valid = (0 <= int(late_hour) <= 23 and 0 <= int(late_minute) <= 59
                     and 0 <= int(email_hour) <= 23 and 0 <= int(email_minute) <= 59
                     and 1 <= int(min_face_samples) <= 15
                     and 0.3 <= float(face_tolerance) <= 0.6
                     and 0.1 <= float(liveness_threshold) <= 0.9)
        except ValueError:
            valid = False
        if not valid:
            flash("Enter a valid time, sample count and recognition thresholds.", "error")
            return redirect(url_for("settings"))

        set_setting('late_cutoff_hour', late_hour)
        set_setting('late_cutoff_minute', late_minute)
        set_setting('company_name', company_name)
        set_setting('face_tolerance', face_tolerance)
        set_setting('min_face_samples', min_face_samples)
        set_setting('liveness_threshold', liveness_threshold)
        set_setting('email_enabled', email_enabled)
        set_setting('email_trigger_hour', email_hour)
        set_setting('email_trigger_minute', email_minute)
        set_setting('hr_email', hr_email_val)
        set_setting('require_scanner_approval',
                    '1' if request.form.get('require_scanner_approval') == '1' else '0')

        # Update scheduler
        global scheduler
        if scheduler:
            try:
                scheduler.remove_job('absentee_emails')
            except Exception:
                pass

            if email_enabled == '1':
                from email_service import send_absentee_emails
                scheduler.add_job(
                    send_absentee_emails, 'cron',
                    hour=int(email_hour), minute=int(email_minute),
                    id='absentee_emails', replace_existing=True,
                    misfire_grace_time=3600
                )
                if not scheduler.running:
                    scheduler.start()

        flash("Settings updated successfully.", "success")
        return redirect(url_for("settings"))

    current_settings = {
        'late_cutoff_hour': get_setting('late_cutoff_hour', '9'),
        'late_cutoff_minute': get_setting('late_cutoff_minute', '0'),
        'company_name': get_setting('company_name', ''),
        'face_tolerance': get_setting('face_tolerance', '0.45'),
        'min_face_samples': get_setting('min_face_samples', '5'),
        'liveness_threshold': get_setting('liveness_threshold', '0.5'),
        'email_enabled': get_setting('email_enabled', '0'),
        'email_trigger_hour': get_setting('email_trigger_hour', '18'),
        'email_trigger_minute': get_setting('email_trigger_minute', '0'),
        'hr_email': get_setting('hr_email', ''),
        'require_scanner_approval': get_setting('require_scanner_approval', '0'),
    }

    conn = get_db_connection()
    scanner_devices = conn.execute(
        "SELECT d.id, d.label, d.user_agent, d.created_at, u.email AS approved_by_email "
        "FROM scanner_devices d LEFT JOIN users u ON u.id = d.approved_by "
        "WHERE d.revoked_at IS NULL ORDER BY d.created_at DESC"
    ).fetchall()
    conn.close()
    this_device = _current_scanner_device()

    return render_template("settings.html", settings=current_settings,
                           scanner_devices=scanner_devices,
                           this_device_id=this_device["id"] if this_device else None)


def _log_admin_action(action, detail):
    conn = get_db_connection()
    admin = conn.execute("SELECT email FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.execute(
        "INSERT INTO admin_audit_log (admin_user_id, admin_email, action, detail, ip_address) VALUES (?, ?, ?, ?, ?)",
        (session["user_id"], admin["email"] if admin else None, action, detail, request.remote_addr),
    )
    conn.commit()
    conn.close()


@app.route("/admin/scanner_devices/approve", methods=["POST"])
def admin_approve_scanner_device():
    """Approves THIS browser as an attendance scanner (sets its cookie)."""
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    if _current_scanner_device():
        flash("This device is already an approved scanner.", "success")
        return redirect(url_for("settings"))

    label = (request.form.get("label") or "").strip()[:60] or "Scanner"
    token = secrets.token_urlsafe(32)
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO scanner_devices (token_hash, label, approved_by, user_agent) VALUES (?, ?, ?, ?)",
        (_hash_scanner_token(token), label, session["user_id"], (request.headers.get("User-Agent") or "")[:200]),
    )
    conn.commit()
    conn.close()
    _log_admin_action("SCANNER_APPROVED", f"label={label}")

    flash(f'This device is now an approved scanner ("{label}").', "success")
    resp = redirect(url_for("settings"))
    resp.set_cookie(SCANNER_COOKIE_NAME, token, max_age=SCANNER_COOKIE_MAX_AGE,
                    httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


@app.route("/admin/scanner_devices/<int:device_id>/revoke", methods=["POST"])
def admin_revoke_scanner_device(device_id):
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))
    conn = get_db_connection()
    row = conn.execute("SELECT label FROM scanner_devices WHERE id = ? AND revoked_at IS NULL",
                       (device_id,)).fetchone()
    if row:
        conn.execute("UPDATE scanner_devices SET revoked_at = CURRENT_TIMESTAMP WHERE id = ?", (device_id,))
        conn.commit()
    conn.close()
    if row:
        _log_admin_action("SCANNER_REVOKED", f"device_id={device_id} label={row['label']}")
        flash(f'Scanner approval removed ("{row["label"]}").', "success")
    else:
        flash("That scanner device was not found or is already removed.", "danger")
    return redirect(url_for("settings"))


@app.route("/export_csv")
def export_csv():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    dept_filter = request.args.get('department', '')
    status_filter = request.args.get('status', '')

    conn = get_db_connection()
    query = '''
        SELECT users.employee_id, users.name, users.department,
               attendance.date, attendance.time, attendance.status, attendance.method
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE 1=1
    '''
    params = []

    if date_from:
        query += " AND attendance.date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND attendance.date <= ?"
        params.append(date_to)
    if dept_filter:
        query += " AND users.department = ?"
        params.append(dept_filter)
    if status_filter:
        query += " AND attendance.status = ?"
        params.append(status_filter)

    query += " ORDER BY attendance.date DESC, attendance.time DESC"

    records = conn.execute(query, params).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Student ID', 'Name', 'Department', 'Date', 'Time', 'Status', 'Method'])

    for r in records:
        writer.writerow([r['employee_id'], r['name'], r['department'],
                         r['date'], r['time'], r['status'], r['method'] or 'Face Recognition'])

    csv_content = output.getvalue()
    output.close()

    filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


# ============================================================
#  EMPLOYEE ROUTES
# ============================================================

@app.route("/employee")
def employee_dashboard():
    if session.get("role") != "employee":
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    history = conn.execute(
        "SELECT id, date, time, status FROM attendance WHERE user_id = ? ORDER BY date DESC, time DESC LIMIT 30",
        (session["user_id"],)
    ).fetchall()

    present_count = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE user_id = ? AND status = 'Present'",
        (session["user_id"],)
    ).fetchone()[0]

    # Stored status, over ALL records -- not recomputed from is_late() against
    # the *current* cutoff, and not limited to the last-30 `history` slice
    # (which under-counted late days beyond the most recent 30 records).
    late_count = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE user_id = ? AND status = 'Late'",
        (session["user_id"],)
    ).fetchone()[0]

    absent_count = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE user_id = ? AND status = 'Absent'",
        (session["user_id"],)
    ).fetchone()[0]

    leave_count = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE user_id = ? AND status = 'Leave'",
        (session["user_id"],)
    ).fetchone()[0]

    chart_data = {
        "Present": present_count,
        "Late": late_count,
        "Absent": absent_count,
        "Leave": leave_count
    }

    today = datetime.now().strftime("%Y-%m-%d")
    today_record = conn.execute(
        "SELECT time, status FROM attendance WHERE user_id = ? AND date = ?",
        (session["user_id"], today)
    ).fetchone()

    today_status = None
    if today_record:
        today_status = {
            'time': today_record['time'],
            'status': today_record['status']
        }

    current_month = datetime.fromisoformat(today).date().replace(day=1)
    try:
        selected_month = datetime.strptime(request.args.get("month", ""), "%Y-%m").date()
        selected_month = min(selected_month, current_month)
    except ValueError:
        selected_month = current_month
    days_in_month = monthrange(selected_month.year, selected_month.month)[1]
    month_records = conn.execute(
        "SELECT date, time, status FROM attendance WHERE user_id = ? AND date BETWEEN ? AND ? ORDER BY date, time",
        (session["user_id"], selected_month.isoformat(), selected_month.replace(day=days_in_month).isoformat())
    ).fetchall()

    calendar_data = {}
    for r in month_records:
        calendar_data[r['date']] = r['status']

    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good Morning"
    elif hour < 17:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"

    conn.close()

    return render_template("employee_dashboard.html",
                           user=user,
                           history=history,
                           chart_data=chart_data,
                           today_status=today_status,
                           calendar_data=calendar_data,
                           selected_month=selected_month, days_in_month=days_in_month,
                           current_month=current_month.isoformat()[:7], today=today,
                           greeting=greeting,
                           late_days=late_count,
                           # A Late check-in is still a day you showed up --
                           # "Days Present" should count Present + Late rows,
                           # with "Late Arrivals" as the "of which" breakdown,
                           # not a day excluded from Present entirely.
                           days_present_total=present_count + late_count,
                           min_face_samples=int(get_setting('min_face_samples', '5')))


@app.route("/scanner")
def scanner():
    return redirect(url_for('home'))


@app.route("/admin/wipe_model", methods=["POST"])
def admin_wipe_model():
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"})

    # Typed confirmation word required -- this deletes every user's face
    # data and ALL attendance rows.
    data = request.get_json(silent=True) or {}
    if str(data.get("confirm", "")).strip() != WIPE_CONFIRM_WORD:
        return jsonify({
            "success": False,
            "msg": f'Confirmation required: type "{WIPE_CONFIRM_WORD}" to wipe all face data and attendance.',
        }), 400

    conn = get_db_connection()
    admin = conn.execute("SELECT email FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    faces = conn.execute("SELECT COUNT(*) AS n FROM users WHERE face_registered = 1").fetchone()["n"]
    rows = conn.execute("SELECT COUNT(*) AS n FROM attendance").fetchone()["n"]
    conn.execute('UPDATE users SET face_registered = 0, face_embedding = NULL')
    conn.execute('DELETE FROM attendance')
    conn.execute(
        "INSERT INTO admin_audit_log (admin_user_id, admin_email, action, detail, ip_address) VALUES (?, ?, ?, ?, ?)",
        (session["user_id"], admin["email"] if admin else None, "WIPE_MODEL",
         f"face_registered_users_cleared={faces} attendance_rows_deleted={rows}", request.remote_addr),
    )
    conn.commit()
    app.logger.warning("WIPE_MODEL by admin user_id=%s email=%s ip=%s (faces=%s attendance_rows=%s)",
                       session["user_id"], admin["email"] if admin else None, request.remote_addr, faces, rows)
    conn.close()
    
    # Invalidate in-memory embedding cache
    from utils.face_utils import invalidate_cache
    invalidate_cache()
        
    return jsonify({"success": True, "msg": "All facial data and models successfully wiped!"})


# ============================================================
#  API ENDPOINTS
# ============================================================

@app.route("/api/register_face", methods=["POST"])
def api_register_face():
    if "user_id" not in session:
        return jsonify({"success": False, "msg": "Unauthorized"}), 401

    data = request.get_json(silent=True)
    base64_img = data.get("image") if isinstance(data, dict) else None

    success = register_face(session["user_id"], base64_img)
    if success == REGISTER_REJECTED_IDENTITY:
        return jsonify({"success": False, "identity_rejected": True,
                        "msg": "Only the account owner should be in front of the camera."}), 400
    if not success:
        return jsonify({"success": False, "msg": "No face detected. Please try again."}), 400

    saved, required = success
    return jsonify(success=True, saved=saved, required=required, complete=saved >= required,
                   msg="Face sample saved.")


@app.route("/api/recognize_face", methods=["POST"])
def api_recognize_face():
    """
    Pipeline (approved plan): frame -> BlazeFace detect -> trained
    MobileNetV3-Small liveness model -> if SPOOF/UNCERTAIN, reject + log +
    stop (Face Mesh/ArcFace never run) -> if LIVE, Face Mesh + ArcFace
    recognition -> blink/head-turn challenge -> mark attendance.

    The liveness model keeps running on every frame, including during the
    challenge (not just once) -- this is what makes the same-face
    consistency check below possible across the whole LIVE -> recognize ->
    challenge sequence, and smooths over a single bad frame during head
    movement via the per-session rolling average rather than hard-rejecting
    on one blip.
    """
    timings = {}
    t_request_start = time.perf_counter()
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = {}
        base64_img = data.get("image")
        # Any blink flag the browser sends ("liveness_verified") is ignored:
        # the blink is verified server-side from Face Mesh (_advance_blink).

        # Dev-only test-category label (e.g. "printed_photo") from the
        # scanner page's debug-only dropdown -- only honored with
        # FLASK_DEBUG=1, so this can't be used to write arbitrary free text
        # from an unauthenticated caller outside local development. Purely
        # descriptive: passed straight through to scan_events, never read
        # by any classification/decision code above or below this line.
        test_label = None
        if DEBUG_MODE:
            raw_label = data.get("test_label")
            if isinstance(raw_label, str) and raw_label.strip():
                test_label = raw_label.strip()[:40]

        # When "Require approved scanner device" is on, only approved devices
        # may run the pipeline / mark attendance.
        if not _scanner_allowed():
            return jsonify({
                "success": False, "recognized": False,
                "state": "DEVICE_NOT_APPROVED", "device_not_approved": True,
                "msg": MSG_DEVICE_NOT_APPROVED,
            }), 403

        if not base64_img:
            return jsonify({"success": False, "recognized": False, "msg": "No image data received."})

        if data.get("restart_attempt") is True:
            _reset_liveness_session()

        t0 = time.perf_counter()
        frame = data_uri_to_cv2_img(base64_img)
        timings["decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if frame is None:
            return jsonify({"success": False, "recognized": False, "msg": "Could not decode image."})

        # --- Stage 1: BlazeFace detect (the liveness model's own detector) ---
        t0 = time.perf_counter()
        detections = _liveness_detector.detect(frame)
        timings["detect_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if len(detections) == 0:
            _reset_liveness_session(keep_log_dedup=True)
            _log_scan_event("NO_FACE", test_label=test_label)
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            return jsonify({
                "success": False, "recognized": False, "face_found": False,
                "state": "NO_FACE", "timings": timings, "face_box": None,
                "msg": "Step into the frame.",
            })

        # Largest detected face's box, in pixel coords of THIS frame -- sent
        # to the frontend so it can draw a box at the real detection
        # location (mirrored for the mirrored video preview) instead of a
        # fixed decorative frame. Computed even in the multi-face case below
        # so the UI still has something real to show.
        best = max(detections, key=lambda d: d.width * d.height)
        cur_bbox = [best.x, best.y, best.width, best.height]
        frame_h, frame_w = frame.shape[:2]
        face_box = {
            "x": best.x, "y": best.y, "width": best.width, "height": best.height,
            "frame_width": frame_w, "frame_height": frame_h,
        }

        if len(detections) > 1:
            _reset_liveness_session(keep_log_dedup=True)
            _log_scan_event("MULTI_FACE", test_label=test_label)
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            return jsonify({
                "success": False, "recognized": False,
                "state": "MULTI_FACE", "timings": timings, "face_box": face_box,
                "msg": "Only one person at a time.",
                "multi_face": True,
            })

        # Same-face consistency: an abrupt position jump between frames means
        # this is no longer the same physical subject that was mid-sequence
        # -- reset everything (liveness smoothing + any in-progress
        # recognition challenge) rather than silently continuing as if
        # nothing happened.
        prev_bbox = session.get('last_bbox')
        frame_iou = _face_box_iou(prev_bbox, cur_bbox) if prev_bbox else None
        if _is_different_face(prev_bbox, cur_bbox):
            _log_scan_event("FACE_MOVED", dedup=False, update_dedup_key=False,
                             detail=f"prev={prev_bbox} cur={cur_bbox} iou={_face_box_iou(prev_bbox, cur_bbox):.3f}",
                             test_label=test_label, iou=frame_iou)
            _reset_liveness_session()
        session['last_bbox'] = cur_bbox

        # --- Stage 2: crop_face() (identical to anti_spoof_attendance's, see
        # tests/test_crop_identical.py) + the trained MobileNetV3-Small model
        # -- this REPLACES SmartFace's old rule-based screen/texture/depth
        # anti-spoofing entirely. ---
        crop = crop_face(frame, best)
        if crop is None or crop.size == 0:
            _log_scan_event("NO_FACE", detail="face too close to the frame edge", test_label=test_label)
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            return jsonify({
                "success": False, "recognized": False, "face_found": False,
                "state": "NO_FACE", "timings": timings, "face_box": face_box,
                "msg": "Step into the frame.",
            })

        t0 = time.perf_counter()
        raw_prob_spoof = _liveness_predictor.predict_prob(crop)
        timings["liveness_classify_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        smoothed_prob_spoof = _smoothed_prob_spoof(raw_prob_spoof)
        liveness_threshold = float(get_setting('liveness_threshold', '0.5'))
        decision = classify_three_way(smoothed_prob_spoof, *_gray_zone_band, threshold=liveness_threshold)
        frame_scores = {"p_spoof_frame": raw_prob_spoof, "p_spoof_avg": smoothed_prob_spoof, "iou": frame_iou}

        # Mask check (approved by the owner 2026-11-23): only block when the mask model is
        # confident on several frames in a row. A single 0.5 cut-off fired on glasses and a
        # hand passing the face, which blocked real check-ins.
        p_mask = mask_probability(frame[best.y:best.y + best.height, best.x:best.x + best.width])             if decision != "SPOOF" else 0.0
        mask_streak = session.get('mask_streak', 0) + 1 if p_mask >= MASK_MIN_PROB else 0
        session['mask_streak'] = mask_streak
        if mask_streak >= MASK_STREAK_FRAMES:
            _clear_blink_attempt()
            _log_scan_event("UNRECOGNIZED", detail=f"Face mask detected p_mask={round(p_mask, 3)}",
                            test_label=test_label, **frame_scores)
            return jsonify(success=False, recognized=False, state="MASK", face_box=face_box,
                           msg="Remove your face mask and look at the camera.")

        if decision != "LIVE":
            _log_liveness_event_if_changed(decision, smoothed_prob_spoof)
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            if decision == "SPOOF":
                # A CONFIRMED spoof frame ends any in-progress challenge --
                # the subject is not currently verified live, so nothing
                # they do should complete a challenge issued while they
                # briefly were (e.g. swapped to a photo mid-challenge).
                _clear_blink_attempt()
                _log_scan_event("SPOOF", detail=f"p_spoof={round(smoothed_prob_spoof, 4)}", test_label=test_label, **frame_scores)
                return jsonify({
                    "success": False, "recognized": False,
                    "state": "SPOOF", "spoofing_detected": True,
                    "p_spoof": round(smoothed_prob_spoof, 4), "timings": timings, "face_box": face_box,
                    # Binary classifier -- it cannot tell photo/screen/phone
                    # apart, so the message never claims a specific attack type.
                    "msg": "Spoof detected — please use your real face.",
                })
            # UNCERTAIN: reject + log + retry THIS frame only (decision #1)
            # -- deliberately does NOT clear the in-progress challenge.
            # A real head turn or blink can transiently blur the crop enough
            # to tip the smoothed score into the gray zone for a frame or
            # two; clearing the challenge on that blip would force a brand
            # new RANDOM challenge once liveness recovers, discarding
            # whatever progress the same person already made on the
            # original one. This was a real bug (caught via live testing):
            # the challenge type changed mid-attempt even though the same
            # physical subject never left the frame. No security loss from
            # keeping it -- an UNCERTAIN frame never reaches recognition or
            # advances the challenge state machine below, so preserving the
            # challenge slot only ever helps a genuinely live, still-present
            # subject resume where they left off.
            _log_scan_event("UNCERTAIN", detail=f"p_spoof={round(smoothed_prob_spoof, 4)}", test_label=test_label, **frame_scores)
            return jsonify({
                "success": False, "recognized": False,
                "state": "UNCERTAIN",
                "p_spoof": round(smoothed_prob_spoof, 4), "timings": timings, "face_box": face_box,
                "msg": "Hold still — checking again.",
            })

        _log_liveness_event_if_changed("LIVE", smoothed_prob_spoof)

        # --- Stage 3: LIVE -> now (and only now) run Face Mesh + ArcFace ---
        face_tolerance = float(get_setting('face_tolerance', '0.45'))
        t0 = time.perf_counter()
        user_id, liveness_metrics, confidence, multi_face = recognize_face_with_liveness(frame, match_threshold=face_tolerance)
        timings["recognition_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)

        if multi_face:
            _log_scan_event("MULTI_FACE", test_label=test_label, **frame_scores)
            _reset_liveness_session()
            return jsonify({
                "success": False, "recognized": False,
                "state": "MULTI_FACE", "timings": timings, "face_box": face_box,
                "msg": "Only one person at a time.",
                "multi_face": True,
            })

        if user_id:
            conn = get_db_connection()
            user_info = conn.execute("SELECT name, employee_id, department FROM users WHERE id=?", (user_id,)).fetchone()
            conn.close()

            if not user_info:
                return jsonify({"success": False, "recognized": False, "state": "LIVE", "timings": timings, "face_box": face_box, "msg": "User not found."})

            # Logged-in student: only their own face may mark attendance.
            # The public scanner (no student session) still recognizes anyone.
            if session.get("role") == "employee" and user_id != session.get("user_id"):
                _clear_blink_attempt()
                _log_scan_event("UNRECOGNIZED", user_id=session.get("user_id"),
                                 detail=f"account_mismatch matched_user_id={user_id}", test_label=test_label, **frame_scores)
                return jsonify({
                    "success": False, "recognized": False, "state": "LIVE",
                    "face_found": True, "account_mismatch": True,
                    "p_spoof": round(smoothed_prob_spoof, 4),
                    "timings": timings, "face_box": face_box,
                    "msg": MSG_ACCOUNT_MISMATCH,
                })

            # Random active challenge, decided only by the server (see
            # CHALLENGE_TIMEOUT_SECONDS above): one random head turn, then one
            # natural blink, by this same user within the time limit. A different
            # matched user, a timeout or a turn the wrong way starts a NEW random
            # challenge, so a pre-recorded video can't just be replayed.
            if 'challenge_type' not in session or session.get('challenge_user') != user_id \
                    or time.time() - session.get('challenge_assigned_at', 0) > CHALLENGE_TIMEOUT_SECONDS:
                _new_challenge(user_id)

            progress = _advance_challenge(liveness_metrics)
            if progress == 'wrong':
                _log_scan_event("RECOGNIZED", user_id=user_id, detail=f"turned the wrong way (asked {session.get('challenge_turn')}); new challenge",
                                dedup=False, test_label=test_label, **frame_scores)
                _new_challenge(user_id)
                progress = 'turn'

            if progress != 'done':
                step = f"turn_{session.get('challenge_turn')}" if progress == 'turn' else 'blink'
                _log_scan_event("RECOGNIZED", user_id=user_id, detail=f"challenge: {step}", test_label=test_label, **frame_scores)
                return jsonify({
                    "success": False, "recognized": True, "state": "LIVE",
                    "liveness_metrics": liveness_metrics,
                    "p_spoof": round(smoothed_prob_spoof, 4),
                    "confidence": confidence, "timings": timings, "face_box": face_box,
                    "challenge": step, "challenge_pending": True,
                    "instruction": _challenge_instruction(),
                    "msg": _challenge_instruction(),
                    "user": user_info['name'],
                    "emp_id": user_info['employee_id'],
                    "department": user_info['department'],
                })
            challenge_done = f"turn_{session.get('challenge_turn')}+blink"

            # Turn + blink verified by the server. Clear the attempt.
            _clear_blink_attempt()

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            status = "Late" if is_late(time_str) else "Present"

            conn = get_db_connection()
            existing = conn.execute(
                "SELECT time, status FROM attendance WHERE user_id=? AND date=? ORDER BY time DESC LIMIT 1",
                (user_id, date_str)
            ).fetchone()

            if not existing:
                conn.execute("INSERT INTO attendance (user_id, date, time, status, method) VALUES (?, ?, ?, ?, ?)",
                             (user_id, date_str, time_str, status, 'Face Recognition'))
                conn.commit()
                conn.close()
                _log_scan_event("CHALLENGE_PASSED", user_id=user_id, detail=f"challenge={challenge_done}",
                                 dedup=False, update_dedup_key=False, test_label=test_label, **frame_scores)
                _log_scan_event("ATTENDANCE_MARKED", user_id=user_id, detail=f"status={status} time={time_str}",
                                 dedup=False, test_label=test_label, **frame_scores)
                return jsonify({
                    "success": True, "recognized": True, "state": "LIVE", "face_box": face_box,
                    "msg": MSG_ATTENDANCE_MARKED,
                    "user": user_info['name'],
                    "emp_id": user_info['employee_id'],
                    "department": user_info['department'],
                    "status": status, "time": time_str,
                    "p_spoof": round(smoothed_prob_spoof, 4),
                    "confidence": confidence, "timings": timings,
                })
            else:
                conn.close()
                _log_scan_event("CHALLENGE_PASSED", user_id=user_id, detail=f"challenge={challenge_done}",
                                 dedup=False, update_dedup_key=False, test_label=test_label, **frame_scores)
                _log_scan_event("ALREADY_MARKED", user_id=user_id, detail="already checked in today",
                                 dedup=False, test_label=test_label, **frame_scores)
                return jsonify({
                    "success": True, "recognized": True, "state": "LIVE", "face_box": face_box,
                    "msg": MSG_ALREADY_MARKED,
                    "status": existing["status"],
                    "time": existing["time"],
                    "user": user_info['name'],
                    "emp_id": user_info['employee_id'],
                    "department": user_info['department'],
                    "p_spoof": round(smoothed_prob_spoof, 4),
                    "confidence": confidence, "timings": timings,
                })

        # LIVE, face found, but not recognized (either unregistered, or
        # fewer than min_face_samples enrolled -- see register_face()).
        # If a blink attempt is in progress for this same tracked face (any
        # face jump / no-face / multi-face would have reset the session) and
        # still inside its window, a closed-eye frame may still count: closed
        # eyes can make recognition fail for that one frame.
        # (A turned head or closed eyes can make recognition miss one frame of the
        # same tracked face; the challenge may progress but can't finish here.)
        if session.get('challenge_type'):
            if time.time() - session.get('challenge_assigned_at', 0) <= CHALLENGE_TIMEOUT_SECONDS \
                    and _advance_challenge(liveness_metrics, allow_complete=False) != 'wrong':
                pass
            else:
                _clear_blink_attempt()
        _log_scan_event("UNRECOGNIZED", test_label=test_label, **frame_scores)
        return jsonify({
            "success": False, "recognized": False, "state": "LIVE",
            "face_found": True, "liveness_metrics": liveness_metrics,
            "challenge_pending": bool(session.get("challenge_type")),
            "instruction": _challenge_instruction() if session.get("challenge_type") else None,
            "p_spoof": round(smoothed_prob_spoof, 4),
            "confidence": confidence, "timings": timings, "face_box": face_box,
            "msg": MSG_FACE_NOT_RECOGNIZED,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False, "recognized": False,
            "msg": f"Recognition engine error: {str(e)}",
            "error": True,
        }), 500


@app.route("/api/edit_attendance", methods=["POST"])
def api_edit_attendance():
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401

    data = request.json
    record_id = data.get("id")

    new_time, t_err = validate_time(data.get("time", ""))
    new_date, d_err = validate_date(data.get("date", ""))
    new_status, s_err = validate_status(data.get("status", ""))

    if t_err:
        return jsonify({"success": False, "msg": t_err}), 400
    if d_err:
        return jsonify({"success": False, "msg": d_err}), 400
    if s_err:
        return jsonify({"success": False, "msg": s_err}), 400

    conn = get_db_connection()
    conn.execute('UPDATE attendance SET time = ?, date = ?, status = ? WHERE id = ?',
                 (new_time, new_date, new_status, record_id))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "msg": "Attendance updated."})


@app.route("/api/system_status")
def api_system_status():
    """System health check for admin dashboard.

    Previously imported get_face_detector/TRAINER_PATH from utils.face_utils
    and read a 'face_data' directory -- none of which exist in this engine
    (leftover from an older LBPH-based version), so this always fell into
    the except branch and reported 'ai: Error'. Now reports the real,
    currently-loaded liveness model and ArcFace engine state."""
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401

    status = {}

    # Camera (always 'Available' on server — actual check is client-side)
    status['camera'] = {'status': 'Available', 'detail': 'Browser webcam access'}

    # AI Engine: the real trained liveness model + ArcFace recognition engine
    try:
        from utils.face_utils import _arcface_session
        arcface_ready = _arcface_session is not None
        liveness_ready = _liveness_predictor is not None and _liveness_detector is not None
        if liveness_ready and arcface_ready:
            status['ai'] = {
                'status': 'Active',
                'detail': f"Liveness: MobileNetV3-Small on {_liveness_predictor.device} | ArcFace: w600k_r50 loaded",
            }
        else:
            missing = []
            if not liveness_ready:
                missing.append("liveness model")
            if not arcface_ready:
                missing.append("ArcFace model")
            status['ai'] = {'status': 'Degraded', 'detail': f"Not loaded: {', '.join(missing)}"}
    except Exception as e:
        status['ai'] = {'status': 'Error', 'detail': str(e)}

    # Database
    try:
        conn = get_db_connection()
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        status['database'] = {'status': 'Healthy', 'detail': f'{user_count} users'}
    except Exception as e:
        status['database'] = {'status': 'Error', 'detail': str(e)}

    # Email
    email_enabled = get_setting('email_enabled', '0')
    hr = get_setting('email_trigger_hour', '18')
    mn = get_setting('email_trigger_minute', '0')
    if email_enabled == '1':
        status['email'] = {'status': 'Enabled', 'detail': f'Auto-trigger: {hr}:{int(mn):02d}'}
    else:
        status['email'] = {'status': 'Disabled', 'detail': f'Auto-trigger: {hr}:{int(mn):02d}'}

    return jsonify(status)


@app.route("/api/manual_attendance", methods=["POST"])
def api_manual_attendance():
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401

    data = request.json
    user_id = data.get("user_id")

    status, s_err = validate_status(data.get("status", "Present"))
    date_str, d_err = validate_date(data.get("date", datetime.now().strftime("%Y-%m-%d")))
    time_str, t_err = validate_time(data.get("time", datetime.now().strftime("%H:%M:%S")))

    if s_err:
        return jsonify({"success": False, "msg": s_err}), 400
    if d_err:
        return jsonify({"success": False, "msg": d_err}), 400
    if t_err:
        return jsonify({"success": False, "msg": t_err}), 400

    if not user_id:
        return jsonify({"success": False, "msg": "Student not specified."}), 400

    conn = get_db_connection()
    existing = conn.execute("SELECT id FROM attendance WHERE user_id=? AND date=?", (user_id, date_str)).fetchone()
    if existing:
        conn.close()
        return jsonify({"success": False, "msg": "Attendance already marked for this date."})

    conn.execute(
        "INSERT INTO attendance (user_id, date, time, status, method) VALUES (?, ?, ?, ?, ?)",
        (user_id, date_str, time_str, status, 'Manual Entry')
    )
    conn.commit()
    conn.close()

    return jsonify({"success": True, "msg": "Attendance marked successfully."})


@app.route("/api/admin_add_employee", methods=["POST"])
def api_admin_add_employee():
    """Admin can add an employee directly with auto-generated password."""
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401

    data = request.json

    # Validate all fields
    emp_id, err = validate_employee_id(data.get("employee_id", ""))
    if err:
        return jsonify({"success": False, "msg": err}), 400

    name, err = validate_name(data.get("name", ""))
    if err:
        return jsonify({"success": False, "msg": err}), 400

    email, err = validate_email(data.get("email", ""))
    if err:
        return jsonify({"success": False, "msg": err}), 400

    phone, err = validate_phone(data.get("phone", ""))
    if err:
        return jsonify({"success": False, "msg": err}), 400

    dept, err = validate_department(data.get("department", ""))
    if err:
        return jsonify({"success": False, "msg": err}), 400

    # Generate secure password
    gen_password = generate_secure_password(12)
    hashed_pw = generate_password_hash(gen_password)

    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO users (employee_id, name, email, phone, department, password_hash, role)
            VALUES (?, ?, ?, ?, ?, ?, 'employee')
        ''', (emp_id, name, email, phone, dept, hashed_pw))
        conn.commit()
        conn.close()
        return jsonify({
            "success": True,
            "msg": f"Student {name} added successfully.",
            "generated_password": gen_password
        })
    except Exception:
        conn.close()
        return jsonify({"success": False, "msg": "Student ID or Email already exists."}), 400


@app.route("/api/search_employees")
def api_search_employees():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401

    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"results": []})

    conn = get_db_connection()
    results = conn.execute('''
        SELECT id, employee_id, name, department, face_registered, profile_photo
        FROM users
        WHERE role = 'employee' AND (name LIKE ? OR employee_id LIKE ? OR email LIKE ?)
        ORDER BY name
        LIMIT 20
    ''', (f'%{query}%', f'%{query}%', f'%{query}%')).fetchall()
    conn.close()

    return jsonify({
        "results": [dict(r) for r in results]
    })


@app.route("/api/attendance_feed")
def api_attendance_feed():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db_connection()

    records = conn.execute('''
        SELECT attendance.time, attendance.status, attendance.method,
               users.name, users.employee_id, users.department, users.profile_photo
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE attendance.date = ?
        ORDER BY attendance.time DESC LIMIT 15
    ''', (today,)).fetchall()

    feed = []
    for r in records:
        feed.append({
            "name": r["name"],
            "emp_id": r["employee_id"],
            "time": r["time"],
            "status": r["status"],
            "department": r["department"],
            "method": r["method"] or "Face Recognition",
            "photo": r["profile_photo"] or ""
        })

    conn.close()
    return jsonify({"feed": feed})



@app.route("/api/upload_photo", methods=["POST"])
def api_upload_photo():
    """Upload profile photo via API."""
    if "user_id" not in session:
        return jsonify({"success": False, "msg": "Unauthorized"}), 401

    if 'photo' not in request.files:
        return jsonify({"success": False, "msg": "No file uploaded."}), 400

    file = request.files['photo']
    is_valid, err = validate_profile_photo(file)
    if not is_valid:
        return jsonify({"success": False, "msg": err}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"profile_{session['user_id']}_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)

    conn = get_db_connection()
    # Delete old photo
    old = conn.execute("SELECT profile_photo FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if old and old["profile_photo"]:
        old_path = os.path.join(Config.UPLOAD_FOLDER, old["profile_photo"])
        if os.path.exists(old_path):
            os.remove(old_path)

    conn.execute("UPDATE users SET profile_photo = ? WHERE id = ?", (filename, session["user_id"]))
    conn.commit()
    conn.close()

    session["profile_photo"] = filename

    return jsonify({"success": True, "msg": "Photo uploaded.", "filename": filename})


@app.route("/api/send_test_email", methods=["POST"])
def api_send_test_email():
    """Send a test email to verify SMTP configuration."""
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"}), 401

    if not Config.MAIL_USERNAME or not Config.MAIL_PASSWORD:
        return jsonify({"success": False, "msg": "Mail credentials not configured in environment variables."}), 400

    try:
        import smtplib
        from email.mime.text import MIMEText

        server = smtplib.SMTP(Config.MAIL_SERVER, Config.MAIL_PORT)
        server.starttls()
        server.login(Config.MAIL_USERNAME, Config.MAIL_PASSWORD)

        msg = MIMEText("This is a test email from SecureFace Attendance. Email notifications are working correctly!")
        msg['From'] = Config.MAIL_DEFAULT_SENDER
        msg['To'] = Config.MAIL_USERNAME
        msg['Subject'] = "SecureFace Attendance: Test Email ✓"

        server.send_message(msg)
        server.quit()

        return jsonify({"success": True, "msg": "Test email sent successfully!"})
    except Exception as e:
        return jsonify({"success": False, "msg": f"Email failed: {str(e)}"}), 500


if __name__ == "__main__":
    # debug=True previously hardcoded -- Flask's debugger (Werkzeug) exposes
    # a remote code execution console on any unhandled exception, which must
    # never be on outside local development. Set FLASK_DEBUG=1 to enable it.
    app.run(debug=DEBUG_MODE)
