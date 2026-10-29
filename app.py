import os
import io
import csv
import uuid
import sqlite3
import random
import time
import hashlib
import math
from collections import deque
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, session, jsonify, flash, url_for, Response
from flask_wtf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
from database.db import get_db_connection, init_db, get_setting, set_setting, _DEFAULT_SETTINGS
from utils.face_utils import register_face, recognize_face_with_liveness, data_uri_to_cv2_img
from utils.validators import (
    validate_registration, validate_profile_update, validate_name,
    validate_employee_id, validate_email, validate_phone, validate_password,
    validate_department, validate_status, validate_date, validate_time,
    validate_profile_photo, generate_secure_password, ALLOWED_DEPARTMENTS
)
from liveness.face_utils import FaceDetector, crop_face
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
FACE_MOVE_REJECT_RATIO = 0.6     # center displacement > 60% of face size => treat as a different physical subject


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


def _reset_liveness_session():
    """Clears all per-session liveness/challenge state -- called on no-face,
    multi-face, an abrupt face-position jump, or a logged-out SPOOF/UNCERTAIN
    decision, so a new physical subject always starts from a clean slate."""
    session.pop('recent_probs', None)
    session.pop('last_bbox', None)
    session.pop('last_logged_decision', None)
    session.pop('last_logged_stage', None)
    session.pop('challenge_type', None)
    session.pop('challenge_user', None)
    session.pop('challenge_neutral_seen', None)


def _face_moved_too_much(prev_bbox, cur_bbox):
    """True if the face's center moved more than FACE_MOVE_REJECT_RATIO of
    its own size since the last frame -- a same-face consistency check: the
    face that passes liveness must stay the same physical subject through
    recognition and the challenge, not be swapped mid-sequence."""
    if prev_bbox is None:
        return False
    px, py, pw, ph = prev_bbox
    cx, cy, cw, ch = cur_bbox
    pcx, pcy = px + pw / 2.0, py + ph / 2.0
    ccx, ccy = cx + cw / 2.0, cy + ch / 2.0
    dist = math.hypot(ccx - pcx, ccy - pcy)
    ref_size = max(cw, ch, 1)
    return (dist / ref_size) > FACE_MOVE_REJECT_RATIO


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


def _log_scan_event(stage, user_id=None, detail=None, dedup=True, update_dedup_key=True):
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
    """
    sid = _get_liveness_session_id()
    if dedup and session.get('last_logged_stage') == stage:
        return
    if update_dedup_key:
        session['last_logged_stage'] = stage
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO scan_events (session_id, stage, user_id, detail) VALUES (?, ?, ?, ?)",
        (sid, stage, user_id, detail),
    )
    conn.commit()
    conn.close()


# ============================================================
#  PUBLIC ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("scanner.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session and session.get("role") == "employee":
        return redirect(url_for("employee_dashboard"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("login.html")

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ? AND role = 'employee'", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["role"] = user["role"]
            session["name"] = user["name"]
            session["employee_id"] = user["employee_id"]
            session["profile_photo"] = user["profile_photo"] or ""
            return redirect(url_for("employee_dashboard"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if "user_id" in session and session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("admin_login.html")

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ? AND role = 'admin'", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["role"] = user["role"]
            session["name"] = user["name"]
            session["employee_id"] = user["employee_id"]
            session["profile_photo"] = user["profile_photo"] or ""
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin email or password.", "danger")

    return render_template("admin_login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
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

    present_today = conn.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ?", (today,)).fetchone()[0]
    present_yesterday = conn.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ?", (yesterday,)).fetchone()[0]

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

    weekly_data = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        count = conn.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ?", (d,)).fetchone()[0]
        weekly_data.append(count)

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
               scan_events.detail, scan_events.created_at,
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


@app.route("/attendance_history")
def attendance_history():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    dept_filter = request.args.get('department', '')
    status_filter = request.args.get('status', '')
    page = int(request.args.get('page', 1))
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

    conn.close()

    return render_template("attendance_history.html",
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
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    conn = get_db_connection()

    weekly_labels = []
    weekly_present = []
    weekly_late = []
    weekly_absent = []
    total_emp = conn.execute("SELECT COUNT(*) FROM users WHERE role='employee'").fetchone()[0]

    for i in range(13, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        label = (datetime.now() - timedelta(days=i)).strftime("%d %b")
        weekly_labels.append(label)

        present = conn.execute("SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date = ?", (d,)).fetchone()[0]
        late = conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Late'", (d,)
        ).fetchone()[0]

        weekly_present.append(present)
        weekly_late.append(late)
        weekly_absent.append(max(0, total_emp - present))

    month_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
    dept_data = conn.execute('''
        SELECT users.department, COUNT(DISTINCT attendance.user_id) as unique_present, COUNT(*) as total_records
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE attendance.date >= ?
        GROUP BY users.department
        ORDER BY total_records DESC
    ''', (month_start,)).fetchall()

    dept_labels = [d['department'] or 'Unknown' for d in dept_data]
    dept_values = [d['total_records'] for d in dept_data]

    this_month_present = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date >= ? AND status IN ('Present', 'Late')",
        (month_start,)
    ).fetchone()[0]

    this_month_late = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date >= ? AND status = 'Late'",
        (month_start,)
    ).fetchone()[0]

    days_elapsed = (datetime.now() - datetime.strptime(month_start, "%Y-%m-%d")).days + 1
    avg_daily = round(this_month_present / max(1, days_elapsed), 1)
    avg_rate = round((avg_daily / max(1, total_emp)) * 100, 1) if total_emp > 0 else 0

    top_late = conn.execute('''
        SELECT users.name, users.employee_id, users.department, COUNT(*) as late_count
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE attendance.status = 'Late' AND attendance.date >= ?
        GROUP BY attendance.user_id
        ORDER BY late_count DESC
        LIMIT 5
    ''', (month_start,)).fetchall()

    top_punctual = conn.execute('''
        SELECT users.name, users.employee_id, users.department,
               COUNT(*) as total_days,
               SUM(CASE WHEN attendance.status = 'Present' THEN 1 ELSE 0 END) as present_days
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE attendance.date >= ?
        GROUP BY attendance.user_id
        ORDER BY present_days DESC
        LIMIT 5
    ''', (month_start,)).fetchall()

    conn.close()

    return render_template("analytics.html",
                           weekly_labels=weekly_labels,
                           weekly_present=weekly_present,
                           weekly_late=weekly_late,
                           weekly_absent=weekly_absent,
                           dept_labels=dept_labels,
                           dept_values=dept_values,
                           total_emp=total_emp,
                           avg_rate=avg_rate,
                           avg_daily=avg_daily,
                           this_month_present=this_month_present,
                           this_month_late=this_month_late,
                           top_late=top_late,
                           top_punctual=top_punctual)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        late_hour = request.form.get("late_cutoff_hour", "9")
        late_minute = request.form.get("late_cutoff_minute", "0")
        company_name = request.form.get("company_name", "Sofzenix Technologies")
        face_tolerance = request.form.get("face_tolerance", "0.45")
        min_face_samples = request.form.get("min_face_samples", "5")
        liveness_threshold = request.form.get("liveness_threshold", "0.5")

        # Email settings
        email_enabled = request.form.get("email_enabled", "0")
        email_hour = request.form.get("email_trigger_hour", "18")
        email_minute = request.form.get("email_trigger_minute", "0")
        hr_email_val = (request.form.get("hr_email", "") or "").strip().lower()

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
        'company_name': get_setting('company_name', 'Sofzenix Technologies'),
        'face_tolerance': get_setting('face_tolerance', '0.45'),
        'min_face_samples': get_setting('min_face_samples', '5'),
        'liveness_threshold': get_setting('liveness_threshold', '0.5'),
        'email_enabled': get_setting('email_enabled', '0'),
        'email_trigger_hour': get_setting('email_trigger_hour', '18'),
        'email_trigger_minute': get_setting('email_trigger_minute', '0'),
        'hr_email': get_setting('hr_email', ''),
    }

    return render_template("settings.html", settings=current_settings)


@app.route("/export_csv")
def export_csv():
    if session.get("role") != "admin":
        return redirect(url_for("admin_login"))

    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')

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

    month_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
    month_records = conn.execute(
        "SELECT date, time, status FROM attendance WHERE user_id = ? AND date >= ? ORDER BY date",
        (session["user_id"], month_start)
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
                           greeting=greeting,
                           late_days=late_count,
                           min_face_samples=int(get_setting('min_face_samples', '5')))


@app.route("/scanner")
def scanner():
    return redirect(url_for('home'))


@app.route("/admin/wipe_model", methods=["POST"])
def admin_wipe_model():
    if session.get("role") != "admin":
        return jsonify({"success": False, "msg": "Unauthorized"})
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET face_registered = 0, face_embedding = NULL')
    conn.execute('DELETE FROM attendance')
    conn.commit()
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

    data = request.json
    base64_img = data.get("image")

    success = register_face(session["user_id"], base64_img)
    if not success:
        return jsonify({"success": False, "msg": "No face detected. Please try again."}), 400

    return jsonify({"success": True, "msg": "Face registered successfully!"})


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
        data = request.json
        base64_img = data.get("image")
        liveness_verified = data.get("liveness_verified", False)

        if not base64_img:
            return jsonify({"success": False, "recognized": False, "msg": "No image data received."})

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
            _reset_liveness_session()
            _log_scan_event("NO_FACE")
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            return jsonify({
                "success": False, "recognized": False, "face_found": False,
                "state": "NO_FACE", "timings": timings,
                "msg": "Waiting for valid subject...",
            })

        if len(detections) > 1:
            _reset_liveness_session()
            _log_scan_event("MULTI_FACE")
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            return jsonify({
                "success": False, "recognized": False,
                "state": "MULTI_FACE", "timings": timings,
                "msg": "Multiple faces detected. Only one person at a time.",
                "multi_face": True,
            })

        best = max(detections, key=lambda d: d.width * d.height)
        cur_bbox = [best.x, best.y, best.width, best.height]

        # Same-face consistency: an abrupt position jump between frames means
        # this is no longer the same physical subject that was mid-sequence
        # -- reset everything (liveness smoothing + any in-progress
        # recognition challenge) rather than silently continuing as if
        # nothing happened.
        prev_bbox = session.get('last_bbox')
        if _face_moved_too_much(prev_bbox, cur_bbox):
            _log_scan_event("FACE_MOVED", dedup=False, update_dedup_key=False,
                             detail=f"prev={prev_bbox} cur={cur_bbox}")
            _reset_liveness_session()
        session['last_bbox'] = cur_bbox

        # --- Stage 2: crop_face() (identical to anti_spoof_attendance's, see
        # tests/test_crop_identical.py) + the trained MobileNetV3-Small model
        # -- this REPLACES SmartFace's old rule-based screen/texture/depth
        # anti-spoofing entirely. ---
        crop = crop_face(frame, best)
        if crop is None or crop.size == 0:
            _log_scan_event("NO_FACE", detail="face too close to the frame edge")
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            return jsonify({
                "success": False, "recognized": False, "face_found": False,
                "state": "NO_FACE", "timings": timings,
                "msg": "Face too close to the frame edge.",
            })

        t0 = time.perf_counter()
        raw_prob_spoof = _liveness_predictor.predict_prob(crop)
        timings["liveness_classify_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        smoothed_prob_spoof = _smoothed_prob_spoof(raw_prob_spoof)
        liveness_threshold = float(get_setting('liveness_threshold', '0.5'))
        decision = classify_three_way(smoothed_prob_spoof, *_gray_zone_band, threshold=liveness_threshold)

        if decision != "LIVE":
            _log_liveness_event_if_changed(decision, smoothed_prob_spoof)
            # A SPOOF/UNCERTAIN frame also ends any in-progress challenge --
            # the subject in front of the camera is not currently verified
            # live, so nothing they do should be able to complete a
            # challenge that was issued while they briefly were.
            session.pop('challenge_type', None)
            session.pop('challenge_user', None)
            session.pop('challenge_neutral_seen', None)
            timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)
            if decision == "SPOOF":
                _log_scan_event("SPOOF", detail=f"p_spoof={round(smoothed_prob_spoof, 4)}")
                return jsonify({
                    "success": False, "recognized": False,
                    "state": "SPOOF", "spoofing_detected": True,
                    "p_spoof": round(smoothed_prob_spoof, 4), "timings": timings,
                    "msg": "Spoofing detected — live presence required.",
                })
            _log_scan_event("UNCERTAIN", detail=f"p_spoof={round(smoothed_prob_spoof, 4)}")
            return jsonify({
                "success": False, "recognized": False,
                "state": "UNCERTAIN",
                "p_spoof": round(smoothed_prob_spoof, 4), "timings": timings,
                "msg": "Hold still — checking again",
            })

        _log_liveness_event_if_changed("LIVE", smoothed_prob_spoof)

        # --- Stage 3: LIVE -> now (and only now) run Face Mesh + ArcFace ---
        face_tolerance = float(get_setting('face_tolerance', '0.45'))
        t0 = time.perf_counter()
        user_id, liveness_metrics, confidence, multi_face = recognize_face_with_liveness(frame, match_threshold=face_tolerance)
        timings["recognition_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t_request_start) * 1000, 1)

        if multi_face:
            _log_scan_event("MULTI_FACE")
            _reset_liveness_session()
            return jsonify({
                "success": False, "recognized": False,
                "state": "MULTI_FACE", "timings": timings,
                "msg": "Multiple faces detected. Only one person at a time.",
                "multi_face": True,
            })

        if user_id:
            conn = get_db_connection()
            user_info = conn.execute("SELECT name, employee_id, department FROM users WHERE id=?", (user_id,)).fetchone()
            conn.close()

            if not user_info:
                return jsonify({"success": False, "recognized": False, "state": "LIVE", "timings": timings, "msg": "User not found."})

            # Liveness Challenge — RANDOM action to defeat video replay.
            # Videos can fake blinks but can't predict random challenges.
            # Unchanged from the original SmartFace logic.
            challenge_types = ['blink', 'turn_left', 'turn_right']
            if 'challenge_type' not in session or session.get('challenge_user') != user_info['name']:
                session['challenge_type'] = random.choice(challenge_types)
                session['challenge_user'] = user_info['name']
                session['challenge_neutral_seen'] = False

            challenge = session['challenge_type']
            head_yaw = liveness_metrics.get('head_yaw', 0)

            challenge_passed = False
            if challenge == 'blink' and liveness_verified:
                challenge_passed = True
            elif challenge == 'turn_left':
                if abs(head_yaw) < 0.08:
                    session['challenge_neutral_seen'] = True
                if session.get('challenge_neutral_seen') and head_yaw < -0.12:
                    challenge_passed = True
            elif challenge == 'turn_right':
                if abs(head_yaw) < 0.08:
                    session['challenge_neutral_seen'] = True
                if session.get('challenge_neutral_seen') and head_yaw > 0.12:
                    challenge_passed = True

            if not challenge_passed:
                challenge_msgs = {
                    'blink': '👁️ Blink your eyes to verify',
                    'turn_left': '↩️ Turn your head LEFT slowly',
                    'turn_right': '↪️ Turn your head RIGHT slowly',
                }
                _log_scan_event("RECOGNIZED", user_id=user_id, detail=f"awaiting challenge={challenge}")
                return jsonify({
                    "success": False, "recognized": True, "state": "LIVE",
                    "liveness_metrics": liveness_metrics,
                    "p_spoof": round(smoothed_prob_spoof, 4),
                    "confidence": confidence, "timings": timings,
                    "challenge": challenge,
                    "msg": f"{user_info['name']} — {challenge_msgs.get(challenge, 'Verify')}",
                    "user": user_info['name'],
                    "emp_id": user_info['employee_id'],
                    "department": user_info['department'],
                })

            # Challenge passed! Clear session
            session.pop('challenge_type', None)
            session.pop('challenge_user', None)
            session.pop('challenge_neutral_seen', None)

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            status = "Late" if is_late(time_str) else "Present"

            conn = get_db_connection()
            existing = conn.execute("SELECT id FROM attendance WHERE user_id=? AND date=?", (user_id, date_str)).fetchone()

            if not existing:
                conn.execute("INSERT INTO attendance (user_id, date, time, status, method) VALUES (?, ?, ?, ?, ?)",
                             (user_id, date_str, time_str, status, 'Face Recognition'))
                conn.commit()
                conn.close()
                _log_scan_event("CHALLENGE_PASSED", user_id=user_id, detail=f"challenge={challenge}",
                                 dedup=False, update_dedup_key=False)
                _log_scan_event("ATTENDANCE_MARKED", user_id=user_id, detail=f"status={status} time={time_str}",
                                 dedup=False)
                return jsonify({
                    "success": True, "recognized": True, "state": "LIVE",
                    "msg": f"Access Granted. Attendance verified for {user_info['name']}",
                    "user": user_info['name'],
                    "emp_id": user_info['employee_id'],
                    "department": user_info['department'],
                    "status": status, "time": time_str,
                    "p_spoof": round(smoothed_prob_spoof, 4),
                    "confidence": confidence, "timings": timings,
                })
            else:
                conn.close()
                _log_scan_event("CHALLENGE_PASSED", user_id=user_id, detail=f"challenge={challenge}",
                                 dedup=False, update_dedup_key=False)
                _log_scan_event("ALREADY_MARKED", user_id=user_id, detail="already checked in today",
                                 dedup=False)
                return jsonify({
                    "success": True, "recognized": True, "state": "LIVE",
                    "msg": f"Welcome back, {user_info['name']}. Already checked in today.",
                    "user": user_info['name'],
                    "emp_id": user_info['employee_id'],
                    "department": user_info['department'],
                    "p_spoof": round(smoothed_prob_spoof, 4),
                    "confidence": confidence, "timings": timings,
                })

        # LIVE, face found, but not recognized (either unregistered, or
        # fewer than min_face_samples enrolled -- see register_face()).
        _log_scan_event("UNRECOGNIZED")
        return jsonify({
            "success": False, "recognized": False, "state": "LIVE",
            "face_found": True, "liveness_metrics": liveness_metrics,
            "p_spoof": round(smoothed_prob_spoof, 4),
            "confidence": confidence, "timings": timings,
            "msg": "Identity Unknown — Please register first",
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

        msg = MIMEText("This is a test email from SmartFace AI. Email notifications are working correctly!")
        msg['From'] = Config.MAIL_DEFAULT_SENDER
        msg['To'] = Config.MAIL_USERNAME
        msg['Subject'] = "SmartFace: Test Email ✓"

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
