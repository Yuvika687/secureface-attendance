"""
Shared fixtures for the app tests.

The real database (database/smartface.db) is never touched: before app.py is
imported, Config.DATABASE_URI is pointed at a temporary file, app.py's own
startup creates the schema there, and that file is used as a template. Every
test then gets its own fresh copy of it (Config.DATABASE_URI is read at
connection time by database/db.py), so tests can't affect each other -- the
wipe_model test included.

Sample face images are the same real CelebA-Spoof test images
tests/test_crop_identical.py uses, read from the sibling
anti_spoof_attendance project (read-only).
"""

import os
import shutil
import tempfile

import pytest

os.environ.pop("DATABASE_URL", None)        # force the SQLite path
os.environ.pop("ADMIN_EMAIL", None)         # no env-created admin in the template DB
os.environ.pop("ADMIN_PASSWORD", None)

from config import Config  # noqa: E402

_TMP_ROOT = tempfile.mkdtemp(prefix="secureface_tests_")
_TEMPLATE_DB = os.path.join(_TMP_ROOT, "template.db")
Config.DATABASE_URI = _TEMPLATE_DB

import app as app_module  # noqa: E402  (creates the schema in _TEMPLATE_DB)
from werkzeug.security import generate_password_hash  # noqa: E402
from database.db import get_db_connection  # noqa: E402
from tests.helpers import PASSWORD, approve_scanner, csrf_token, login  # noqa: E402




@pytest.fixture(scope="session")
def flask_app():
    app_module.app.config["TESTING"] = True
    yield app_module.app
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, flask_app):
    db_path = str(tmp_path / "test.db")
    shutil.copy(_TEMPLATE_DB, db_path)
    Config.DATABASE_URI = db_path
    from utils.face_utils import invalidate_cache
    invalidate_cache()
    yield db_path
    invalidate_cache()


@pytest.fixture
def db():
    def run(sql, params=()):
        conn = get_db_connection()
        rows = conn.execute(sql, params).fetchall()
        conn.commit()
        conn.close()
        return rows
    return run


@pytest.fixture
def make_user():
    counter = {"n": 0}

    def make(role="employee", name=None):
        counter["n"] += 1
        n = counter["n"]
        email = f"{role}{n}@example.com"
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO users (employee_id, name, email, password_hash, role) VALUES (?, ?, ?, ?, ?)",
            (f"T{role[:1].upper()}{n:03d}", name or f"Test {role} {n}", email, generate_password_hash(PASSWORD), role),
        )
        conn.commit()
        uid = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()["id"]
        conn.close()
        return {"id": uid, "email": email, "employee_id": f"T{role[:1].upper()}{n:03d}"}
    return make




@pytest.fixture
def admin_client(flask_app, make_user):
    admin = make_user("admin")
    c = flask_app.test_client()
    r = login(c, admin["email"], "admin")
    assert r.status_code == 302 and r.headers["Location"].endswith("/admin")
    c.csrf = csrf_token(c, "/settings")
    c.user = admin
    return c



@pytest.fixture
def scanner_client(flask_app, admin_client):
    """A separate browser (no login) holding an approved scanner cookie."""
    token = approve_scanner(admin_client)
    c = flask_app.test_client()
    c.set_cookie("scanner_device", token)
    c.csrf = csrf_token(c, "/")
    return c


@pytest.fixture
def eye_script(monkeypatch):
    """Scripts the eye state the server sees, for blink tests.

    Static sample photos can't blink, so this wraps the REAL
    recognize_face_with_liveness() (Face Mesh + ArcFace still run on the real
    image) and only replaces liveness_metrics["eyes_closed"] -- the value the
    server derives from Face Mesh's eye-aspect-ratio -- with the next scripted
    step. Append ("open"|"closed", recognized: bool) tuples to the returned
    list; recognized=False makes that frame an unrecognized LIVE frame."""
    real = app_module.recognize_face_with_liveness
    script = []

    def scripted(frame, match_threshold=0.40):
        user_id, metrics, confidence, multi_face = real(frame, match_threshold=match_threshold)
        if script:
            eyes, recognized = script.pop(0)
            metrics = dict(metrics, eyes_closed=(eyes == "closed"), face_mesh_detected=True)
            if not recognized:
                user_id = None
        return user_id, metrics, confidence, multi_face

    monkeypatch.setattr(app_module, "recognize_face_with_liveness", scripted)
    return script
