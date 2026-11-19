"""/admin/wipe_model requires the typed confirmation word and is audit-logged."""

import pytest

from tests.helpers import csrf_token, login


@pytest.fixture
def seeded(db, make_user):
    s = make_user("employee")
    db("UPDATE users SET face_registered = 1, face_embedding = 'x' WHERE id = ?", (s["id"],))
    db("INSERT INTO attendance (user_id, date, time, status, method) VALUES (?, '2026-11-01', '09:00:00', 'Present', 'Test')",
       (s["id"],))
    return s


def _state(db):
    return (db("SELECT COUNT(*) AS n FROM users WHERE face_registered = 1")[0]["n"],
            db("SELECT COUNT(*) AS n FROM attendance")[0]["n"])


@pytest.mark.parametrize("body", [{}, {"confirm": ""}, {"confirm": "wipe"}, {"confirm": "yes"}])
def test_wipe_without_correct_word_does_nothing(admin_client, seeded, db, body):
    r = admin_client.post("/admin/wipe_model", json=body, headers={"X-CSRFToken": admin_client.csrf})
    assert r.status_code == 400
    assert _state(db) == (1, 1)
    assert db("SELECT COUNT(*) AS n FROM admin_audit_log")[0]["n"] == 0


def test_wipe_with_word_wipes_and_logs_who(admin_client, seeded, db):
    r = admin_client.post("/admin/wipe_model", json={"confirm": "WIPE"}, headers={"X-CSRFToken": admin_client.csrf})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert _state(db) == (0, 0)
    log = db("SELECT admin_user_id, admin_email, action, detail FROM admin_audit_log")
    assert len(log) == 1
    assert log[0]["admin_user_id"] == admin_client.user["id"]
    assert log[0]["admin_email"] == admin_client.user["email"]
    assert log[0]["action"] == "WIPE_MODEL"
    assert "attendance_rows_deleted=1" in log[0]["detail"]


def test_student_cannot_wipe(flask_app, seeded, db, make_user):
    s = make_user("employee")
    c = flask_app.test_client()
    login(c, s["email"], "student")
    r = c.post("/admin/wipe_model", json={"confirm": "WIPE"}, headers={"X-CSRFToken": csrf_token(c, "/employee")})
    assert r.get_json()["success"] is False
    assert _state(db) == (1, 1)
