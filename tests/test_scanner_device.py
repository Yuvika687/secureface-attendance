"""Scanner device approval: only admin-approved devices can run the scanner."""

import hashlib

from tests.helpers import approve_scanner, csrf_token, login, sample_image

NOT_APPROVED = "This device is not an approved scanner."


def test_unapproved_device_sees_message_and_api_refuses(flask_app):
    c = flask_app.test_client()
    r = c.get("/")
    assert r.status_code == 403
    assert NOT_APPROVED in r.get_data(as_text=True)
    token = csrf_token(c, "/")
    r = c.post("/api/recognize_face", json={"image": sample_image("test_10059_001096.png")},
               headers={"X-CSRFToken": token})
    assert r.status_code == 403
    assert r.get_json()["state"] == "DEVICE_NOT_APPROVED"


def test_approve_then_revoke(flask_app, admin_client, db):
    token = approve_scanner(admin_client)
    rows = db("SELECT id, token_hash FROM scanner_devices WHERE revoked_at IS NULL")
    assert len(rows) == 1
    assert rows[0]["token_hash"] == hashlib.sha256(token.encode()).hexdigest()   # only the hash is stored
    assert rows[0]["token_hash"] != token

    kiosk = flask_app.test_client()
    kiosk.set_cookie("scanner_device", token)
    assert kiosk.get("/").status_code == 200

    r = admin_client.post(f"/admin/scanner_devices/{rows[0]['id']}/revoke", data={"csrf_token": admin_client.csrf})
    assert r.status_code == 302
    r = kiosk.get("/")
    assert r.status_code == 403 and NOT_APPROVED in r.get_data(as_text=True)
    actions = [r["action"] for r in db("SELECT action FROM admin_audit_log ORDER BY id")]
    assert actions == ["SCANNER_APPROVED", "SCANNER_REVOKED"]


def test_forged_cookie_is_not_approved(flask_app, admin_client):
    approve_scanner(admin_client)
    c = flask_app.test_client()
    c.set_cookie("scanner_device", "not-a-real-token")
    assert c.get("/").status_code == 403


def test_student_cannot_approve_a_device(flask_app, make_user, db):
    s = make_user("employee")
    c = flask_app.test_client()
    login(c, s["email"], "student")
    token = csrf_token(c, "/employee")
    r = c.post("/admin/scanner_devices/approve", data={"csrf_token": token})
    assert r.status_code == 302 and "login" in r.headers["Location"]
    assert db("SELECT COUNT(*) AS n FROM scanner_devices")[0]["n"] == 0
