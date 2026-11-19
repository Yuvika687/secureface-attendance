"""CSRF protection on form posts and JSON APIs."""

from tests.helpers import PASSWORD, csrf_token


def test_login_post_without_token_is_rejected(flask_app, make_user):
    s = make_user("employee")
    r = flask_app.test_client().post("/login", data={"email": s["email"], "password": PASSWORD, "role": "student"})
    assert r.status_code == 400
    assert "CSRF" in r.get_data(as_text=True)


def test_json_api_without_token_is_rejected(admin_client, db):
    before = db("SELECT COUNT(*) AS n FROM admin_audit_log")[0]["n"]
    r = admin_client.post("/admin/wipe_model", json={"confirm": "WIPE"})   # no X-CSRFToken header
    assert r.status_code == 400
    assert db("SELECT COUNT(*) AS n FROM admin_audit_log")[0]["n"] == before


def test_token_from_another_session_is_rejected(flask_app, admin_client):
    other = flask_app.test_client()
    foreign = csrf_token(other, "/login")
    r = admin_client.post("/admin/scanner_devices/approve", data={"csrf_token": foreign})
    assert r.status_code == 400


def test_valid_token_passes(admin_client):
    r = admin_client.post("/api/admin/test_mode/new_session", headers={"X-CSRFToken": admin_client.csrf})
    assert r.status_code == 200 and r.get_json()["success"] is True
