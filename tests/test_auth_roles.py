"""Login (single page with Student/Admin toggle) and role-based access."""

import pytest

from tests.helpers import login

ADMIN_PAGES = [
    "/admin", "/manage_employees", "/attendance_history", "/analytics", "/settings",
    "/export_csv", "/admin/scan_log", "/admin/spoof_log", "/admin/results",
    "/admin/test_mode", "/admin/pipeline_view",
]
ADMIN_JSON_GET = ["/api/system_status", "/api/search_employees?q=a", "/api/attendance_feed",
                  "/api/admin/test_mode/metrics"]


def test_student_login_with_student_toggle(flask_app, make_user):
    s = make_user("employee")
    c = flask_app.test_client()
    r = login(c, s["email"], "student")
    assert r.status_code == 302 and r.headers["Location"].endswith("/employee")
    assert c.get("/employee").status_code == 200


def test_admin_login_with_admin_toggle(flask_app, make_user):
    a = make_user("admin")
    c = flask_app.test_client()
    r = login(c, a["email"], "admin")
    assert r.status_code == 302 and r.headers["Location"].endswith("/admin")


@pytest.mark.parametrize("account_role,toggle", [("employee", "admin"), ("admin", "student")])
def test_wrong_role_toggle_is_invalid_credentials(flask_app, make_user, account_role, toggle):
    u = make_user(account_role)
    c = flask_app.test_client()
    r = login(c, u["email"], toggle)
    assert r.status_code == 200
    assert "Invalid credentials" in r.get_data(as_text=True)
    # not logged in as anything
    assert c.get("/admin").status_code == 302
    assert c.get("/employee").status_code == 302


def test_wrong_password_and_unknown_email_same_message(flask_app, make_user):
    s = make_user("employee")
    c = flask_app.test_client()
    assert "Invalid credentials" in login(c, s["email"], "student", password="wrong").get_data(as_text=True)
    assert "Invalid credentials" in login(c, "nobody@example.com", "student").get_data(as_text=True)


def test_admin_login_url_redirects_to_single_login_page(flask_app):
    r = flask_app.test_client().get("/admin_login")
    assert r.status_code == 302 and r.headers["Location"].endswith("/login?role=admin")


@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_student_cannot_open_admin_pages(flask_app, make_user, path):
    s = make_user("employee")
    c = flask_app.test_client()
    login(c, s["email"], "student")
    r = c.get(path)
    assert r.status_code in (302, 401)
    if r.status_code == 302:
        assert "/admin" not in r.headers["Location"] or "login" in r.headers["Location"]


@pytest.mark.parametrize("path", ADMIN_JSON_GET)
def test_student_cannot_call_admin_apis(flask_app, make_user, path):
    s = make_user("employee")
    c = flask_app.test_client()
    login(c, s["email"], "student")
    assert c.get(path).status_code == 401


@pytest.mark.parametrize("path", ADMIN_PAGES + ["/employee", "/profile"])
def test_anonymous_cannot_open_private_pages(flask_app, path):
    r = flask_app.test_client().get(path)
    assert r.status_code in (302, 401)


def test_public_pages_open_without_login(flask_app):
    c = flask_app.test_client()
    for path in ("/login", "/register", "/how_it_works"):
        assert c.get(path).status_code == 200
