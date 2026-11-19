"""Plain helper functions for the tests (no fixtures, no side effects)."""

import base64
import os
import re

import pytest

MERGED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(os.path.dirname(MERGED_ROOT), "anti_spoof_attendance", "data", "celeba_spoof", "test")
PASSWORD = "Test#Pass123"


def sample_image(name, kind="live"):
    path = os.path.join(SAMPLES_DIR, kind, name)
    if not os.path.isfile(path):
        pytest.skip(f"sample image not found: {path}")
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def csrf_token(client, path="/login"):
    html = client.get(path).get_data(as_text=True)
    m = re.search(r'name="csrf_token" value="([^"]+)"', html) or re.search(r'name="csrf-token" content="([^"]+)"', html)
    assert m, f"no CSRF token on {path}"
    return m.group(1)


def login(client, email, role="student", password=PASSWORD):
    token = csrf_token(client, "/login")
    return client.post("/login", data={"email": email, "password": password, "role": role, "csrf_token": token})


def approve_scanner(admin_client):
    r = admin_client.post("/admin/scanner_devices/approve", data={"csrf_token": admin_client.csrf, "label": "Test kiosk"})
    assert r.status_code == 302
    cookie = admin_client.get_cookie("scanner_device")
    assert cookie is not None
    return cookie.value
