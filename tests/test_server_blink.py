"""The blink is verified by the server from its own Face Mesh eye readings;
the browser's "liveness_verified" flag is ignored."""

import base64

import cv2
import numpy as np
import pytest

import app as app_module
from tests.helpers import csrf_token, login, sample_image

FACE_A = "test_10059_001096.png"


def _blank_frame():
    ok, buf = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def registered(flask_app, make_user):
    s = make_user("employee")
    c = flask_app.test_client()
    login(c, s["email"], "student")
    token = csrf_token(c, "/employee")
    for _ in range(5):
        assert c.post("/api/register_face", json={"image": sample_image(FACE_A)},
                      headers={"X-CSRFToken": token}).status_code == 200
    return s


def scan(scanner, image=None, **extra):
    body = {"image": image or sample_image(FACE_A), **extra}
    return scanner.post("/api/recognize_face", json=body, headers={"X-CSRFToken": scanner.csrf}).get_json()


def attendance_count(db, user_id):
    return db("SELECT COUNT(*) AS n FROM attendance WHERE user_id = ?", (user_id,))[0]["n"]


def test_fake_browser_blink_flag_does_not_mark_attendance(registered, scanner_client, db):
    """No scripting: real models, a real photo whose eyes never close. The
    browser claims a blink on every frame; attendance must not be marked."""
    responses = [scan(scanner_client, liveness_verified=True) for _ in range(10)]

    assert all(r["recognized"] is True and r["state"] == "LIVE" for r in responses)   # the face is matched...
    # ...and the server's own Face Mesh reading says the eyes are open on every frame
    assert all(r["liveness_metrics"]["face_mesh_detected"] for r in responses)
    assert not any(r["liveness_metrics"]["eyes_closed"] for r in responses)
    assert not any(r["success"] for r in responses)                                    # ...but no blink, so:
    assert attendance_count(db, registered["id"]) == 0


def test_open_closed_open_marks_attendance(registered, scanner_client, db, eye_script):
    eye_script.extend([("open", True), ("closed", True), ("open", True)])
    r1, r2, r3 = (scan(scanner_client) for _ in range(3))
    assert not r1["success"] and not r2["success"]
    assert r3["success"] is True and r3["msg"] == "Attendance marked"
    assert attendance_count(db, registered["id"]) == 1


def test_browser_flag_false_does_not_block_a_real_blink(registered, scanner_client, db, eye_script):
    eye_script.extend([("open", True), ("closed", True), ("open", True)])
    results = [scan(scanner_client, liveness_verified=False) for _ in range(3)]
    assert results[-1]["success"] is True
    assert attendance_count(db, registered["id"]) == 1


def test_closed_then_open_without_open_first_is_not_enough(registered, scanner_client, db, eye_script):
    eye_script.extend([("closed", True), ("open", True)])
    assert not any(scan(scanner_client)["success"] for _ in range(2))
    assert attendance_count(db, registered["id"]) == 0
    # the open frame started the sequence; a real closed -> open now completes it
    eye_script.extend([("closed", True), ("open", True)])
    assert [scan(scanner_client)["success"] for _ in range(2)] == [False, True]


def test_blink_must_finish_within_the_window(registered, scanner_client, db, eye_script, monkeypatch):
    real_time = app_module.time.time
    clock = {"t": real_time()}          # start at real now so CSRF/session timestamps stay valid
    monkeypatch.setattr(app_module.time, "time", lambda: clock["t"])
    eye_script.extend([("open", True), ("closed", True), ("open", True)])
    scan(scanner_client)
    scan(scanner_client)
    clock["t"] += app_module.CHALLENGE_TIMEOUT_SECONDS + 1
    late = scan(scanner_client)
    assert late["success"] is False
    assert attendance_count(db, registered["id"]) == 0


def test_face_leaving_resets_the_blink(registered, scanner_client, db, eye_script):
    eye_script.extend([("open", True), ("closed", True)])
    scan(scanner_client)
    scan(scanner_client)
    assert scan(scanner_client, image=_blank_frame())["state"] == "NO_FACE"   # face gone: session reset
    eye_script.append(("open", True))
    assert scan(scanner_client)["success"] is False
    assert attendance_count(db, registered["id"]) == 0


def test_closed_eye_frame_that_fails_recognition_still_counts(registered, scanner_client, db, eye_script):
    """Closed eyes can make ArcFace miss for one frame; the same tracked face
    may still record 'closed', but only a recognized open frame completes it."""
    eye_script.extend([("open", True), ("closed", False), ("open", True)])
    r1, r2, r3 = (scan(scanner_client) for _ in range(3))
    assert r2["recognized"] is False
    assert r3["success"] is True
    assert attendance_count(db, registered["id"]) == 1


def test_unrecognized_open_frame_cannot_complete_the_blink(registered, scanner_client, db, eye_script):
    eye_script.extend([("open", True), ("closed", True), ("open", False)])
    results = [scan(scanner_client) for _ in range(3)]
    assert not any(r["success"] for r in results)
    assert attendance_count(db, registered["id"]) == 0
