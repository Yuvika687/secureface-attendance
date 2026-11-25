"""The random active challenge (second check after the liveness model): after LIVE +
recognized, one random head turn ("Turn your head left/right"), then one natural blink,
by the same tracked face within CHALLENGE_TIMEOUT_SECONDS. Decided only by the server
from its own Face Mesh readings; anything the browser claims is ignored."""

import base64
import os

import cv2
import numpy as np
import pytest

import app as app_module
from tests.helpers import SAMPLES_DIR, TURNED, challenge_steps, csrf_token, login, sample_image
from utils.face_utils import _run_face_mesh, compute_liveness_metrics

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


def test_fake_browser_flags_do_not_mark_attendance(registered, scanner_client, db):
    """No scripting: real models on a real photo that never turns or blinks. The browser
    claims a blink, a passed challenge and a turned head on every frame."""
    fake = {"liveness_verified": True, "challenge_passed": True, "challenge": "done", "head_yaw": 0.5,
            "eyes_closed": True, "turn_done": True, "blink_done": True}
    responses = [scan(scanner_client, **fake) for _ in range(10)]
    assert all(r["recognized"] is True and r["state"] == "LIVE" for r in responses)
    assert all(r["challenge_pending"] and r["instruction"].startswith("Turn your head") for r in responses)
    assert not any(r["success"] for r in responses)
    assert attendance_count(db, registered["id"]) == 0


@pytest.mark.parametrize("direction", ["left", "right"])
def test_turn_then_blink_marks_attendance(direction, registered, scanner_client, db, eye_script, force_turn):
    force_turn(direction)
    eye_script.extend(challenge_steps(direction))
    results = [scan(scanner_client) for _ in range(5)]
    assert results[0]["instruction"] == f"Turn your head {direction}"
    assert results[1]["instruction"] == "Now blink once"
    assert not any(r["success"] for r in results[:4])
    assert results[4]["success"] is True and results[4]["msg"] == "Attendance marked"
    assert attendance_count(db, registered["id"]) == 1


def test_blink_alone_is_not_enough(registered, scanner_client, db, eye_script, force_turn):
    eye_script.extend([("open", True, 0.0), ("closed", True, 0.0), ("open", True, 0.0)] * 2)
    assert not any(scan(scanner_client)["success"] for _ in range(6))
    assert attendance_count(db, registered["id"]) == 0


def test_blink_before_the_turn_does_not_count(registered, scanner_client, db, eye_script, force_turn):
    y = TURNED["left"]
    eye_script.extend([("open", True, 0.0), ("closed", True, 0.0), ("open", True, 0.0),   # blink too early
                       ("open", True, y), ("open", True, y)])
    assert not any(scan(scanner_client)["success"] for _ in range(5))
    eye_script.extend([("closed", True, y), ("open", True, y)])                         # blink after the turn
    assert [scan(scanner_client)["success"] for _ in range(2)] == [False, True]


def test_turn_must_start_from_looking_straight(registered, scanner_client, db, eye_script, force_turn):
    y = TURNED["left"]
    eye_script.extend([("open", True, y), ("open", True, y), ("closed", True, y), ("open", True, y)])
    assert not any(scan(scanner_client)["success"] for _ in range(4))
    assert attendance_count(db, registered["id"]) == 0


def test_wrong_turn_starts_a_new_random_challenge(registered, scanner_client, db, eye_script, force_turn):
    eye_script.extend([("open", True, 0.0), ("open", True, TURNED["right"])])   # asked left, turned right
    scan(scanner_client)
    r = scan(scanner_client)
    assert r["success"] is False and r["instruction"].startswith("Turn your head")
    assert db("SELECT COUNT(*) AS n FROM scan_events WHERE detail LIKE 'turned the wrong way%'")[0]["n"] == 1
    eye_script.extend(challenge_steps("left"))
    assert [scan(scanner_client)["success"] for _ in range(5)][-1] is True


def test_challenge_must_finish_within_the_time_limit(registered, scanner_client, db, eye_script, force_turn, monkeypatch):
    real_time = app_module.time.time
    clock = {"t": real_time()}          # start at real now so CSRF/session timestamps stay valid
    monkeypatch.setattr(app_module.time, "time", lambda: clock["t"])
    steps = challenge_steps("left")
    eye_script.extend(steps)
    for _ in range(4):
        scan(scanner_client)
    clock["t"] += app_module.CHALLENGE_TIMEOUT_SECONDS + 1
    late = scan(scanner_client)
    assert late["success"] is False and late["instruction"] == "Turn your head left"   # a new challenge
    assert attendance_count(db, registered["id"]) == 0


def test_face_leaving_resets_the_challenge(registered, scanner_client, db, eye_script, force_turn):
    steps = challenge_steps("left")
    eye_script.extend(steps[:4])
    for _ in range(4):
        scan(scanner_client)
    assert scan(scanner_client, image=_blank_frame())["state"] == "NO_FACE"
    eye_script.append(steps[4])
    assert scan(scanner_client)["success"] is False
    assert attendance_count(db, registered["id"]) == 0


def test_unrecognized_frames_of_the_same_face_still_count(registered, scanner_client, db, eye_script, force_turn):
    """A turned head or closed eyes can make recognition miss one frame of the same tracked face."""
    y = TURNED["left"]
    eye_script.extend([("open", True, 0.0), ("open", False, y), ("open", True, y), ("closed", False, y), ("open", True, y)])
    results = [scan(scanner_client) for _ in range(5)]
    assert results[1]["recognized"] is False and results[1]["challenge_pending"] is True
    assert results[4]["success"] is True


def test_unrecognized_frame_cannot_finish_the_challenge(registered, scanner_client, db, eye_script, force_turn):
    steps = challenge_steps("left")
    steps[4] = ("open", False, steps[4][2])
    eye_script.extend(steps)
    assert not any(scan(scanner_client)["success"] for _ in range(5))
    assert attendance_count(db, registered["id"]) == 0


def test_direction_is_random():
    assert {app_module._pick_turn() for _ in range(200)} == {"left", "right"}


def test_tilting_a_flat_photo_is_not_a_head_turn():
    """Real Face Mesh on a real photo: rotating the flat picture (perspective warp)
    hardly moves the yaw, so it never reaches the turn threshold."""
    path = os.path.join(SAMPLES_DIR, "live", FACE_A)
    if not os.path.isfile(path):
        pytest.skip("sample image not found")
    img = cv2.imread(path)
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    for k in (0.15, 0.3, 0.45):
        for side in ("right", "left"):
            if side == "right":
                dst = np.float32([[0, 0], [w * (1 - k / 2), h * k / 4], [w * (1 - k / 2), h * (1 - k / 4)], [0, h]])
            else:
                dst = np.float32([[w * k / 2, h * k / 4], [w, 0], [w, h], [w * k / 2, h * (1 - k / 4)]])
            warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(90, 90, 90))
            m = compute_liveness_metrics(warped, _run_face_mesh(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))
            assert m["face_mesh_detected"]
            assert abs(m["head_yaw"]) < app_module.HEAD_TURN_MIN_YAW, (k, side, m["head_yaw"])
