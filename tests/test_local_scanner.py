"""Exercise the scanner's decisions with deterministic model outputs on a clean database."""
import base64
from liveness.face_utils import Detection

import cv2
import numpy as np
from types import SimpleNamespace

import app
from utils import face_utils
from tests.helpers import csrf_token, login


def test_mask_cancels_blink_and_repeated_checkin_keeps_one_row(scanner_client, db, make_user, monkeypatch):
    student = make_user("employee")
    face = Detection(0, 0, 64, 64, 1.0)
    monkeypatch.setattr(app._liveness_detector, "detect", lambda frame: [face])
    scores = iter([1.0] + [0.0] * (18 + app.MASK_STREAK_FRAMES - 1))
    monkeypatch.setattr(app._liveness_predictor, "predict_prob", lambda crop: next(scores))
    masked = False
    monkeypatch.setattr(app, "mask_probability", lambda crop: 0.99 if masked else 0.6)
    closed = False
    yaw = 0.0
    monkeypatch.setattr(app, "_pick_turn", lambda: "left")

    def recognize(frame, match_threshold):
        return student["id"], {"face_mesh_detected": True, "eyes_closed": closed, "head_yaw": yaw}, 99, False

    monkeypatch.setattr(app, "recognize_face_with_liveness", recognize)
    _, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    data = {"image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}

    def scan():
        response = scanner_client.post("/api/recognize_face", json=data,
                                       headers={"X-CSRFToken": scanner_client.csrf})
        assert response.status_code == 200
        return response.get_json()

    assert scan()["state"] == "SPOOF"
    for _ in range(4):
        assert scan()["state"] == "UNCERTAIN"
    assert scan()["recognized"]
    closed = True
    assert not scan()["success"]
    masked = True
    for _ in range(app.MASK_STREAK_FRAMES - 1):   # one or two confident frames do not block
        assert scan()["state"] != "MASK"
    assert scan()["state"] == "MASK"
    masked = False
    for expected in ("Attendance marked", "Already marked today"):
        yaw = 0.0
        assert not scan()["success"]             # new random challenge: looking straight
        yaw = 0.2
        assert not scan()["success"]             # turned left as asked
        for closed in (False, True):
            assert not scan()["success"]
        closed = False
        assert scan()["msg"] == expected
    assert db("SELECT COUNT(*) AS n FROM attendance")[0]["n"] == 1


def test_enrollment_resumes_from_stored_samples(flask_app, make_user, monkeypatch):
    student = make_user()
    client = flask_app.test_client()
    login(client, student["email"])
    headers = {"X-CSRFToken": csrf_token(client, "/employee")}
    invalid = client.post("/api/register_face", json={"image": "data:image/jpeg;base64,?"},
                          headers=headers)
    assert invalid.status_code == 400
    assert not invalid.get_json()["success"]
    mesh = SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=[])])
    monkeypatch.setattr(face_utils, "_run_face_mesh", lambda frame: mesh)
    monkeypatch.setattr(face_utils, "_align_face", lambda image, *args: image)
    monkeypatch.setattr(face_utils, "_get_embedding", lambda image: np.ones(512, dtype=np.float32))
    _, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    data = {"image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}
    for count in range(1, 6):
        if count == 3:
            client = flask_app.test_client()
            login(client, student["email"])
            headers = {"X-CSRFToken": csrf_token(client, "/employee")}
        result = client.post("/api/register_face", json=data, headers=headers).get_json()
        assert result["saved"] == count
        assert result["required"] == 5
        assert result["complete"] == (count == 5)


def test_calendar_shows_selected_month_and_own_status(flask_app, make_user, db):
    student = make_user()
    other = make_user()
    for user, status in ((student, "Late"), (other, "Leave")):
        db("INSERT INTO attendance (user_id,date,time,status) VALUES (?, '2024-02-29','09:00',?)",
           (user["id"], status))
    client = flask_app.test_client()
    login(client, student["email"])
    page = client.get("/employee?month=2024-02").get_data(as_text=True)
    assert 'aria-label="2024-02-29, Late"' in page
    assert 'aria-label="2024-02-29, Leave"' not in page
    assert client.get("/employee?month=invalid").status_code == 200

