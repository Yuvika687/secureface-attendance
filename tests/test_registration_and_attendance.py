"""Face registration owner checks, attendance marking / already-marked, and
Test Mode never marking attendance. Uses real sample images and the real
models (BlazeFace, liveness model, Face Mesh, ArcFace)."""

import pytest

from tests.helpers import csrf_token, login, sample_image

OWNER_MSG = "Only the account owner should be in front of the camera."
FACE_A = "test_10059_001096.png"          # CelebA-Spoof subject 10059 (live)
FACE_B1 = "test_10066_000532.png"         # subject 10066, photo 1
FACE_B2 = "test_10066_001907.png"         # subject 10066, photo 2 (same person)
FACE_C = "test_10152_000030.png"          # subject 10152 (different person)
SPOOF_A = "test_10059_001051.png"         # spoof image of subject 10059


def student_client(flask_app, make_user):
    s = make_user("employee")
    c = flask_app.test_client()
    login(c, s["email"], "student")
    c.csrf = csrf_token(c, "/employee")
    c.user = s
    return c


def register(c, image_name):
    return c.post("/api/register_face", json={"image": sample_image(image_name)}, headers={"X-CSRFToken": c.csrf})


def register_fully(c, image_name, times=5):
    for _ in range(times):
        assert register(c, image_name).status_code == 200


# ---------------- registration owner checks ----------------

def test_registration_rejects_another_registered_persons_face(flask_app, make_user, db):
    a = student_client(flask_app, make_user)
    register_fully(a, FACE_A)
    assert db("SELECT face_registered FROM users WHERE id = ?", (a.user["id"],))[0]["face_registered"] == 1

    b = student_client(flask_app, make_user)
    r = register(b, FACE_A)
    assert r.status_code == 400
    assert r.get_json()["msg"] == OWNER_MSG
    assert db("SELECT face_embedding FROM users WHERE id = ?", (b.user["id"],))[0]["face_embedding"] is None


def test_registration_rejects_sample_not_matching_own_samples(flask_app, make_user):
    b = student_client(flask_app, make_user)
    assert register(b, FACE_B1).status_code == 200           # first sample: nothing to compare yet
    assert register(b, FACE_B2).status_code == 200           # same person, other photo
    r = register(b, FACE_C)                                    # different person
    assert r.status_code == 400 and r.get_json()["msg"] == OWNER_MSG


# ---------------- attendance ----------------

def scan_until_done(scanner, image_name, tries=8):
    res = None
    for _ in range(tries):
        res = scanner.post("/api/recognize_face", json={"image": sample_image(image_name), "liveness_verified": True},
                           headers={"X-CSRFToken": scanner.csrf}).get_json()
        if res.get("success") or res.get("face_found"):
            break
    return res


def test_attendance_marked_then_already_marked(flask_app, make_user, scanner_client, db):
    a = student_client(flask_app, make_user)
    register_fully(a, FACE_A)

    first = scan_until_done(scanner_client, FACE_A)
    assert first["success"] is True
    assert first["msg"] == "Attendance marked"
    assert first["emp_id"] == a.user["employee_id"]
    assert db("SELECT COUNT(*) AS n FROM attendance WHERE user_id = ?", (a.user["id"],))[0]["n"] == 1

    second = scan_until_done(scanner_client, FACE_A)
    assert second["success"] is True
    assert second["msg"] == "Already marked today"
    assert db("SELECT COUNT(*) AS n FROM attendance WHERE user_id = ?", (a.user["id"],))[0]["n"] == 1


def test_unregistered_face_marks_nothing(flask_app, make_user, scanner_client, db):
    a = student_client(flask_app, make_user)
    register_fully(a, FACE_A)
    res = scan_until_done(scanner_client, FACE_C)
    assert res["success"] is False
    assert res["msg"].startswith("Face not recognized")
    assert db("SELECT COUNT(*) AS n FROM attendance")[0]["n"] == 0


def test_spoof_image_marks_nothing(flask_app, make_user, scanner_client, db):
    a = student_client(flask_app, make_user)
    register_fully(a, FACE_A)
    res = scanner_client.post("/api/recognize_face", json={"image": sample_image(SPOOF_A, "spoof"), "liveness_verified": True},
                              headers={"X-CSRFToken": scanner_client.csrf}).get_json()
    assert res["state"] == "SPOOF"
    assert db("SELECT COUNT(*) AS n FROM attendance")[0]["n"] == 0


# ---------------- Test Mode ----------------

def test_test_mode_never_marks_attendance(flask_app, make_user, admin_client, db):
    a = student_client(flask_app, make_user)
    register_fully(a, FACE_A)                       # the face IS registered...
    scan_rows_before = db("SELECT COUNT(*) AS n FROM scan_events")[0]["n"]

    decisions = []
    for category, image, kind in [("real_face", FACE_A, "live")] * 4 + [("phone_photo", SPOOF_A, "spoof")] * 2:
        r = admin_client.post("/api/admin/test_scan", json={"image": sample_image(image, kind), "category": category},
                              headers={"X-CSRFToken": admin_client.csrf})
        assert r.status_code == 200
        body = r.get_json()
        assert "user" not in body and "recognized" not in body
        decisions.append(body["decision"])

    assert db("SELECT COUNT(*) AS n FROM attendance")[0]["n"] == 0        # ...but nothing is marked
    assert db("SELECT COUNT(*) AS n FROM scan_events")[0]["n"] == scan_rows_before
    assert db("SELECT COUNT(*) AS n FROM test_mode_results")[0]["n"] == 6
    m = body["metrics"]
    counted = {r["key"]: r["n"] for r in m["rows"]}
    assert counted["real_face"] == decisions[:4].count("LIVE") + decisions[:4].count("SPOOF") + decisions[:4].count("UNCERTAIN")
    assert counted["phone_photo"] == 2


@pytest.mark.parametrize("path", ["/api/admin/test_scan", "/api/admin/pipeline_frame"])
def test_student_cannot_use_admin_camera_tools(flask_app, make_user, path):
    s = student_client(flask_app, make_user)
    r = s.post(path, json={"image": sample_image(FACE_A), "category": "real_face"}, headers={"X-CSRFToken": s.csrf})
    assert r.status_code == 401
