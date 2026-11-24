"""Scan-log de-dup for an empty camera, and CSV export honouring the Attendance filters."""
import base64

import cv2
import numpy as np


def _blank_frame():
    ok, buf = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def test_empty_camera_logs_no_face_once(scanner_client, db):
    for _ in range(5):
        r = scanner_client.post("/api/recognize_face", json={"image": _blank_frame()},
                                headers={"X-CSRFToken": scanner_client.csrf}).get_json()
        assert r["state"] == "NO_FACE"
    assert db("SELECT COUNT(*) AS n FROM scan_events WHERE stage = 'NO_FACE'")[0]["n"] == 1


def test_csv_export_uses_status_and_department_filters(admin_client, make_user, db):
    a, b = make_user("employee"), make_user("employee")
    db("UPDATE users SET department = 'CSE' WHERE id = ?", (a["id"],))
    db("UPDATE users SET department = 'ECE' WHERE id = ?", (b["id"],))
    db("INSERT INTO attendance (user_id, date, time, status) VALUES (?, '2026-11-20', '09:10:00', 'Late')", (a["id"],))
    db("INSERT INTO attendance (user_id, date, time, status) VALUES (?, '2026-11-20', '08:50:00', 'Present')", (b["id"],))

    def rows(qs):
        return admin_client.get("/export_csv" + qs).get_data(as_text=True).strip().splitlines()[1:]

    assert len(rows("")) == 2
    late = rows("?status=Late")
    assert len(late) == 1 and a["employee_id"] in late[0]
    ece = rows("?department=ECE")
    assert len(ece) == 1 and b["employee_id"] in ece[0]
