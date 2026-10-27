"""
scripts/measure_pipeline_latency.py
-------------------------------------
Real, measured end-to-end latency of the full merged pipeline (BlazeFace
detect -> crop -> MobileNetV3-Small liveness -> Face Mesh -> ArcFace align
-> embed -> cosine match), run against real images through the actual
running Flask server -- not a synthetic/estimated number.

Usage (server must already be running, e.g. `python app.py`):
    python scripts/measure_pipeline_latency.py [base_url]
"""

import json
import re
import statistics
import sys
import time
import urllib.request
import base64
import os

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000"

# Real CelebA-Spoof images from anti_spoof_attendance, reused as realistic
# face inputs (this script only measures pipeline speed, not accuracy --
# accuracy numbers come from results/summary.json / live camera testing).
IMAGE_PATHS = [
    "/Users/yuvikamalhotra/Documents/project/anti_spoof_attendance/data/celeba_spoof/test/live/test_10059_001096.png",
    "/Users/yuvikamalhotra/Documents/project/anti_spoof_attendance/data/celeba_spoof/test/live/test_10066_000532.png",
    "/Users/yuvikamalhotra/Documents/project/anti_spoof_attendance/data/celeba_spoof/test/spoof/test_10059_001051.png",
]

N_REQUESTS_PER_IMAGE = 15


def b64_data_uri(path):
    with open(path, "rb") as f:
        raw = f.read()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def get_session_and_csrf():
    """Fetches a session cookie + the real CSRF token embedded in the page,
    same as the browser's csrf.js wrapper does from the <meta> tag."""
    req = urllib.request.Request(BASE_URL + "/")
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
        cookie = resp.headers.get("Set-Cookie", "").split(";")[0]
    m = re.search(r'name="csrf-token" content="([^"]+)"', html)
    return cookie, m.group(1) if m else None


def post_recognize(data_uri, cookie, csrf_token):
    req = urllib.request.Request(
        BASE_URL + "/api/recognize_face",
        data=json.dumps({"image": data_uri, "liveness_verified": False}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-CSRFToken": csrf_token,
            **({"Cookie": cookie} if cookie else {}),
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        wall_ms = (time.perf_counter() - t0) * 1000
        body = json.loads(resp.read())
        new_cookie = resp.headers.get("Set-Cookie")
    return wall_ms, body, (new_cookie.split(";")[0] if new_cookie else cookie)


def main():
    all_wall = []
    all_server_total = []
    all_detect = []
    all_classify = []
    all_recognition = []
    states = {}

    cookie, csrf_token = get_session_and_csrf()
    for path in IMAGE_PATHS:
        if not os.path.isfile(path):
            print(f"SKIP (not found): {path}")
            continue
        uri = b64_data_uri(path)
        for i in range(N_REQUESTS_PER_IMAGE):
            wall_ms, body, cookie = post_recognize(uri, cookie, csrf_token)
            all_wall.append(wall_ms)
            t = body.get("timings", {})
            if "total_ms" in t:
                all_server_total.append(t["total_ms"])
            if "detect_ms" in t:
                all_detect.append(t["detect_ms"])
            if "liveness_classify_ms" in t:
                all_classify.append(t["liveness_classify_ms"])
            if "recognition_ms" in t:
                all_recognition.append(t["recognition_ms"])
            state = body.get("state", "?")
            states[state] = states.get(state, 0) + 1

    def stats(name, values):
        if not values:
            print(f"{name}: no data")
            return
        print(f"{name}: n={len(values)}  median={statistics.median(values):.1f}ms  "
              f"mean={statistics.mean(values):.1f}ms  min={min(values):.1f}ms  max={max(values):.1f}ms")

    print(f"\n=== Real pipeline latency, {sum(states.values())} requests against the running server ===")
    print(f"States seen: {states}")
    stats("Full round-trip (wall clock, incl. HTTP + JSON)", all_wall)
    stats("Server-reported total_ms (decode+detect+crop+classify[+recognition])", all_server_total)
    stats("  BlazeFace detect_ms", all_detect)
    stats("  Liveness classify_ms (MobileNetV3-Small)", all_classify)
    stats("  Recognition_ms (Face Mesh + ArcFace, only on LIVE frames)", all_recognition)


if __name__ == "__main__":
    main()
