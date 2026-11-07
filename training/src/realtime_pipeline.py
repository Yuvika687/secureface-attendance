"""
realtime_pipeline.py
---------------------
End-to-end real-time demo:

    Webcam Frame
    -> Stage 1: Mask-aware face detection (MediaPipe, face_utils.py)
    -> Crop face
    -> Stage 2: CNN liveness classifier (liveness_infer.py)
    -> SPOOF -> red box, "SPOOF - REJECTED", frame skipped
    -> LIVE  -> green box, "LIVE", call mark_attendance() placeholder

INTEGRATION POINT FOR YOUR PARTNER:
    The mark_attendance() function below is intentionally a clean, isolated
    placeholder. Replace its body with a call into the existing ArcFace
    recognition + attendance-logging code. It already receives the exact
    face crop (BGR numpy array, same format OpenCV/ArcFace expect) that
    passed the liveness check, so no extra plumbing should be needed:

        def mark_attendance(face_crop_bgr, confidence, frame_timestamp):
            identity = arcface_recognize(face_crop_bgr)
            attendance_db.log(identity, frame_timestamp)

Usage:
    python realtime_pipeline.py                      # uses config.ACTIVE_MODEL_PATH (v2 by default)
    python realtime_pipeline.py --checkpoint ../models/best_liveness.pth --threshold 0.6 --camera 0
    python realtime_pipeline.py --save_samples        # also saves accepted/rejected crops to outputs/predictions
    LIVENESS_MODEL_VERSION=v1 python realtime_pipeline.py   # explicitly run the v1 (raw-frame-trained) model instead
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2

import config
from face_utils import FaceDetector, crop_face
from liveness_infer import LivenessPredictor

COLOR_LIVE = (0, 200, 0)     # green (BGR)
COLOR_SPOOF = (0, 0, 220)    # red (BGR)
COLOR_TEXT_BG = (0, 0, 0)


# ---------------------------------------------------------------------------
# Placeholder for the partner's ArcFace attendance module.
# ---------------------------------------------------------------------------
def mark_attendance(face_crop_bgr, confidence: float, timestamp: str):
    """
    Called ONLY for faces that passed the liveness check (LIVE).

    Replace this body with the real ArcFace recognition + attendance
    logging call. Currently it just prints a placeholder message so the
    pipeline is runnable end-to-end without ArcFace wired in yet.
    """
    print(f"[mark_attendance] LIVE face at {timestamp} "
          f"(liveness_confidence={confidence:.3f}) -> TODO: run ArcFace recognition here")


# ---------------------------------------------------------------------------
# Simple CSV logger for accepted/rejected frames (feature 7 in the spec).
# ---------------------------------------------------------------------------
class AttendanceLogger:
    def __init__(self, log_path: str = config.ATTENDANCE_LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # "not os.path.exists" alone misses a file that exists but is empty
        # (e.g. truncated some other way) -- that case needs a header
        # written too, or every row downstream gets silently misread as a
        # header by csv.DictReader (found via a real empty-file case).
        is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        self._file = open(log_path, "a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(["timestamp", "decision", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])

    def log(self, decision: str, confidence: float, detection):
        self._writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            decision, f"{confidence:.4f}",
            detection.x, detection.y, detection.width, detection.height,
        ])
        self._file.flush()

    def close(self):
        self._file.close()


class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last_time = time.time()

    def tick(self) -> float:
        now = time.time()
        instant_fps = 1.0 / max(1e-6, now - self._last_time)
        self._last_time = now
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        return self.fps


def draw_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, max(0, y - th - 10)), (x + tw + 6, y), COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (x + 3, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run(args):
    detector = FaceDetector()
    predictor = LivenessPredictor(checkpoint_path=args.checkpoint, threshold=args.threshold)
    logger = AttendanceLogger()
    fps_counter = FPSCounter()

    if args.save_samples:
        accepted_dir = os.path.join(config.PREDICTIONS_DIR, "accepted_samples")
        rejected_dir = os.path.join(config.PREDICTIONS_DIR, "rejected_samples")
        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Real-time liveness pipeline running. Press 'q' to quit.")
    print(f"Decision threshold on P(spoof): {args.threshold}")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera. Exiting.")
                break

            detections = detector.detect(frame)

            for det in detections:
                face_crop = crop_face(frame, det)
                if face_crop is None or face_crop.size == 0:
                    continue

                label, confidence = predictor.predict(face_crop)
                timestamp = datetime.now().isoformat(timespec="seconds")

                if label == "SPOOF":
                    color = COLOR_SPOOF
                    text = f"SPOOF - REJECTED ({confidence*100:.1f}%)"
                    logger.log("SPOOF_REJECTED", confidence, det)
                    if args.save_samples and frame_count % args.sample_every == 0:
                        cv2.imwrite(os.path.join(rejected_dir, f"spoof_{frame_count}.jpg"), face_crop)
                else:
                    color = COLOR_LIVE
                    text = f"LIVE ({confidence*100:.1f}%)"
                    logger.log("LIVE_ACCEPTED", confidence, det)
                    if args.save_samples and frame_count % args.sample_every == 0:
                        cv2.imwrite(os.path.join(accepted_dir, f"live_{frame_count}.jpg"), face_crop)
                    # Attendance is only ever triggered for LIVE faces.
                    mark_attendance(face_crop, confidence, timestamp)

                cv2.rectangle(frame, (det.x, det.y), (det.x2, det.y2), color, 2)
                draw_label(frame, text, det.x, det.y, color)

            if not detections:
                cv2.putText(frame, "No face detected", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            fps = fps_counter.tick()
            cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Mask-Aware Liveness / Anti-Spoofing Access Gate", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        logger.close()
        print(f"Session ended. Attendance/decision log saved to: {config.ATTENDANCE_LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Real-time mask-aware liveness/anti-spoofing pipeline")
    parser.add_argument("--checkpoint", default=config.ACTIVE_MODEL_PATH,
                         help="Defaults to config.ACTIVE_MODEL_PATH (v2, face-crop-trained; set "
                              "LIVENESS_MODEL_VERSION=v1 to default to v1 instead, or pass an "
                              "explicit path here to override either).")
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD,
                         help="Decision threshold on P(spoof); higher = stricter on spoof detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--save_samples", action="store_true",
                         help="Save periodic accepted/rejected face crops to outputs/predictions/")
    parser.add_argument("--sample_every", type=int, default=15,
                         help="Save a sample crop every N frames when --save_samples is set")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
