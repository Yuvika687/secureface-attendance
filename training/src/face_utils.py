"""
face_utils.py
-------------
Stage 1 of the pipeline: mask-aware face detection using MediaPipe, plus
cropping / preprocessing helpers shared by training-data prep, evaluation,
and the real-time demo.

Why MediaPipe Face Detection is "mask-aware":
The underlying model here is Google's "BlazeFace short-range" detector,
which localizes faces primarily from the upper-face region (eyes,
eyebrows, nose bridge, forehead) rather than requiring the mouth/jawline
to be visible. In practice this means it keeps detecting faces even when
the lower half is covered by a surgical/cloth mask, which is exactly the
behaviour this project needs.

Implementation note: recent MediaPipe releases (>=0.10.30 / 1.0.x) removed
the old `mediapipe.solutions.face_detection` API in favor of the newer
Tasks API (`mediapipe.tasks.python.vision.FaceDetector`), which loads a
small `.tflite` model file instead of being bundled as compiled code. This
module uses the Tasks API and auto-downloads the official BlazeFace
short-range model into `models/` the first time it's needed, so no manual
setup step is required.

This module has NO dependency on the liveness model or ArcFace, so it can
be imported independently by whichever teammate needs it.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector as _MPFaceDetector,
    FaceDetectorOptions,
    RunningMode,
)

from config import FACE_DETECTION_CONFIDENCE, FACE_CROP_MARGIN, IMAGE_SIZE, MODELS_DIR

# Official Google-hosted BlazeFace short-range detector used by the
# MediaPipe Face Detector Task (see https://ai.google.dev/edge/mediapipe/
# solutions/vision/face_detector). Auto-downloaded once and cached locally.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_FILENAME = "blaze_face_short_range.tflite"


def _ensure_model_downloaded() -> str:
    """Downloads the BlazeFace .tflite model into models/ if not already present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, _MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe face detector model to {model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("Download complete.")
    return model_path


@dataclass
class Detection:
    """A single detected face, in absolute pixel coordinates of the source frame."""
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


class FaceDetector:
    """Thin wrapper around the MediaPipe Tasks Face Detector (IMAGE running mode)."""

    def __init__(self, min_confidence: float = FACE_DETECTION_CONFIDENCE, model_path: Optional[str] = None):
        model_path = model_path or _ensure_model_downloaded()

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = _MPFaceDetector.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Runs detection on a BGR frame (as returned by cv2.VideoCapture) and
        returns a list of Detection boxes in pixel coordinates, clipped to the
        frame bounds."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

        result = self._detector.detect(mp_image)

        detections = []
        for det in result.detections:
            bbox = det.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            bw, bh = bbox.width, bbox.height

            # Clip to frame bounds (the model can return slightly out-of-frame
            # boxes near the edges).
            x_clipped = max(0, x)
            y_clipped = max(0, y)
            bw_clipped = min(bw - (x_clipped - x), w - x_clipped)
            bh_clipped = min(bh - (y_clipped - y), h - y_clipped)

            if bw_clipped <= 0 or bh_clipped <= 0:
                continue

            score = det.categories[0].score if det.categories else 0.0
            detections.append(Detection(x_clipped, y_clipped, bw_clipped, bh_clipped, score))

        return detections

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def crop_face(frame_bgr: np.ndarray, detection: Detection, margin: float = FACE_CROP_MARGIN) -> Optional[np.ndarray]:
    """
    Crops the face region out of the frame with extra margin around the
    detected bounding box. The margin matters for the liveness model: too
    tight a crop hides spoof artifacts near the face border (screen bezels,
    photo paper edges, hand holding a printed photo).

    Returns a BGR numpy array, or None if the resulting crop is invalid.
    """
    h, w = frame_bgr.shape[:2]

    pad_x = int(detection.width * margin)
    pad_y = int(detection.height * margin)

    x1 = max(0, detection.x - pad_x)
    y1 = max(0, detection.y - pad_y)
    x2 = min(w, detection.x2 + pad_x)
    y2 = min(h, detection.y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame_bgr[y1:y2, x1:x2].copy()


def bgr_crop_to_pil(face_bgr: np.ndarray):
    """Converts an OpenCV BGR crop to a PIL RGB image (for torchvision transforms)."""
    from PIL import Image as PILImage
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(rgb)


def resize_for_display(face_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Simple resize helper, e.g. for saving sample accepted/rejected crops."""
    return cv2.resize(face_bgr, (size, size))


if __name__ == "__main__":
    # Quick webcam smoke test for Stage 1 only (no liveness model involved).
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    print("Running face_utils smoke test. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for det in detector.detect(frame):
            cv2.rectangle(frame, (det.x, det.y), (det.x2, det.y2), (255, 200, 0), 2)
            cv2.putText(frame, f"{det.confidence:.2f}", (det.x, max(0, det.y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("face_utils smoke test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
