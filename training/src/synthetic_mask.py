"""
synthetic_mask.py
-------------------
TASK I: overlays a synthetic surgical-style mask (nose-to-chin) on a face
image, using real facial landmarks from MediaPipe's FaceLandmarker Task
(pinned mediapipe==0.10.30, same package already used for BlazeFace
detection -- no new dependency).

Landmark indices used (standard 468/478-point MediaPipe Face Mesh
topology, stable across versions):
    4        - nose tip (top anchor)
    215, 435 - wide cheek points, roughly jaw-corner height, left/right
    172, 365 - lower jaw, left/right (narrowing toward the chin)
    152      - chin (bottom of jaw)

These were NOT trusted from memory alone: a first attempt using a
different index set (168/147/376/172/397) rendered a mask that covered
the eyes, visibly wrong. The indices above were chosen by rendering all
478 landmarks on a real face image, visually locating good anchors, and
re-rendering the mask until it actually looked like a nose-to-chin mask
(see the module's __main__ for the same render-and-look workflow).

Usage:
    from synthetic_mask import FaceLandmarkerWrapper, apply_synthetic_mask
    landmarker = FaceLandmarkerWrapper()
    landmarks = landmarker.detect(image_bgr)   # None if synthesis should be
                                                # recorded as a failure
    if landmarks is not None:
        masked_bgr = apply_synthetic_mask(image_bgr, landmarks, color="white")
"""

import os
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker as _MPFaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from config import MODELS_DIR

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_FILENAME = "face_landmarker.task"

# Mask colors: (BGR base color, BGR shading color for a light gradient).
MASK_COLORS = {
    "white": ((235, 235, 235), (200, 200, 200)),
    "light_blue": ((235, 206, 173), (200, 170, 140)),  # BGR: light blue/cyan surgical-mask tone
    "black": ((35, 35, 35), (10, 10, 10)),
}

# Landmark indices anchoring the mask polygon (see module docstring).
_NOSE_TIP = 4
_WIDE_CHEEK_LEFT = 215
_WIDE_CHEEK_RIGHT = 435
_LOWER_JAW_LEFT = 172
_LOWER_JAW_RIGHT = 365
_CHIN = 152


def _ensure_model_downloaded() -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, _MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe FaceLandmarker model to {model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("Download complete.")
    return model_path


class FaceLandmarkerWrapper:
    """Thin wrapper around the MediaPipe Tasks FaceLandmarker (IMAGE mode)."""

    def __init__(self, model_path: Optional[str] = None, min_confidence: float = 0.5):
        model_path = model_path or _ensure_model_downloaded()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
        )
        self._landmarker = _MPFaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Returns a list of 478 (x, y) pixel-coordinate landmarks, or None
        if no face/landmarks were found (a mask-synthesis failure, to be
        recorded as such -- not a Stage-1 BlazeFace detection miss)."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        return [(int(lm.x * w), int(lm.y * h)) for lm in face]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _mask_polygon(landmarks: List[Tuple[int, int]]) -> np.ndarray:
    """Builds a rounded, nose-to-chin mask polygon from the anchor
    landmarks: narrow at the nose tip, widening to the cheeks, narrowing
    again through the jaw down to the chin -- a surgical-mask-like
    silhouette, not a literal anatomical jaw trace."""
    nose = np.array(landmarks[_NOSE_TIP], dtype=np.float32)
    cheek_l = np.array(landmarks[_WIDE_CHEEK_LEFT], dtype=np.float32)
    cheek_r = np.array(landmarks[_WIDE_CHEEK_RIGHT], dtype=np.float32)
    jaw_l = np.array(landmarks[_LOWER_JAW_LEFT], dtype=np.float32)
    jaw_r = np.array(landmarks[_LOWER_JAW_RIGHT], dtype=np.float32)
    chin = np.array(landmarks[_CHIN], dtype=np.float32)

    # Push the top of the mask slightly up from the nose tip (toward the
    # nose bridge) and the chin slightly further down, for a fuller,
    # more mask-like silhouette than a polygon drawn exactly through the
    # raw landmark points.
    nose_top = nose + (nose - chin) * 0.18
    chin_bottom = chin + (chin - nose) * 0.08

    points = [nose_top, cheek_l, jaw_l, chin_bottom, jaw_r, cheek_r]
    return np.array(points, dtype=np.int32)


def apply_synthetic_mask(image_bgr: np.ndarray, landmarks: List[Tuple[int, int]],
                          color: str = "white", blur_ksize: int = 15,
                          alpha: float = 0.92) -> np.ndarray:
    """Overlays a synthetic surgical-style mask (nose-to-chin) on
    image_bgr, using real landmark positions. Soft edges via a blurred
    alpha mask; light shading via a vertical gradient between the two
    tones in MASK_COLORS.

    Returns a new BGR image (input is not modified in place).
    """
    if color not in MASK_COLORS:
        raise ValueError(f"color must be one of {list(MASK_COLORS)}, got {color!r}")
    base_color, shade_color = MASK_COLORS[color]

    h, w = image_bgr.shape[:2]
    polygon = _mask_polygon(landmarks)

    # Binary mask of the polygon region, then blur its edges for a soft
    # transition instead of a hard cutout.
    alpha_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(alpha_mask, cv2.convexHull(polygon), 255)
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    alpha_mask = cv2.GaussianBlur(alpha_mask, (ksize, ksize), 0).astype(np.float32) / 255.0

    # Light vertical shading: base_color at the top of the mask region,
    # fading toward shade_color at the bottom (chin), for a touch of
    # fabric-like depth rather than a flat color fill.
    ys, xs = np.where(alpha_mask > 0.01)
    if len(ys) == 0:
        return image_bgr.copy()  # polygon degenerate -- nothing to draw, caller can check via landmarks

    y_top, y_bottom = ys.min(), ys.max()
    # Vertical color gradient spanning only the mask's own extent (not the
    # whole image), base_color at the top fading to shade_color at the chin.
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        col = np.interp(np.arange(h), [y_top, y_bottom], [base_color[c], shade_color[c]])
        gradient[:, :, c] = col.reshape(-1, 1)

    alpha_3ch = (alpha_mask * alpha)[:, :, None]
    result = image_bgr.astype(np.float32) * (1 - alpha_3ch) + gradient * alpha_3ch
    return result.astype(np.uint8)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python synthetic_mask.py <image_path> [color]")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    color = sys.argv[2] if len(sys.argv) > 2 else "white"

    with FaceLandmarkerWrapper() as landmarker:
        lm = landmarker.detect(img)

    if lm is None:
        print("No landmarks found (mask-synthesis failure).")
        sys.exit(1)

    masked = apply_synthetic_mask(img, lm, color=color)
    out_path = "synthetic_mask_debug.jpg"
    cv2.imwrite(out_path, masked)
    print(f"Saved masked image to {out_path}")
