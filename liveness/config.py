"""
liveness/config.py
-------------------
Constants needed by liveness/face_utils.py and liveness/predictor.py, copied
from anti_spoof_attendance/src/config.py (values unchanged, not re-derived).
Kept separate from this project's own config.py (the SmartFace `Config`
class) so the two don't collide -- the liveness module is a self-contained
subpackage reusing the original project's trained artifacts.
"""

import os
import torch

LIVENESS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(LIVENESS_DIR, "weights")
MODELS_DIR = WEIGHTS_DIR  # face_utils.py expects a MODELS_DIR to cache the downloaded .tflite detector into

BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_liveness.pth")
GRAY_ZONE_PATH = os.path.join(WEIGHTS_DIR, "gray_zone_v2.json")

IMAGE_SIZE = 224
BACKBONE = "mobilenet_v3_small"

DECISION_THRESHOLD = 0.5
FACE_DETECTION_CONFIDENCE = 0.5
FACE_CROP_MARGIN = 0.25  # fraction of bbox size added as padding around the face -- must match training

LABEL_LIVE = 0
LABEL_SPOOF = 1
CLASS_NAMES = {LABEL_LIVE: "LIVE", LABEL_SPOOF: "SPOOF"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
