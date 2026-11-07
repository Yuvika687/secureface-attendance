"""
Central configuration for the Mask-Aware Liveness / Anti-Spoofing module.

Keeping all shared constants in one place makes it easy for a teammate
(e.g. the partner integrating ArcFace attendance) to see every tunable
value without digging through each script.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CELEBA_SPOOF_DIR = os.path.join(DATA_DIR, "celeba_spoof")
CASIA_FASD_DIR = os.path.join(DATA_DIR, "casia_fasd")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_liveness.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "last_liveness.pth")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "liveness.onnx")

# ---------------------------------------------------------------------------
# Model version selection (TASK G/H)
#
# "v1" = BEST_MODEL_PATH, trained on raw full CelebA-Spoof frames (the
#        original, mismatched setup -- never saw a face crop in training).
# "v2" = trained on face crops built by build_face_cropped_dataset.py,
#        matching what the live deployment pipeline actually feeds the
#        model. v2 is the default for realtime_pipeline.py and
#        webapp/backend/app.py; v1 stays selectable (for comparison only)
#        via the LIVENESS_MODEL_VERSION env var, never silently.
# ---------------------------------------------------------------------------
MODEL_VERSION = os.environ.get("LIVENESS_MODEL_VERSION", "v2")

V1_MODEL_PATH = BEST_MODEL_PATH
V2_MODEL_PATH = os.path.join(MODELS_DIR, "v2_cropped", "best_liveness.pth")
MODEL_PATHS_BY_VERSION = {"v1": V1_MODEL_PATH, "v2": V2_MODEL_PATH}

# Gray-zone band JSON produced by evaluate_gray_zone.py for each version
# (band selected on validation only -- see that script). v1's predates
# TASK G and used the raw-frame model; v2's is the one actually used live.
GRAY_ZONE_PATH_BY_VERSION = {
    "v1": os.path.join(PROJECT_ROOT, "results", "gray_zone.json"),
    "v2": os.path.join(PROJECT_ROOT, "results", "gray_zone_v2.json"),
}


def get_active_model_path(version: str = None) -> str:
    version = version or MODEL_VERSION
    if version not in MODEL_PATHS_BY_VERSION:
        raise ValueError(f"Unknown model version {version!r}, expected one of {list(MODEL_PATHS_BY_VERSION)}")
    return MODEL_PATHS_BY_VERSION[version]


def get_gray_zone_path(version: str = None) -> str:
    version = version or MODEL_VERSION
    if version not in GRAY_ZONE_PATH_BY_VERSION:
        raise ValueError(f"Unknown model version {version!r}, expected one of {list(GRAY_ZONE_PATH_BY_VERSION)}")
    return GRAY_ZONE_PATH_BY_VERSION[version]


ACTIVE_MODEL_PATH = get_active_model_path()  # resolved once at import time, from LIVENESS_MODEL_VERSION

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
ATTENDANCE_LOG_PATH = os.path.join(LOGS_DIR, "attendance_log.csv")

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
# CelebA-Spoof convention: 0 = live, 1 = spoof (we follow the same mapping
# so the raw dataset folders can be used with minimal renaming).
LABEL_LIVE = 0
LABEL_SPOOF = 1
CLASS_NAMES = {LABEL_LIVE: "LIVE", LABEL_SPOOF: "SPOOF"}

# ---------------------------------------------------------------------------
# Model / training defaults
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224                 # face crop input size (also supports 128)
BACKBONE = "mobilenet_v3_small"  # or "efficientnet_b0"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
RANDOM_SEED = 42

# Decision threshold on P(spoof). Anything >= threshold is classified SPOOF.
# Tunable at inference time without retraining.
DECISION_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))

# ---------------------------------------------------------------------------
# Face detection (MediaPipe)
# ---------------------------------------------------------------------------
FACE_DETECTION_CONFIDENCE = 0.5
FACE_CROP_MARGIN = 0.25   # fraction of bbox size added as padding around the face
