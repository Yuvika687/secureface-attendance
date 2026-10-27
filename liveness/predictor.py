"""
liveness/predictor.py
----------------------
Adapted from anti_spoof_attendance/src/liveness_infer.py. Same logic
(LivenessPredictor, classify_three_way, load_gray_zone_band); adjusted to
load from this subpackage's own weights/ directory (a copy of the real
trained v2 checkpoint and the real gray_zone_v2.json band -- no retraining,
no invented numbers) instead of the original project's models/ and
results/ layout.
"""

import json
import os
from typing import Tuple

import numpy as np
import torch

from . import config
from .transform import get_eval_transform
from .face_utils import bgr_crop_to_pil
from .model import build_model


def load_gray_zone_band(path: str = config.GRAY_ZONE_PATH) -> Tuple[float, float]:
    """Loads the real (band_low, band_high) UNCERTAIN band computed by
    evaluate_gray_zone.py in the original project (results/gray_zone_v2.json,
    copied verbatim into liveness/weights/). Never hardcoded."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Gray-zone results not found at {path} -- copy "
            f"anti_spoof_attendance/results/gray_zone_v2.json here."
        )
    with open(path) as f:
        data = json.load(f)
    return data["band_low"], data["band_high"]


def classify_three_way(prob_spoof: float, band_low: float, band_high: float,
                        threshold: float = config.DECISION_THRESHOLD) -> str:
    """Three-way decision: "LIVE", "SPOOF", or "UNCERTAIN" -- identical logic
    to the original project's classify_three_way()."""
    if band_low <= prob_spoof <= band_high:
        return "UNCERTAIN"
    return "SPOOF" if prob_spoof >= threshold else "LIVE"


class LivenessPredictor:
    """Loads the real trained checkpoint once; exposes single-crop inference."""

    def __init__(self, checkpoint_path: str = config.BEST_MODEL_PATH,
                 device: torch.device = config.DEVICE,
                 threshold: float = config.DECISION_THRESHOLD):
        self.device = device
        self.threshold = threshold

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.backbone = checkpoint.get("backbone", config.BACKBONE)
        self.image_size = checkpoint.get("image_size", config.IMAGE_SIZE)

        self.model = build_model(backbone=self.backbone, pretrained=False).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = get_eval_transform(self.image_size)

    @torch.no_grad()
    def predict_prob(self, face_crop_bgr: np.ndarray) -> float:
        """Raw, unsmoothed P(spoof) for a single crop -- caller does its own
        temporal smoothing (per-session, see app.py)."""
        pil_image = bgr_crop_to_pil(face_crop_bgr)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        logit = self.model(tensor)
        return torch.sigmoid(logit).item()
