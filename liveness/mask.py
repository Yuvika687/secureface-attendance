"""Binary face-mask classifier. Model provenance and license: docs/MASK_MODEL.md."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

_model = ort.InferenceSession(str(Path(__file__).parent / "weights" / "mask_detector.onnx"),
                              providers=["CPUExecutionProvider"])

def mask_probability(face):
    """Model's probability (0..1) that the face is wearing a mask."""
    if face.size == 0:
        return 0.0
    rgb = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2RGB)
    batch = (rgb.astype(np.float32) / 127.5 - 1)[None]
    masked, uncovered = _model.run(None, {_model.get_inputs()[0].name: batch})[0].reshape(2)
    return float(masked)


def wearing_mask(face):
    return mask_probability(face) > 0.5
