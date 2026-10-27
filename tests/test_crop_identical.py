"""
Confirms liveness/face_utils.py (copied into this project) produces exactly
the same detections and crops as the original anti_spoof_attendance project,
on the same real input frames -- not merely "similar," byte-for-byte
identical. This is the regression test requested when FaceDetector/crop_face
were reused verbatim instead of reimplemented.

The original project's face_utils.py is run in an isolated subprocess (its
own cwd -> its own `import config`), because both the original and the copy
define a module literally named `config` with different contents, and
importing both as bare top-level modules in one process would collide.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

MERGED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_SRC = os.path.join(
    os.path.dirname(os.path.dirname(MERGED_ROOT)),
    "project", "anti_spoof_attendance", "src",
)
ORIGINAL_SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(MERGED_ROOT)),
    "project", "anti_spoof_attendance", "data", "celeba_spoof", "test",
)

TEST_IMAGES = [
    os.path.join(ORIGINAL_SAMPLES_DIR, "live", "test_10059_001096.png"),
    os.path.join(ORIGINAL_SAMPLES_DIR, "spoof", "test_10059_001051.png"),
]

_ORIGINAL_SCRIPT = """
import sys, json
sys.path.insert(0, {src!r})
import cv2
import numpy as np
from face_utils import FaceDetector, crop_face

detector = FaceDetector()
frame = cv2.imread({image!r})
detections = detector.detect(frame)
if not detections:
    print(json.dumps({{"face_detected": False}}))
else:
    best = max(detections, key=lambda d: d.width * d.height)
    crop = crop_face(frame, best)
    np.save({out!r}, crop)
    print(json.dumps({{
        "face_detected": True,
        "bbox": [best.x, best.y, best.width, best.height],
        "confidence": best.confidence,
        "crop_shape": list(crop.shape),
    }}))
"""


def _run_original_pipeline(image_path, out_npy_path):
    script = _ORIGINAL_SCRIPT.format(src=ORIGINAL_SRC, image=image_path, out=out_npy_path)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"original pipeline subprocess failed:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("image_path", TEST_IMAGES)
def test_crop_identical_to_original(tmp_path, image_path):
    if not os.path.isfile(image_path):
        pytest.skip(f"original project sample image not found: {image_path}")

    from liveness.face_utils import FaceDetector, crop_face
    import cv2

    out_npy = str(tmp_path / "original_crop.npy")
    original_meta = _run_original_pipeline(image_path, out_npy)
    assert original_meta["face_detected"], f"original pipeline found no face in {image_path}"
    original_crop = np.load(out_npy)

    detector = FaceDetector()
    frame = cv2.imread(image_path)
    detections = detector.detect(frame)
    assert detections, f"merged pipeline found no face in {image_path}"
    best = max(detections, key=lambda d: d.width * d.height)
    merged_crop = crop_face(frame, best)

    assert [best.x, best.y, best.width, best.height] == original_meta["bbox"], (
        "detected bbox differs between merged and original pipeline"
    )
    assert merged_crop.shape == tuple(original_meta["crop_shape"]), (
        "crop shape differs between merged and original pipeline"
    )
    assert np.array_equal(merged_crop, original_crop), (
        "crop pixels differ between merged and original pipeline -- "
        "FaceDetector/crop_face are not actually producing identical output"
    )
