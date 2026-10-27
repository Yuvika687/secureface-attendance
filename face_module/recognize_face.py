"""
SmartFace Face Module (Legacy Wrapper)
=======================================
This module previously used the dlib-based `face_recognition` library.
It has been deprecated in favor of the LBPH + MediaPipe engine in utils/face_utils.py.

The current recognition pipeline uses:
  - OpenCV DNN (ResNet-10 SSD) for face detection
  - LBPH for face matching (scales to 500+ employees)
  - MediaPipe FaceMesh for 3D liveness detection
  - 10-layer anti-spoofing engine
  - Fernet AES-256 encryption for face data

For face recognition, import from utils.face_utils:
    from utils.face_utils import register_face, recognize_face_with_liveness

For model retraining, use:
    python retrain.py
"""

# Legacy compatibility — redirect imports to the new engine
from utils.face_utils import register_face, recognize_face_with_liveness
