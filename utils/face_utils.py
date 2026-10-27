"""
SmartFace Recognition Engine
=============================
Face recognition powered by:
  - ArcFace ResNet50 (w600k_r50) via ONNX Runtime
  - MediaPipe Face Mesh 478-point landmarks — alignment + blink/head-turn
    challenge metrics (EAR, head yaw)
  - Vectorized cosine similarity with in-memory cache

Passive spoof detection (screen/texture/depth heuristics) used to live in
this file; it has been removed and replaced by a trained MobileNetV3-Small
classifier that runs as a separate, earlier stage in app.py's
/api/recognize_face (see liveness/predictor.py) -- this module only runs
after that model has already classified the frame as LIVE.
"""

import cv2
import numpy as np
import base64
import os
import json
import time
import traceback
import urllib.request
import zipfile

import onnxruntime as ort
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker as _MPFaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from cryptography.fernet import Fernet, InvalidToken

from config import Config
from database.db import get_db_connection

# Face embeddings are encrypted at rest with Config.FACE_ENCRYPTION_KEY (the
# key was already being generated/persisted -- it just wasn't being used to
# encrypt anything). See _encrypt_embeddings/_decrypt_embeddings below.
_fernet = Fernet(Config.FACE_ENCRYPTION_KEY)


def _encrypt_embeddings(embeddings_list):
    """Serializes a list of embedding lists to encrypted bytes (stored as a
    string in the face_embedding TEXT column)."""
    plaintext = json.dumps(embeddings_list).encode("utf-8")
    return _fernet.encrypt(plaintext).decode("utf-8")


def _decrypt_embeddings(raw):
    """Decrypts a face_embedding column value back to a list of embedding
    lists. Falls back to parsing as legacy plaintext JSON for any row
    written before encryption was added, so existing enrollments aren't
    silently lost by this change."""
    try:
        plaintext = _fernet.decrypt(raw.encode("utf-8"))
        return json.loads(plaintext)
    except (InvalidToken, ValueError):
        return json.loads(raw)

# ============================================================
#  PATHS & CONSTANTS
# ============================================================

MODELS_DIR = os.path.join(Config.BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'w600k_r50.onnx')

# ArcFace standard alignment reference points (112×112 target)
ARCFACE_REF = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)

# MediaPipe eye landmark indices for EAR (blink detection)
# Each: [outer_corner, upper1, upper2, inner_corner, lower2, lower1]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
#  MODEL DOWNLOAD
# ============================================================

def _download_arcface_model():
    """Download ArcFace w600k_r50 from insightface releases (one-time ~300MB)."""
    zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    zip_path = os.path.join(MODELS_DIR, 'buffalo_l.zip')

    print("=" * 60)
    print("  SmartFace AI — First-Time Model Download")
    print("=" * 60)
    print(f"  Downloading ArcFace recognition model (~300 MB)...")
    print(f"  Source: {zip_url}")
    print(f"  This is a ONE-TIME download. Please wait...\n")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, int(downloaded / total_size * 100))
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                bar_len = 40
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(zip_url, zip_path, reporthook=_progress)
        print()  # newline after progress bar

        print("  Extracting recognition model...")
        extracted = False
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('w600k_r50.onnx'):
                    data = zf.read(name)
                    with open(ARCFACE_MODEL_PATH, 'wb') as f:
                        f.write(data)
                    extracted = True
                    break

        # Cleanup zip
        if os.path.exists(zip_path):
            os.remove(zip_path)

        if extracted:
            size_mb = os.path.getsize(ARCFACE_MODEL_PATH) / (1024 * 1024)
            print(f"  ✅ ArcFace model ready! ({size_mb:.0f} MB)")
            print("=" * 60)
            return True
        else:
            print("  ❌ Model file not found in archive!")
            return False

    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        print(f"  Manual fix: Download buffalo_l.zip from the URL above,")
        print(f"  extract w600k_r50.onnx to: {MODELS_DIR}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


# ============================================================
#  MODEL INITIALIZATION
# ============================================================

print("[SmartFace] Initializing recognition engine...")

# --- MediaPipe FaceLandmarker (Tasks API) ---
# The legacy mp.solutions.face_mesh.FaceMesh API used here previously was
# REMOVED in mediapipe>=0.10.30 (confirmed: raises
# "AttributeError: module 'mediapipe' has no attribute 'solutions'" at
# import time). mediapipe is pinned to exactly 0.10.30 in this project
# because newer versions crash on Apple M-series + Metal 4 -- the same
# constraint anti_spoof_attendance already hit and solved by switching to
# the modern mediapipe.tasks.python.vision.FaceLandmarker Task (see that
# project's src/face_utils.py and src/synthetic_mask.py). Same fix applied
# here. FaceLandmarkerOptions has no min_tracking_confidence (that was a
# video-mode-only legacy param); IMAGE running mode re-detects every call,
# matching how this module is actually used (one frame per request, no
# cross-frame tracking state).
_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.join(Config.BASE_DIR, 'models'), 'face_landmarker.task')


def _ensure_face_landmarker_downloaded():
    os.makedirs(os.path.dirname(_FACE_LANDMARKER_MODEL_PATH), exist_ok=True)
    if not os.path.exists(_FACE_LANDMARKER_MODEL_PATH):
        print(f"[SmartFace] Downloading MediaPipe FaceLandmarker model to {_FACE_LANDMARKER_MODEL_PATH} ...")
        urllib.request.urlretrieve(_FACE_LANDMARKER_MODEL_URL, _FACE_LANDMARKER_MODEL_PATH)
        print("[SmartFace] Download complete.")
    return _FACE_LANDMARKER_MODEL_PATH


_face_landmarker = _MPFaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_ensure_face_landmarker_downloaded()),
    running_mode=RunningMode.IMAGE,
    num_faces=2,                       # detect up to 2 to catch multi-face
    min_face_detection_confidence=0.5,
))


class _MeshResult:
    """Adapter so the rest of this file (written against the old
    mp.solutions.face_mesh result shape) doesn't need to change: wraps the
    Tasks API's `.face_landmarks` (list of lists of landmarks with
    .x/.y/.z, same shape as the old API's per-face `.landmark`) as
    `.multi_face_landmarks[i].landmark`."""
    class _Face:
        def __init__(self, landmarks):
            self.landmark = landmarks

    def __init__(self, face_landmarks_list):
        self.multi_face_landmarks = [self._Face(lms) for lms in face_landmarks_list] if face_landmarks_list else []


def _run_face_mesh(img_rgb):
    """Runs the FaceLandmarker and returns an object shaped like the old
    mp.solutions.face_mesh FaceMesh.process() result."""
    mp_image = Image(image_format=ImageFormat.SRGB, data=img_rgb)
    result = _face_landmarker.detect(mp_image)
    return _MeshResult(result.face_landmarks)


print("[SmartFace] ✅ MediaPipe FaceLandmarker ready (478 landmarks + iris, Tasks API)")

# --- ArcFace ONNX Recognition Model ---
_arcface_session = None

if not os.path.exists(ARCFACE_MODEL_PATH):
    _download_arcface_model()

if os.path.exists(ARCFACE_MODEL_PATH):
    try:
        _arcface_session = ort.InferenceSession(
            ARCFACE_MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        print("[SmartFace] ✅ ArcFace recognition model loaded (512-D embeddings)")
    except Exception as e:
        print(f"[SmartFace] ❌ Failed to load ArcFace model: {e}")
else:
    print("[SmartFace] ⚠️  ArcFace model not available — recognition disabled")
    print(f"[SmartFace]    Place w600k_r50.onnx in: {MODELS_DIR}")


# ============================================================
#  EMBEDDING CACHE (In-memory for instant matching)
# ============================================================

_embedding_cache = None  # {user_id: np.array shape (N, 512)}


def invalidate_cache():
    """Clear embedding cache — called after registration or wipe."""
    global _embedding_cache
    _embedding_cache = None
    print("[Cache] Invalidated")


def _ensure_cache():
    """Load all face embeddings into memory for fast cosine matching."""
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache

    conn = get_db_connection()
    users = conn.execute(
        "SELECT id, face_embedding FROM users "
        "WHERE face_registered = 1 AND face_embedding IS NOT NULL"
    ).fetchall()
    conn.close()

    cache = {}
    total_embs = 0
    for user in users:
        raw = user['face_embedding']
        if raw:
            try:
                embs = _decrypt_embeddings(raw)
                if embs:
                    cache[user['id']] = np.array(embs, dtype=np.float32)
                    total_embs += len(embs)
            except Exception:
                continue

    _embedding_cache = cache
    print(f"[Cache] Loaded {total_embs} embeddings for {len(cache)} users")
    return cache


# ============================================================
#  IMAGE UTILITIES
# ============================================================

def data_uri_to_cv2_img(uri):
    """Convert base64 data URI to OpenCV BGR image."""
    if not uri or ',' not in uri:
        return None
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ============================================================
#  FACE ALIGNMENT (5-point → 112×112 ArcFace crop)
# ============================================================

def _get_5_keypoints(landmarks, w, h):
    """Extract 5 alignment keypoints from MediaPipe 478 landmarks."""
    def pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

    try:
        # Use iris centers for most accurate eye positioning
        left_eye = pt(468)
        right_eye = pt(473)
    except (IndexError, AttributeError):
        # Fallback: average of eye corners
        le_outer = pt(33)
        le_inner = pt(133)
        left_eye = (le_outer + le_inner) / 2
        re_outer = pt(362)
        re_inner = pt(263)
        right_eye = (re_outer + re_inner) / 2

    nose = pt(1)
    left_mouth = pt(61)
    right_mouth = pt(291)

    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


def _align_face(img_bgr, landmarks, w, h):
    """Align face to 112×112 using ArcFace reference points."""
    src_pts = _get_5_keypoints(landmarks, w, h)

    # Similarity transform (rotation + scale + translation)
    M, _ = cv2.estimateAffinePartial2D(src_pts, ARCFACE_REF)
    if M is None:
        return None

    aligned = cv2.warpAffine(img_bgr, M, (112, 112))
    return aligned


def _get_embedding(aligned_face):
    """Extract 512-D ArcFace embedding from aligned 112×112 face."""
    if _arcface_session is None:
        return None

    # Preprocess: BGR→RGB, HWC→CHW, normalize to [-1, 1]
    img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.expand_dims(img, axis=0)

    input_name = _arcface_session.get_inputs()[0].name
    embedding = _arcface_session.run(None, {input_name: img})[0][0]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ============================================================
#  LIVENESS DETECTION (MediaPipe Face Mesh)
# ============================================================

def _lm_point(landmark, w, h):
    """Convert normalized landmark to pixel coords."""
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float64)


def calculate_ear(landmarks, w, h):
    """
    Eye Aspect Ratio (EAR) — gold-standard blink detection.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Open eye:   EAR ≈ 0.25–0.35
    Closed eye: EAR < 0.20
    """
    def _eye_ear(indices):
        p = [_lm_point(landmarks[i], w, h) for i in indices]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h_dist = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h_dist + 1e-6)

    return (_eye_ear(LEFT_EYE) + _eye_ear(RIGHT_EYE)) / 2.0


def estimate_head_pose(landmarks, w, h):
    """Estimate head yaw from face landmarks."""
    nose = _lm_point(landmarks[1], w, h)
    left_eye = _lm_point(landmarks[33], w, h)
    right_eye = _lm_point(landmarks[263], w, h)
    eye_center = (left_eye + right_eye) / 2
    face_width = np.linalg.norm(left_eye - right_eye)
    return float((nose[0] - eye_center[0]) / (face_width + 1e-6))


def compute_liveness_metrics(img_bgr, mesh_results):
    """
    Compute the active-liveness-challenge metrics (blink EAR, head yaw) from
    existing Face Mesh results. No passive spoof scoring here anymore -- that
    is now the trained MobileNetV3-Small model's job (see app.py), not this
    function's. This only feeds the blink/head-turn challenge.
    """
    metrics = {
        "eyes_closed": False, "ear": 0.0,
        "head_yaw": 0.0,
        "face_mesh_detected": False,
    }
    if not mesh_results or not mesh_results.multi_face_landmarks:
        return metrics

    lms = mesh_results.multi_face_landmarks[0].landmark
    h, w = img_bgr.shape[:2]
    metrics["face_mesh_detected"] = True

    ear = calculate_ear(lms, w, h)
    metrics["ear"] = round(float(ear), 4)
    metrics["eyes_closed"] = bool(ear < 0.21)
    metrics["head_yaw"] = round(float(estimate_head_pose(lms, w, h)), 3)

    return metrics


# ============================================================
#  FACE REGISTRATION
# ============================================================

def register_face(user_id, base64_img):
    """
    Register a face using ArcFace embedding via MediaPipe alignment.
    Stores 512-D embedding in database as JSON.
    Multiple embeddings per user for robustness.
    """
    if _arcface_session is None:
        print("[Register] ArcFace model not loaded!")
        return False

    img = data_uri_to_cv2_img(base64_img)
    if img is None:
        return False

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = _run_face_mesh(img_rgb)

    if not results.multi_face_landmarks:
        return False

    lms = results.multi_face_landmarks[0].landmark

    # Align face to 112×112
    aligned = _align_face(img, lms, w, h)
    if aligned is None:
        return False

    # Extract 512-D embedding
    embedding = _get_embedding(aligned)
    if embedding is None:
        return False

    # Store in database (encrypted -- see _encrypt_embeddings)
    conn = get_db_connection()
    user = conn.execute("SELECT face_embedding FROM users WHERE id = ?", (user_id,)).fetchone()

    existing = []
    if user and user['face_embedding']:
        try:
            existing = _decrypt_embeddings(user['face_embedding'])
        except Exception:
            existing = []

    existing.append(embedding.tolist())

    MAX_EMBEDDINGS = 15
    if len(existing) > MAX_EMBEDDINGS:
        existing = existing[-MAX_EMBEDDINGS:]

    # face_registered only flips to 1 once at least `min_face_samples`
    # embeddings are stored -- previously this was set after a single
    # sample, making the min_face_samples setting have no real effect.
    from database.db import get_setting
    min_samples = int(get_setting('min_face_samples', '5'))
    is_fully_registered = 1 if len(existing) >= min_samples else 0

    conn.execute(
        "UPDATE users SET face_embedding = ?, face_registered = ? WHERE id = ?",
        (_encrypt_embeddings(existing), is_fully_registered, user_id)
    )
    conn.commit()
    conn.close()

    invalidate_cache()
    print(f"[Register] User {user_id}: {len(existing)} embeddings stored "
          f"(fully_registered={bool(is_fully_registered)}, min_required={min_samples})")
    return True


# ============================================================
#  FACE RECOGNITION + MATCHING
# ============================================================

def _find_best_match(query_embedding):
    """
    Find best matching user via vectorized cosine similarity.
    ~0.1ms for 500 users — uses in-memory cache.
    """
    cache = _ensure_cache()
    if not cache:
        return None, 0.0

    query = np.array(query_embedding, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    best_user = None
    best_sim = 0.0

    for user_id, stored_embs in cache.items():
        norms = np.linalg.norm(stored_embs, axis=1, keepdims=True)
        normalized = stored_embs / (norms + 1e-10)
        sims = normalized @ query_norm
        max_sim = float(np.max(sims))
        if max_sim > best_sim:
            best_sim = max_sim
            best_user = user_id

    return best_user, best_sim


def recognize_face_with_liveness(frame_bgr, match_threshold=0.40):
    """
    Recognition + active-challenge-metrics pipeline. Called by app.py ONLY
    after the trained MobileNetV3-Small liveness model (app.py, a separate
    BlazeFace-based pass) has already classified this frame as LIVE -- this
    function no longer does any passive spoof scoring itself (that engine
    was deleted; see MODEL_CARD-equivalent notes in app.py).

      1. MediaPipe Face Mesh → 478 landmarks (detection + challenge metrics)
      2. ArcFace alignment → 112×112 aligned crop
      3. ArcFace ONNX → 512-D embedding
      4. Vectorized cosine similarity → match
      5. EAR + head yaw → blink / head-turn challenge metrics

    Args:
        frame_bgr: already-decoded BGR frame (np.ndarray) -- decoded once by
            the caller and reused here, rather than re-decoding the same
            base64 image twice per request.
        match_threshold: cosine-similarity threshold for a positive ArcFace
            match. Passed in by the caller (app.py), which reads it from the
            admin-configurable `face_tolerance` setting -- not hardcoded.

    Returns: (user_id, liveness_metrics, confidence, multi_face: bool)
    """
    try:
        if frame_bgr is None:
            return None, {}, 0, False

        h, w = frame_bgr.shape[:2]

        # --- MediaPipe Face Mesh (Tasks API FaceLandmarker) ---
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mesh_results = _run_face_mesh(img_rgb)

        if not mesh_results.multi_face_landmarks:
            return None, {}, 0, False

        if len(mesh_results.multi_face_landmarks) > 1:
            return None, {}, 0, True

        lms = mesh_results.multi_face_landmarks[0].landmark

        # --- Challenge metrics (blink EAR, head yaw) ---
        liveness = compute_liveness_metrics(frame_bgr, mesh_results)

        # --- ArcFace alignment + embedding ---
        if _arcface_session is None:
            return None, liveness, 0, False

        aligned = _align_face(frame_bgr, lms, w, h)
        if aligned is None:
            return None, liveness, 0, False

        embedding = _get_embedding(aligned)
        if embedding is None:
            return None, liveness, 0, False

        # --- Match ---
        user_id, similarity = _find_best_match(embedding)
        confidence = int(max(0, similarity) * 100)

        if user_id is not None and similarity >= match_threshold:
            print(f"[Pipeline] MATCH user={user_id} sim={similarity:.3f}")
            return user_id, liveness, confidence, False

        print(f"[Pipeline] NO MATCH (best_sim={similarity:.3f})")
        return None, liveness, confidence, False

    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline] CRASH: {e}")
        return None, {"error": str(e)}, 0, False
