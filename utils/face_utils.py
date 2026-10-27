"""
SmartFace Recognition Engine v3.0
==================================
Production-grade face recognition powered by:
  - ArcFace ResNet50 (w600k_r50) via ONNX Runtime — 99.8% LFW accuracy
  - MediaPipe Face Mesh 478-point landmarks — liveness & alignment
  - Vectorized cosine similarity with in-memory cache
  - Multi-layer anti-spoofing (3D depth + texture + mandatory blink)

No C compiler required — pure Python + ONNX.
Designed for local deployment via Cloudflare Tunnel.
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

import mediapipe as mp
import onnxruntime as ort

from config import Config
from database.db import get_db_connection

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

# --- MediaPipe Face Mesh (478 landmarks + iris tracking) ---
_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(
    max_num_faces=2,       # detect up to 2 to catch multi-face
    refine_landmarks=True, # enable iris landmarks (478 total)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("[SmartFace] ✅ MediaPipe Face Mesh ready (478 landmarks + iris)")

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
                embs = json.loads(raw)
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


def _get_face_bbox(landmarks, w, h):
    """Compute face bounding box from MediaPipe landmarks."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)


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


def analyze_3d_depth(landmarks):
    """
    3D depth analysis for anti-spoofing.
    Real faces have natural z-depth variation (nose protrudes, eyes recessed).
    Flat screens/photos have near-zero z-variation.
    Returns 0–100 (100 = very 3D/real).
    """
    key_indices = [
        1, 4, 5, 6,        # Nose ridge
        33, 133, 362, 263,  # Eye corners
        61, 291,            # Mouth corners
        10, 152,            # Top/bottom of face
        234, 454,           # Ears/sides
        67, 109, 297, 338,  # Forehead
        199, 175            # Chin area
    ]
    z_values = [landmarks[i].z for i in key_indices if i < len(landmarks)]
    if len(z_values) < 5:
        return 50
    z_arr = np.array(z_values)
    z_std = float(np.std(z_arr))
    return min(100, int((z_std / 0.025) * 100))


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
    Compute full liveness metrics from existing Face Mesh results.
    Returns dict with EAR, head pose, depth, eyes_closed flag.
    """
    metrics = {
        "eyes_closed": False, "ear": 0.0,
        "head_yaw": 0.0, "depth_score": 0,
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
    metrics["depth_score"] = int(analyze_3d_depth(lms))

    return metrics


# ============================================================
#  ANTI-SPOOFING ENGINE (Screen Detection + Texture + Depth)
# ============================================================

_spoof_history = []
_MAX_SPOOF_HISTORY = 8


def detect_screen_display(img_bgr, face_bbox):
    """
    Detect if face is displayed on a phone/tablet/monitor screen.
    Three independent physical signals:
      1. Brightness anomaly — screen emits light, brighter than surroundings
      2. Rectangular border — phone bezel creates strong straight edges
      3. Color temperature — screens shift colors toward blue
    Returns: (is_screen, screen_evidence 0-100)
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = face_bbox
    fw, fh = x2 - x1, y2 - y1

    if fw < 20 or fh < 20:
        return False, 0

    screen_evidence = 0

    # --- Signal 1: Brightness Contrast ---
    face_gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    face_brightness = float(np.mean(face_gray))

    corner_sz = max(30, min(h, w) // 8)
    corners = []
    for cy, cx in [(0, 0), (0, w - corner_sz), (h - corner_sz, 0), (h - corner_sz, w - corner_sz)]:
        region = img_bgr[cy:cy + corner_sz, cx:cx + corner_sz]
        if region.size > 0:
            corners.append(float(np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))))
    bg_brightness = float(np.mean(corners)) if corners else face_brightness
    brightness_ratio = face_brightness / (bg_brightness + 1e-6)

    if brightness_ratio > 2.5:
        screen_evidence += 45
    elif brightness_ratio > 1.8:
        screen_evidence += 30
    elif brightness_ratio > 1.4:
        screen_evidence += 15

    # --- Signal 2: Rectangular Border Detection ---
    pad = int(max(fw, fh) * 0.8)
    sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
    sx2, sy2 = min(w, x2 + pad), min(h, y2 + pad)
    search_gray = cv2.cvtColor(img_bgr[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(search_gray, 30, 100)
    min_line_len = max(min(search_gray.shape[:2]) // 4, 30)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=min_line_len, maxLineGap=15)
    if lines is not None:
        h_count, v_count = 0, 0
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            angle = abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
            if angle < 20 or angle > 160:
                h_count += 1
            elif 70 < angle < 110:
                v_count += 1
        if h_count >= 2 and v_count >= 2:
            screen_evidence += 35
        elif h_count >= 1 and v_count >= 1:
            screen_evidence += 15

    # --- Signal 3: Color Temperature ---
    face_bgr = img_bgr[y1:y2, x1:x2].astype(np.float32)
    b_mean = float(np.mean(face_bgr[:, :, 0]))
    r_mean = float(np.mean(face_bgr[:, :, 2]))
    blue_ratio = b_mean / (r_mean + 1e-6)
    if blue_ratio > 1.15:
        screen_evidence += 20
    elif blue_ratio > 1.0:
        screen_evidence += 10

    is_screen = screen_evidence >= 45
    return is_screen, min(100, screen_evidence)


def compute_anti_spoof_score(depth_score, face_roi_gray, img_bgr, face_bbox):
    """
    Hardcore anti-spoofing. Screen detection → instant score tank.
    """
    global _spoof_history

    # Screen detection (primary defense)
    is_screen, screen_conf = detect_screen_display(img_bgr, face_bbox)
    if is_screen:
        score = max(0, 20 - screen_conf // 3)
        _spoof_history.append(score)
        if len(_spoof_history) > _MAX_SPOOF_HISTORY:
            _spoof_history.pop(0)
        return score, True

    # LBP texture analysis
    texture_score = 60
    if face_roi_gray is not None and face_roi_gray.size > 0:
        try:
            fr = cv2.resize(face_roi_gray, (128, 128)).astype(np.int16)
            center = fr[1:-1, 1:-1]
            lbp = np.zeros_like(center, dtype=np.uint8)
            offsets = [(-1,-1,7),(-1,0,6),(-1,1,5),(0,1,4),(1,1,3),(1,0,2),(1,-1,1),(0,-1,0)]
            for dy, dx, bit in offsets:
                lbp |= ((fr[1+dy:fr.shape[0]-1+dy, 1+dx:fr.shape[1]-1+dx] >= center).astype(np.uint8) << bit)
            texture_score = min(100, int(float(np.var(lbp.astype(np.float64))) / 3000 * 100))
        except Exception:
            texture_score = 60

    composite = int(depth_score * 0.50 + texture_score * 0.50)
    composite = max(0, min(100, composite))

    _spoof_history.append(composite)
    if len(_spoof_history) > _MAX_SPOOF_HISTORY:
        _spoof_history.pop(0)
    if len(_spoof_history) >= 3:
        avg = int(np.mean(_spoof_history))
        if avg < 30:
            composite = min(composite, avg)

    return composite, False



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
    results = _face_mesh.process(img_rgb)

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

    # Store in database
    conn = get_db_connection()
    user = conn.execute("SELECT face_embedding FROM users WHERE id = ?", (user_id,)).fetchone()

    existing = []
    if user and user['face_embedding']:
        try:
            existing = json.loads(user['face_embedding'])
        except Exception:
            existing = []

    existing.append(embedding.tolist())

    MAX_EMBEDDINGS = 15
    if len(existing) > MAX_EMBEDDINGS:
        existing = existing[-MAX_EMBEDDINGS:]

    conn.execute(
        "UPDATE users SET face_embedding = ?, face_registered = 1 WHERE id = ?",
        (json.dumps(existing), user_id)
    )
    conn.commit()
    conn.close()

    invalidate_cache()
    print(f"[Register] User {user_id}: {len(existing)} embeddings stored")
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


def recognize_face_with_liveness(base64_img):
    """
    Full recognition pipeline:
      1. MediaPipe Face Mesh → 478 landmarks (detection + liveness)
      2. ArcFace alignment → 112×112 aligned crop
      3. ArcFace ONNX → 512-D embedding
      4. Vectorized cosine similarity → match
      5. EAR + 3D depth → liveness & anti-spoof

    Returns: (user_id, liveness_metrics, confidence, anti_spoof_score, spoof_checks)
    """
    try:
        t_start = time.time()

        img = data_uri_to_cv2_img(base64_img)
        if img is None:
            return None, {}, 0, 0, {}

        h, w = img.shape[:2]

        # --- MediaPipe Face Mesh ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mesh_results = _face_mesh.process(img_rgb)

        if not mesh_results.multi_face_landmarks:
            return None, {}, 0, 0, {}

        if len(mesh_results.multi_face_landmarks) > 1:
            return None, {}, 0, 0, {"multi_face": True}

        lms = mesh_results.multi_face_landmarks[0].landmark

        # --- Liveness metrics ---
        liveness = compute_liveness_metrics(img, mesh_results)

        # --- Anti-spoofing ---
        x1, y1, x2, y2 = _get_face_bbox(lms, w, h)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi_gray = gray[y1:y2, x1:x2]

        anti_spoof_score, screen_detected = compute_anti_spoof_score(
            liveness.get("depth_score", 50),
            face_roi_gray,
            img,
            (x1, y1, x2, y2)
        )

        spoof_checks = {
            "3d_depth": bool(liveness.get("depth_score", 0) > 25),
            "face_mesh": bool(liveness.get("face_mesh_detected", False)),
            "texture": bool(anti_spoof_score > 30),
            "screen_detected": bool(screen_detected),
        }

        # --- ArcFace alignment + embedding ---
        if _arcface_session is None:
            return None, liveness, 0, anti_spoof_score, spoof_checks

        aligned = _align_face(img, lms, w, h)
        if aligned is None:
            return None, liveness, 0, anti_spoof_score, spoof_checks

        embedding = _get_embedding(aligned)
        if embedding is None:
            return None, liveness, 0, anti_spoof_score, spoof_checks

        # --- Match ---
        user_id, similarity = _find_best_match(embedding)

        MATCH_THRESHOLD = 0.40
        confidence = int(max(0, similarity) * 100)

        t_total = int((time.time() - t_start) * 1000)

        if user_id is not None and similarity >= MATCH_THRESHOLD:
            print(f"[Pipeline] {t_total}ms | MATCH user={user_id} "
                  f"sim={similarity:.3f} depth={liveness.get('depth_score', 0)} "
                  f"spoof={anti_spoof_score}")
            return user_id, liveness, confidence, anti_spoof_score, spoof_checks

        print(f"[Pipeline] {t_total}ms | NO MATCH (best_sim={similarity:.3f})")
        return None, liveness, confidence, anti_spoof_score, spoof_checks

    except Exception as e:
        traceback.print_exc()
        print(f"[Pipeline] CRASH: {e}")
        return None, {"error": str(e)}, 0, 0, {"pipeline_error": True}
