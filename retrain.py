"""
SmartFace Retrain Script v2.0
==============================
Retrains the LBPH face recognition model from all face data.
Supports both encrypted (.enc) and legacy unencrypted (.jpg) face files.
Designed for 500+ employee scale with progress tracking and batch loading.

Usage: python retrain.py
"""

import os
import sys
import cv2
import numpy as np
import time

FACE_DATA_DIR = "face_data"
TRAINER_PATH = os.path.join(FACE_DATA_DIR, "trainer.yml")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)

all_faces = []
all_labels = []
errors = 0

print("=" * 60)
print("  SmartFace AI — Model Retraining")
print("=" * 60)

# Count files first for progress
enc_files = [f for f in os.listdir(FACE_DATA_DIR) if f.endswith('.enc')]
jpg_files = [f for f in os.listdir(FACE_DATA_DIR) 
             if (f.endswith('.jpg') or f.endswith('.png')) and not f.endswith('.enc')]
total_files = len(enc_files) + len(jpg_files)

if total_files == 0:
    print("\n❌ No face data found. Register employees first via the web app.")
    sys.exit(0)

print(f"\n📂 Found {len(enc_files)} encrypted + {len(jpg_files)} legacy files ({total_files} total)")

# Load encrypted face files
t_start = time.time()
try:
    from utils.face_utils import load_encrypted_face
    for i, filename in enumerate(enc_files):
        parts = filename.replace('.jpg.enc', '').split('_')
        if len(parts) >= 2:
            try:
                label = int(parts[1])
                filepath = os.path.join(FACE_DATA_DIR, filename)
                img = load_encrypted_face(filepath)
                if img is not None:
                    img_resized = cv2.resize(img, (200, 200))
                    all_faces.append(img_resized)
                    all_labels.append(label)
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only show first 5 errors
                    print(f"  ⚠️  Skipping {filename}: {e}")
        
        # Progress bar
        progress = int((i + 1) / len(enc_files) * 40)
        bar = "█" * progress + "░" * (40 - progress)
        pct = int((i + 1) / len(enc_files) * 100)
        sys.stdout.write(f"\r  Loading encrypted: [{bar}] {pct}% ({i+1}/{len(enc_files)})")
        sys.stdout.flush()
    
    if enc_files:
        print()  # Newline after progress bar

except ImportError:
    print("  ⚠️  Cannot load encrypted faces (missing dependencies)")

# Load legacy unencrypted files
for i, filename in enumerate(jpg_files):
    parts = filename.split('_')
    if len(parts) >= 2:
        try:
            label = int(parts[1])
            filepath = os.path.join(FACE_DATA_DIR, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (200, 200))
                all_faces.append(img_resized)
                all_labels.append(label)
        except (ValueError, IndexError):
            errors += 1
            continue
    
    if jpg_files:
        progress = int((i + 1) / len(jpg_files) * 40)
        bar = "█" * progress + "░" * (40 - progress)
        pct = int((i + 1) / len(jpg_files) * 100)
        sys.stdout.write(f"\r  Loading legacy:    [{bar}] {pct}% ({i+1}/{len(jpg_files)})")
        sys.stdout.flush()

if jpg_files:
    print()  # Newline after progress bar

t_load = time.time() - t_start

if len(all_faces) > 0:
    unique_labels = sorted(set(all_labels))
    print(f"\n📊 Statistics:")
    print(f"   • Total images loaded: {len(all_faces)}")
    print(f"   • Unique users: {len(unique_labels)}")
    print(f"   • Load time: {t_load:.1f}s")
    if errors > 0:
        print(f"   • Skipped (errors): {errors}")
    
    # Per-user breakdown
    print(f"\n👤 Per-user image count:")
    for label in unique_labels:
        count = all_labels.count(label)
        print(f"   User {label}: {count} images")
    
    # Train
    print(f"\n🧠 Training LBPH model...")
    t_train = time.time()
    recognizer.train(all_faces, np.array(all_labels))
    recognizer.save(TRAINER_PATH)
    t_train = time.time() - t_train
    
    file_size = os.path.getsize(TRAINER_PATH)
    print(f"   ✅ Training complete in {t_train:.1f}s")
    print(f"   📁 Model saved: {TRAINER_PATH} ({file_size:,} bytes)")
    
    # Memory estimate
    mem_mb = (len(all_faces) * 200 * 200) / (1024 * 1024)
    print(f"   💾 Peak memory usage: ~{mem_mb:.0f} MB")
    
    print(f"\n{'=' * 60}")
    print(f"  ✅ Model ready for {len(unique_labels)} user(s)")
    print(f"{'=' * 60}")
else:
    print("\n❌ No valid face data found. All files had errors.")
    print("   Try re-registering faces through the web app.")
