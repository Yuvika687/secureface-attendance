"""
prepare_casia_fasd.py
----------------------
CASIA-FASD is used ONLY as an extra generalization test set (never for
training). It is distributed as short .avi videos per subject rather than
still images, so this script extracts frames, runs face detection/crop
(reusing face_utils.py), and writes still images into the same
live/spoof folder convention used by dataset.py.

CASIA-FASD has 20 training subjects and 30 testing subjects; each subject
has 12 videos (3 genuine + 9 attack: warped-photo, cut-photo, and video
replay attacks), 600 videos total. Because raw archives vary in exact
directory/file naming across mirrors, this script expects you to first
sort the videos into two flat folders (one-time manual/scripted step,
following the per-subject genuine/attack split documented with your copy
of the dataset):

    <raw_root>/
        live/   *.avi
        spoof/  *.avi

Usage:
    python prepare_casia_fasd.py \\
        --raw_root /path/to/casia_fasd_sorted \\
        --out_root ../data/casia_fasd/test \\
        --frames_per_video 5
"""

import argparse
import os

import cv2

from face_utils import FaceDetector, crop_face


def extract_frames_from_video(video_path: str, frames_per_video: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Evenly spaced frame indices, skipping the very first/last few frames
    # which are often blank or out of focus.
    start, end = int(total * 0.05), int(total * 0.95)
    step = max(1, (end - start) // max(1, frames_per_video))
    indices = list(range(start, end, step))[:frames_per_video]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from CASIA-FASD videos")
    parser.add_argument("--raw_root", required=True, help="Folder with live/ and spoof/ .avi videos")
    parser.add_argument("--out_root", default="../data/casia_fasd/test")
    parser.add_argument("--frames_per_video", type=int, default=5)
    args = parser.parse_args()

    detector = FaceDetector()

    for class_name in ("live", "spoof"):
        video_dir = os.path.join(args.raw_root, class_name)
        if not os.path.isdir(video_dir):
            print(f"Skipping missing folder: {video_dir}")
            continue

        out_dir = os.path.join(args.out_root, class_name)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        videos = [f for f in sorted(os.listdir(video_dir)) if f.lower().endswith((".avi", ".mp4", ".mov"))]
        for vfile in videos:
            vpath = os.path.join(video_dir, vfile)
            frames = extract_frames_from_video(vpath, args.frames_per_video)

            for i, frame in enumerate(frames):
                detections = detector.detect(frame)
                if not detections:
                    continue
                # Take the largest detected face in the frame.
                best = max(detections, key=lambda d: d.width * d.height)
                face_crop = crop_face(frame, best)
                if face_crop is None or face_crop.size == 0:
                    continue

                base = os.path.splitext(vfile)[0]
                out_path = os.path.join(out_dir, f"{base}_f{i}.jpg")
                cv2.imwrite(out_path, face_crop)
                saved += 1

        print(f"[{class_name}] processed {len(videos)} videos -> saved {saved} face crops -> {out_dir}")

    detector.close()
    print("\nDone. CASIA-FASD test set written to:", os.path.abspath(args.out_root))


if __name__ == "__main__":
    main()
