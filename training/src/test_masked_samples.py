"""
test_masked_samples.py
------------------------
Stage 1 sanity check: verifies that MediaPipe face detection still works
when the person is wearing a face mask, using a small local folder of
masked-face images (data/samples/masked/).

This does NOT involve the liveness model at all -- it is purely a
detection-quality check, as requested in the project spec ("support a
small local folder of masked face images for checking Stage 1").

Usage:
    python test_masked_samples.py --images_dir ../data/samples/masked
"""

import argparse
import os

import cv2

import config
from face_utils import FaceDetector, crop_face

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    parser = argparse.ArgumentParser(description="Check mask-aware face detection on sample images")
    parser.add_argument("--images_dir", default=os.path.join(config.SAMPLES_DIR, "masked"))
    parser.add_argument("--out_dir", default=os.path.join(config.PREDICTIONS_DIR, "masked_detection_check"))
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        print(f"Directory not found: {args.images_dir}")
        print("Add a few masked-face photos there and re-run this script.")
        return

    images = [f for f in sorted(os.listdir(args.images_dir)) if f.lower().endswith(VALID_EXTENSIONS)]
    if not images:
        print(f"No images found in {args.images_dir}. Add some masked-face .jpg/.png files and re-run.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    detector = FaceDetector()

    detected_count = 0
    for fname in images:
        path = os.path.join(args.images_dir, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"  [skip] could not read {fname}")
            continue

        detections = detector.detect(image)
        status = "DETECTED" if detections else "MISSED"
        if detections:
            detected_count += 1
            for det in detections:
                cv2.rectangle(image, (det.x, det.y), (det.x2, det.y2), (0, 200, 0), 2)
                cv2.putText(image, f"{det.confidence:.2f}", (det.x, max(0, det.y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        print(f"  [{status}] {fname}  ({len(detections)} face(s))")
        cv2.imwrite(os.path.join(args.out_dir, f"annotated_{fname}"), image)

    detector.close()

    print(f"\nMask-aware detection rate: {detected_count}/{len(images)} "
          f"({100.0 * detected_count / len(images):.1f}%)")
    print(f"Annotated images saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
