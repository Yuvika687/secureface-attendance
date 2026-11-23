"""Prepare the three external inference assets for a fresh local installation."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from shutil import copyfileobj

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    ("models/w600k_r50.onnx", "https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx",
     "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
    ("models/face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    ("liveness/weights/blaze_face_short_range.tflite", "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
     "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"),
)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def prepare():
    for filename, url, expected in ASSETS:
        target = ROOT / filename
        if target.is_file() and digest(target) == expected:
            continue
        print(f"Downloading {target.name} ...", flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with urlopen(url, timeout=90) as response, temporary.open("wb") as output:
                copyfileobj(response, output)
            if digest(temporary) != expected:
                raise ValueError(f"Checksum mismatch for {target.name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

if __name__ == "__main__":
    prepare()
