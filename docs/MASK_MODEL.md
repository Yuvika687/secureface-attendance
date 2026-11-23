---
license: mit
tags:
- onnx
- face-mask-detection
- mobilenetv2
---

# Face Mask Detector (ONNX)

**File:** `mask_detector.onnx` (9.6 MB, opset 13) — face-mask classifier:
given a cropped face, outputs **Mask** vs **No Mask** probabilities.

## Source & provenance

Converted (Keras → ONNX via `tf2onnx`, opset 13, see upstream `model2onnx.py`)
from `mask_detector.model` of
[chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection).
The weights in this file were verified to be **identical to the upstream
`mask_detector.model`** (same architecture, same parameters).

- Architecture: MobileNetV2 backbone (ImageNet-pretrained, frozen during
  upstream training) + AveragePooling + Flatten + Dense(128, relu) + Dense(2, softmax).
- Upstream training data: 4,095 real face images (`with_mask`: 2,165 /
  `without_mask`: 1,930). Author-reported accuracy ≈ 98%.

## License

Upstream project is **MIT licensed, © 2021 Chandrika Deb**.
This conversion is distributed under the same MIT license:

```
MIT License

Copyright (c) 2021 chandrikadeb7

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model spec

| Item | Value |
|---|---|
| Task | Mask / No-Mask classification on a cropped face |
| Input `input_1` | `[N, 224, 224, 3]`, float32, **RGB** face crop 224×224 with MobileNetV2 `preprocess_input` |
| Output `dense_1` | `[N, 2]` softmax — **index 0 = Mask**, index 1 = No Mask |
| Face detector | Not included — upstream uses an OpenCV SSD face detector (`face_detector/`); this file is the classifier only |

## Usage

```python
import onnxruntime as ort
import numpy as np, cv2

sess = ort.InferenceSession("mask_detector.onnx",
                            providers=["CPUExecutionProvider"])
face = cv2.cvtColor(cv2.imread("face.jpg"), cv2.COLOR_BGR2RGB)
face = cv2.resize(face, (224, 224)).astype(np.float32)
face = (face / 127.5 - 1.0)[None]          # MobileNetV2 preprocess_input
mask, no_mask = sess.run(None, {"input_1": face})[0]
print("Mask:" if mask > no_mask else "No Mask:", float(max(mask, no_mask)))
```
