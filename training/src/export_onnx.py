"""
export_onnx.py
----------------
Exports a trained LivenessCNN checkpoint to ONNX for faster / more portable
CPU inference (useful if the demo needs to run on a machine without a
CUDA GPU, or needs to be embedded in a non-Python attendance app later).

Usage:
    python export_onnx.py --checkpoint ../models/best_liveness.pth --out ../models/liveness.onnx
"""

import argparse

import torch

import config
from model import build_model


def main():
    parser = argparse.ArgumentParser(description="Export LivenessCNN to ONNX")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--out", default=config.ONNX_MODEL_PATH)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backbone = checkpoint.get("backbone", config.BACKBONE)
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)

    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=["face_crop"],
        output_names=["spoof_logit"],
        dynamic_axes={"face_crop": {0: "batch_size"}, "spoof_logit": {0: "batch_size"}},
        opset_version=args.opset,
        dynamo=False,  # use the legacy TorchScript-based exporter (no onnxscript dependency)
    )

    print(f"Exported ONNX model to: {args.out}")
    print(f"Input:  face_crop  shape=(batch, 3, {image_size}, {image_size})")
    print("Output: spoof_logit  (apply sigmoid to get P(spoof))")

    # Sanity check the exported graph loads and runs.
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(args.out)
        outputs = session.run(None, {"face_crop": dummy_input.numpy()})
        print(f"ONNX Runtime sanity check OK. Output shape: {outputs[0].shape}")
    except ImportError:
        print("onnxruntime not installed; skipping runtime sanity check.")


if __name__ == "__main__":
    main()
