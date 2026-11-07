"""
gradcam.py
----------
Manual-hooks Grad-CAM for LivenessCNN's single spoof-logit output,
targeting the last conv block: model.feature_extractor.features[12]
(a Conv2dNormActivation producing a 7x7x576 feature map for a 224x224
input on mobilenet_v3_small -- confirmed by instantiating the real model
and hooking it before writing this module).

Standard multi-class Grad-CAM backprops from the chosen class's logit, not
its softmax/sigmoid probability -- the same reasoning applies here: our
model is often very confident (prob near 0 or 1), where the sigmoid is
saturated and its gradient is close to zero, which would make the raw-
activation gradients vanish too. Backpropagating from the logit avoids
that and is the standard choice for exactly this reason.

Usage:
    from gradcam import GradCAM
    cam = GradCAM(model, target_layer=model.feature_extractor.features[12])
    heatmap, prob_spoof = cam(input_tensor)   # input_tensor: (1,3,224,224)
    cam.remove()   # always safe to call even if already removed

Or as a context manager, which guarantees hook removal even on error:
    with GradCAM(model, model.feature_extractor.features[12]) as cam:
        heatmap, prob_spoof = cam(input_tensor)
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Registers forward/backward hooks on a single target layer, computes a
    class-activation heatmap for LivenessCNN's single-logit output, and
    removes its own hooks afterward -- designed to be safe to use
    repeatedly against a shared, already-loaded model (e.g. the Flask
    backend's single global `predictor.model`) without leaving the model
    in a different state than it started in.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._fwd_handle = None
        self._bwd_handle = None
        self._register()

    def _register(self):
        def forward_hook(module, inp, out):
            self._activations = out

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        # register_full_backward_hook is the modern, non-deprecated API
        # (register_backward_hook has well-known gradient-ordering bugs
        # with modules that have multiple inputs/outputs).
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove(self):
        """Removes both hooks. Safe to call multiple times."""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def __call__(self, input_tensor: torch.Tensor, target: str = "predicted") -> Tuple[np.ndarray, float]:
        """
        Args:
            input_tensor: (1, 3, H, W) preprocessed tensor (same transform
                as the real eval pipeline -- see dataset.get_eval_transform),
                on the same device as the model.
            target: which class's "evidence" the heatmap explains.
                - "predicted" (default): whichever class the model actually
                  predicted for this input (spoof-evidence if P(spoof)>=0.5,
                  live-evidence otherwise). This is the general-purpose,
                  always-meaningful choice -- see below for why it's the
                  default.
                - "spoof": always explain spoof-evidence, regardless of the
                  prediction. Useful when deliberately comparing "where is
                  the spoof evidence" across both correctly- and
                  incorrectly-classified spoof images (e.g. a false accept
                  vs. a correctly-rejected example of the same attack type).
                - "live": always explain live-evidence.

            Why this matters, found empirically while testing this module:
            backpropagating from the raw spoof-logit (the naive choice)
            produces a completely degenerate, all-zero heatmap for any
            confidently-LIVE prediction -- Grad-CAM's final ReLU zeroes out
            an all-negative gradient-weighted activation map, and a
            confident live prediction can genuinely have no spatial region
            of positive "spoof evidence" at all. Backpropagating from
            -logit (live-evidence) for those cases fixes it, analogous to
            standard multi-class Grad-CAM targeting the predicted class's
            own logit rather than always the same fixed class.

        Returns:
            (heatmap, prob_spoof)
            heatmap: (H, W) float32 numpy array in [0, 1], upsampled to the
                input's spatial size.
            prob_spoof: float, P(spoof) for this input (sigmoid of the
                logit, independent of which `target` was used for the
                backward pass).
        """
        if target not in ("predicted", "spoof", "live"):
            raise ValueError(f"target must be 'predicted', 'spoof', or 'live', got {target!r}")

        was_training = self.model.training
        self.model.eval()  # Grad-CAM must not change BN running stats / use dropout

        # Grad-CAM needs gradients, so this must run with grad enabled even
        # if the caller is normally inside a torch.no_grad() context (e.g.
        # liveness_infer.LivenessPredictor.predict uses @torch.no_grad()).
        with torch.enable_grad():
            input_tensor = input_tensor.clone().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)

            logit = self.model(input_tensor)  # shape (1,)
            prob_spoof = torch.sigmoid(logit).item()

            if target == "spoof":
                explain_spoof = True
            elif target == "live":
                explain_spoof = False
            else:  # "predicted"
                explain_spoof = prob_spoof >= 0.5

            # Backprop from the raw logit (or its negation for "live"
            # evidence), not the sigmoid probability -- avoids vanishing
            # gradients at saturated/confident predictions.
            target_score = logit if explain_spoof else -logit
            target_score.sum().backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError(
                    "GradCAM hooks did not fire -- target_layer was not reached "
                    "during the forward/backward pass. Wrong layer reference?"
                )

            activations = self._activations.detach()[0]  # (C, h, w)
            gradients = self._gradients.detach()[0]       # (C, h, w)

            weights = gradients.mean(dim=(1, 2))           # (C,) global-average-pooled gradient per channel
            cam = torch.einsum("c,chw->hw", weights, activations)
            cam = F.relu(cam)

            cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,h,w) for interpolate
            cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                # Degenerate (uniform) activation -- return zeros rather than
                # dividing by ~0, and let the caller's sanity check catch this.
                cam = np.zeros_like(cam)

        if not was_training:
            self.model.eval()  # restore exactly the mode we found it in
        else:
            self.model.train()

        return cam.astype(np.float32), prob_spoof


def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends a Grad-CAM heatmap (H,W) in [0,1] onto a BGR face crop for
    visualization. Returns a BGR image the same size as face_bgr."""
    import cv2
    h, w = face_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, face_bgr, 1 - alpha, 0)
    return overlay


if __name__ == "__main__":
    import sys
    import time
    import cv2

    sys.path.insert(0, ".")
    import config
    from dataset import get_eval_transform
    from face_utils import bgr_crop_to_pil
    from model import build_model

    if len(sys.argv) < 2:
        print("Usage: python gradcam.py <path_to_face_crop_image>")
        sys.exit(1)

    device = config.DEVICE
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model = build_model(backbone=checkpoint.get("backbone", config.BACKBONE), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_eval_transform(checkpoint.get("image_size", config.IMAGE_SIZE))

    img_bgr = cv2.imread(sys.argv[1])
    pil_img = bgr_crop_to_pil(img_bgr)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with GradCAM(model, model.feature_extractor.features[12]) as cam:
        heatmap, prob = cam(tensor)

    print(f"P(spoof) = {prob:.4f}")
    print(f"Heatmap shape: {heatmap.shape}, range [{heatmap.min():.3f}, {heatmap.max():.3f}], std={heatmap.std():.4f}")

    overlay = overlay_heatmap(cv2.resize(img_bgr, (224, 224)), heatmap)
    out_path = "gradcam_debug_overlay.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"Saved overlay to {out_path}")
