"""
model.py
--------
Defines the LivenessCNN: a binary Live-vs-Spoof classifier built on top of
a pretrained backbone (mobilenet_v3_small, the project default; or
efficientnet_b0 / resnet18 for comparison).

Why these backbones?
- All three come with solid ImageNet pretrained weights in torchvision, so
  we get useful low-level features (edges, texture, moire patterns caused
  by screen replay, print artifacts) for free via transfer learning.
- MobileNetV3-Small and EfficientNet-B0 are small enough to run in real
  time on a normal laptop CPU/GPU; ResNet-18 is noticeably heavier
  (roughly 2x slower per training batch on this project's benchmark) and
  included here purely as a comparison point, not a deployment candidate.

The classifier head outputs a single logit (not two). We treat this as the
logit for the "SPOOF" class and train with BCEWithLogitsLoss, matching the
LABEL_SPOOF = 1 convention used across the project.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

from config import BACKBONE


class LivenessCNN(nn.Module):
    """Binary liveness classifier: outputs a single spoof-logit per image."""

    def __init__(self, backbone: str = BACKBONE, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = tvm.mobilenet_v3_small(weights=weights)
            in_features = net.classifier[0].in_features  # 576
            net.classifier = nn.Identity()
            self.feature_extractor = net
            self.feature_dim = in_features

        elif backbone == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features  # 1280
            net.classifier = nn.Identity()
            self.feature_extractor = net
            self.feature_dim = in_features

        elif backbone == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            net = tvm.resnet18(weights=weights)
            in_features = net.fc.in_features  # 512
            net.fc = nn.Identity()
            self.feature_extractor = net
            self.feature_dim = in_features

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose 'mobilenet_v3_small', 'efficientnet_b0', or 'resnet18'."
            )

        # Custom classification head for the binary liveness task.
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, 1),  # single logit -> P(spoof) after sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.head(features)
        return logits.squeeze(1)  # shape: (batch,)

    def freeze_backbone(self):
        """Freeze feature extractor weights (useful for the first warm-up epochs)."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze feature extractor weights for full fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


def build_model(backbone: str = BACKBONE, pretrained: bool = True) -> LivenessCNN:
    """Factory function used by train.py / evaluate.py / liveness_infer.py."""
    return LivenessCNN(backbone=backbone, pretrained=pretrained)


if __name__ == "__main__":
    # Quick sanity check: forward a dummy batch through the model.
    model = build_model()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Backbone: {model.backbone_name}")
    print(f"Output shape: {out.shape}  (expected: torch.Size([4]))")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
