"""Dual-head MobileNetV4 controller with bottleneck QP conditioning."""
from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def upsample_qp_map(delta_qp: Tensor, width: int = 1920, height: int = 1080) -> Tensor:
    """Map CTU centers to QG centers on the CTU64-padded domain, edge replicated."""
    ctu_h, ctu_w = (height + 63) // 64, (width + 63) // 64
    if width <= 0 or height <= 0 or delta_qp.ndim != 4 or delta_qp.shape[1:] != (1, ctu_h, ctu_w):
        raise ValueError("Input must match [B,1,ceil(H/64),ceil(W/64)]")
    return F.interpolate(delta_qp, size=(ctu_h * 4, ctu_w * 4),
                         mode="bilinear", align_corners=False)


class MobileNetV4HybridController(nn.Module):
    def __init__(self, *, backbone: str = "auto", allow_fallback: bool = True):
        super().__init__()
        if backbone not in {"auto", "timm", "small"}:
            raise ValueError("Unknown backbone")
        if backbone != "small":
            try:
                import timm
            except ImportError:
                if not allow_fallback or backbone == "timm":
                    raise
                timm = None
            if timm is not None:
                self.backbone = timm.create_model("mobilenetv4_conv_small", pretrained=False,
                                                  features_only=True, out_indices=(-1,))
                channels = self.backbone.feature_info.channels()[-1]
                self.backbone_name = "mobilenetv4_conv_small"
            else:
                backbone = "small"
        if backbone == "small":
            warnings.warn("Small convolutional fallback is for smoke tests, not model claims", stacklevel=2)
            channels = 128
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(96, channels, 3, stride=2, padding=1), nn.SiLU())
            self.backbone_name = "small_conv_smoke_only"
        self.qp_condition = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, channels))
        self.delta_qp_head = nn.Conv2d(channels, 1, 1)
        self.residual_head = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.SiLU(),
                                            nn.Conv2d(32, 1, 1))

    def forward(self, x_low: Tensor, qp_scalar: Tensor, *, for_encoding: bool = False) -> dict[str, Tensor]:
        if x_low.ndim != 4 or x_low.shape[1:] != (3, 256, 256):
            raise ValueError("Expected x_low [B,3,256,256]")
        if qp_scalar.shape != (x_low.shape[0], 1):
            raise ValueError("Expected QP/51 [B,1]")
        if not torch.isfinite(qp_scalar).all() or torch.any((qp_scalar < 0) | (qp_scalar > 1)):
            raise ValueError("QP scalar must be finite and normalized by 51")
        features = self.backbone(x_low)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        conditioned = features + self.qp_condition(qp_scalar)[..., None, None]
        delta_qp = 6.0 * torch.tanh(F.interpolate(self.delta_qp_head(conditioned),
                     size=(17, 30), mode="bilinear", align_corners=False))
        residual = 8.0 * torch.tanh(F.interpolate(self.residual_head(conditioned),
                     size=(256, 256), mode="bilinear", align_corners=False))
        # No STE on the discrete head; oracle Huber supervises the continuous logits.
        return {"delta_qp": torch.round(delta_qp) if for_encoding else delta_qp,
                "residual": residual}
