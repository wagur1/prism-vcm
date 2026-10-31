"""Local quantizer STE; no discrete QP/quantizer gradient reaches the student."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FRAME_TYPES = {"unknown": 0, "I": 1, "P": 2, "reference_B": 3, "non_reference_B": 4}


class _QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor) -> Tensor:
        return torch.round(value)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> Tensor:
        return gradient


class ProxyCodec(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.analysis = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.dirac_(self.analysis.weight)
        nn.init.zeros_(self.analysis.bias)
        self.log_scale = nn.Parameter(torch.zeros(channels))
        self.frame_log_scale = nn.Embedding(len(FRAME_TYPES), 1)
        with torch.no_grad():
            self.frame_log_scale.weight[:, 0] = torch.tensor([0.0, 0.2, 0.0, -0.1, -0.2])

    def _qstep(self, feature: Tensor, dqp_map: Tensor, base_qp: Tensor) -> Tensor:
        if base_qp.numel() != feature.shape[0] or not torch.isfinite(base_qp).all():
            raise ValueError("base_qp must contain one finite raw QP per sample")
        qp = base_qp.detach().reshape(-1, 1, 1, 1)
        dqp = F.interpolate(dqp_map.detach(), size=feature.shape[-2:], mode="nearest")
        return torch.pow(2.0, (torch.clamp(qp + dqp, 0, 51) - 4.0) / 6.0)

    def R_hat(self, feature: Tensor, dqp_map: Tensor, frame_type: Tensor,
              base_qp: Tensor) -> Tensor:
        """Continuous entropy estimate; QP, frame-type and deltas are explicit inputs.

        feature may receive a continuous rate gradient during verified fine-tuning.
        No round/STE operation is used on this path; QP controls are detached.
        """
        step = self._qstep(feature, dqp_map, base_qp)
        frame_scale = self.frame_log_scale(frame_type.detach().long().reshape(-1))
        scale = (F.softplus(self.log_scale)[None, :, None, None] + 1e-6)
        activity = torch.sqrt(feature.square() + 1e-6)
        estimate = torch.log2(1.0 + activity * scale * frame_scale.exp()[..., None, None] / step)
        return estimate.flatten(1).sum(1)

    def forward(self, feature: Tensor, dqp_map: Tensor, frame_type: Tensor,
                base_qp: Tensor) -> dict[str, Tensor]:
        # Quantization trains only proxy-local analysis parameters.
        latent = self.analysis(feature.detach())
        step = self._qstep(latent, dqp_map, base_qp)
        reconstruction = _QuantizeSTE.apply(latent / step) * step
        return {"reconstruction": reconstruction,
                "rate": self.R_hat(feature, dqp_map, frame_type, base_qp)}
