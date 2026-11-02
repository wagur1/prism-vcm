"""DINO distillation on differentiable virtual-codec reconstructions."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    lambda_semantic: float = 1.0
    lambda_align: float = 1.0
    lambda_rate: float = 0.01
    lambda_bound: float = 0.1
    huber_delta: float = 1.0
    rate_reference: float = 1.0


class DistillationLoss(nn.Module):
    def __init__(self, config: LossConfig = LossConfig()):
        super().__init__()
        if config.rate_reference <= 0 or config.huber_delta <= 0:
            raise ValueError("Positive loss scales are required")
        self.config = config

    def forward(self, *, x_raw: Tensor, residual: Tensor, delta_qp: Tensor,
                oracle_qp: Tensor, dino_target: Tensor,
                semantic_encoder: Callable[[Tensor], Tensor], rate: Tensor,
                non_reference_b: Tensor | None = None,
                background: Tensor | None = None,
                x_reconstructed: Tensor | None = None) -> dict[str, Tensor]:
        # x_raw is RGB in pixel units; residual is shared luma delta in pixel units.
        # Freeze semantic-encoder weights, but retain its input gradient to delta-x.
        # Pass DVC(x_raw + residual)["reconstruction"] to supervise codec damage,
        # including at residual=0. Keep its existing graph all the way to residual.
        # Legacy callers may instead compose DVC into semantic_encoder themselves.
        semantic_input = x_raw + residual if x_reconstructed is None else x_reconstructed
        features = semantic_encoder(semantic_input)
        if features.shape != dino_target.shape or features.shape[-1] != 768:
            raise ValueError("Expected matching DINOv2-Base features with embedding dimension 768")
        semantic = (1.0 - F.cosine_similarity(features, dino_target.detach(), dim=-1)).mean()
        align = F.huber_loss(delta_qp, oracle_qp.detach(), delta=self.config.huber_delta)
        limits = torch.full((residual.shape[0], 1, 1, 1), 3.0,
                            device=residual.device, dtype=residual.dtype)
        if non_reference_b is not None:
            limits = torch.where(non_reference_b.reshape(-1, 1, 1, 1), 8.0, limits)
        bound = F.relu(delta_qp.abs() - 6.0).square().mean()
        bound = bound + F.relu(residual.abs() - limits).square().mean()
        if background is not None:
            bound = bound + (F.relu(delta_qp - 4.0).square() * background).mean()
        rate_loss = rate.mean() / self.config.rate_reference
        cfg = self.config
        total = (cfg.lambda_semantic * semantic + cfg.lambda_align * align +
                 cfg.lambda_rate * rate_loss + cfg.lambda_bound * bound)
        return {"total": total, "semantic": semantic, "align": align,
                "rate": rate_loss, "bound": bound}
