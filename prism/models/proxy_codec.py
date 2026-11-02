"""Differentiable block-DCT virtual codec with a calibrated Laplace rate model."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FRAME_TYPES = {"unknown": 0, "I": 1, "P": 2, "reference_B": 3, "non_reference_B": 4}


class _QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor) -> Tensor:
        ctx.save_for_backward(value)
        # Round-to-nearest supplies the zero bin |value| < 0.5 (the dead zone).
        return torch.round(value)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> Tensor:
        (value,) = ctx.saved_tensors
        # Council CX-2 Fourier soft-STE: slope stays in [0.8, 1.2], even at zero.
        return gradient * (1.0 - 0.2 * torch.cos(2.0 * math.pi * value))


class ProxyCodec(nn.Module):
    """DVC for images in 8-bit pixel units, shaped [B, channels, H, W].

    DCT and quantization weights are fixed. The 64 positive subband scales and
    per-(base QP, frame type) affine rate factors can be fitted on training data
    and frozen before independent CalibrationGate evaluation. Rates are total
    bits per image; fit against x265 bits (multiply measured bytes by eight).
    Initial factors are priors, not a claim of calibration against x265.
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        positions = torch.arange(8, dtype=torch.float64)
        basis = torch.cos(math.pi * positions[:, None] * (2 * positions + 1) / 16)
        basis *= 0.5
        basis[0] /= math.sqrt(2)
        self.register_buffer("dct_basis", basis.float())
        # HEVC default intra 8x8 scaling list, normalized to a unit DC step.
        self.register_buffer("frequency_weights", torch.tensor([
            [16, 16, 16, 16, 17, 18, 21, 24],
            [16, 16, 16, 16, 17, 19, 22, 25],
            [16, 16, 17, 18, 20, 22, 25, 29],
            [16, 16, 18, 21, 24, 27, 31, 36],
            [17, 17, 20, 24, 30, 35, 41, 47],
            [18, 19, 22, 27, 35, 44, 54, 65],
            [21, 22, 25, 31, 41, 54, 70, 88],
            [24, 25, 29, 36, 47, 65, 88, 115],
        ], dtype=torch.float32) / 16.0)
        dc_mask = torch.zeros(8, 8)
        dc_mask[0, 0] = 1.0
        self.register_buffer("dc_mask", dc_mask)
        self.log_scale = nn.Parameter(torch.zeros(channels, 8, 8))
        frame_prior = torch.tensor([0.0, 0.2, 0.0, -0.1, -0.2])
        self.rate_log_scale = nn.Parameter(frame_prior.repeat(52, 1))
        self.rate_log_bias = nn.Parameter(torch.full((52, len(FRAME_TYPES)), -6.0))

    def _transform(self, feature: Tensor) -> Tensor:
        if (feature.ndim != 4 or feature.shape[1] != self.channels or
                min(feature.shape) <= 0 or not feature.is_floating_point()):
            raise ValueError("Expected a nonempty floating image [B, channels, H, W]")
        # DCT and rate arithmetic need float32 under AMP (pixel DC can exceed 1k).
        image = feature if feature.dtype == torch.float64 else feature.float()
        image = image.clamp(0.0, 255.0)
        height, width = image.shape[-2:]
        image = F.pad(image, (0, -width % 8, 0, -height % 8), mode="replicate")
        blocks = image.unfold(2, 8, 8).unfold(3, 8, 8)
        prediction = blocks.mean(dim=(-2, -1), keepdim=True)
        basis = self.dct_basis.to(dtype=image.dtype)
        # Intra block-mean prediction isolates AC from absolute brightness.
        ac = basis @ (blocks - prediction) @ basis.T
        # Preserve/quantize the level-shifted block mean as its own DC subband.
        dc = 8.0 * (prediction - 128.0)
        return ac * (1.0 - self.dc_mask) + dc * self.dc_mask

    def _qstep(self, coefficients: Tensor, dqp_map: Tensor, base_qp: Tensor) -> Tensor:
        batch = coefficients.shape[0]
        if base_qp.numel() != batch or not torch.isfinite(base_qp).all():
            raise ValueError("base_qp must contain one finite raw QP per sample")
        if (dqp_map.ndim != 4 or dqp_map.shape[:2] != (batch, 1) or
                min(dqp_map.shape[-2:]) <= 0 or not torch.isfinite(dqp_map).all()):
            raise ValueError("dqp_map must be finite with shape [B, 1, Hq, Wq]")
        qp = base_qp.to(coefficients).reshape(-1, 1, 1, 1)
        dqp = F.interpolate(dqp_map.to(coefficients), size=coefficients.shape[2:4],
                            mode="bilinear", align_corners=False)
        step = torch.pow(2.0, ((qp + dqp).clamp(0.0, 51.0) - 4.0) / 6.0)
        return step[..., None, None] * self.frequency_weights

    def _rate(self, coefficients: Tensor, step: Tensor, frame_type: Tensor,
              base_qp: Tensor) -> Tensor:
        if (frame_type.numel() != coefficients.shape[0] or
                not torch.isfinite(frame_type).all() or
                torch.any(frame_type != frame_type.long()) or
                torch.any((frame_type < 0) | (frame_type >= len(FRAME_TYPES)))):
            raise ValueError("frame_type must contain one valid FRAME_TYPES ID per sample")
        scale = (F.softplus(self.log_scale) + 1e-6)[None, :, None, None]
        activity = torch.sqrt(coefficients.square() + 1e-6)
        ac_bits = (torch.log1p(scale * activity / step) / math.log(2.0)
                   * (1.0 - self.dc_mask)).flatten(1).sum(1)

        # Predict DC from neighboring blocks, never from absolute brightness.
        dc = coefficients[..., 0, 0]
        horizontal = F.pad(dc[..., 1:] - dc[..., :-1], (1, 0))
        vertical = F.pad(dc[:, :, 1:, :] - dc[:, :, :-1, :], (0, 0, 1, 0))
        dc_activity = torch.sqrt(horizontal.square() + vertical.square() + 1e-6)
        dc_bits = (torch.log1p(scale[..., 0, 0] * dc_activity / step[..., 0, 0])
                   / math.log(2.0)).flatten(1).sum(1)
        raw_rate = ac_bits + dc_bits

        # Interpolate calibration factors for fractional base QPs. Local delta-QP
        # remains on the differentiable quantization path, not a discrete index.
        qp = base_qp.to(coefficients).reshape(-1).clamp(0.0, 51.0)
        lower = qp.floor().long()
        upper = (lower + 1).clamp(max=51)
        fraction = qp - lower
        frame = frame_type.to(device=coefficients.device, dtype=torch.long).reshape(-1)
        alpha = torch.lerp(self.rate_log_scale[lower, frame].exp(),
                           self.rate_log_scale[upper, frame].exp(), fraction)
        beta = torch.lerp(F.softplus(self.rate_log_bias[lower, frame]),
                          F.softplus(self.rate_log_bias[upper, frame]), fraction)
        return alpha * raw_rate + beta

    def R_hat(self, feature: Tensor, dqp_map: Tensor, frame_type: Tensor,
              base_qp: Tensor) -> Tensor:
        """Continuous 64-subband rate; gradients reach both image and delta-QP."""
        with torch.autocast(device_type=feature.device.type, enabled=False):
            coefficients = self._transform(feature)
            step = self._qstep(coefficients, dqp_map, base_qp)
            return self._rate(coefficients, step, frame_type, base_qp)

    def forward(self, feature: Tensor, dqp_map: Tensor, frame_type: Tensor,
                base_qp: Tensor) -> dict[str, Tensor]:
        with torch.autocast(device_type=feature.device.type, enabled=False):
            coefficients = self._transform(feature)
            step = self._qstep(coefficients, dqp_map, base_qp)
            quantized = _QuantizeSTE.apply(coefficients / step) * step
            basis = self.dct_basis.to(dtype=coefficients.dtype)
            blocks = basis.T @ quantized @ basis + 128.0
            batch, channels, rows, cols = blocks.shape[:4]
            reconstruction = blocks.permute(0, 1, 2, 4, 3, 5).reshape(
                batch, channels, rows * 8, cols * 8)
            height, width = feature.shape[-2:]
            reconstruction = reconstruction[..., :height, :width].clamp(0.0, 255.0)
            return {"reconstruction": reconstruction.to(dtype=feature.dtype),
                    "rate": self._rate(coefficients, step, frame_type, base_qp)}
