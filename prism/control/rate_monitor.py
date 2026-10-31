"""Actual ABR/VBV rate accounting, separate from the projection surrogate."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RateMonitor:
    target_bits_per_second: float
    duration_seconds: float = 0.0
    total_bits: int = 0

    def observe(self, *, whole_bitstream_bytes: int, frame_duration: float,
                encoder_buffer_fullness_bits: float | None = None,
                encoder_buffer_capacity_bits: float | None = None) -> dict:
        if (whole_bitstream_bytes < 0 or not math.isfinite(frame_duration) or frame_duration <= 0 or
                not math.isfinite(self.target_bits_per_second) or self.target_bits_per_second <= 0):
            raise ValueError("Invalid actual bitrate observation")
        self.total_bits += whole_bitstream_bytes * 8
        self.duration_seconds += frame_duration
        vbv_ok = None
        if encoder_buffer_capacity_bits is not None or encoder_buffer_fullness_bits is not None:
            if encoder_buffer_capacity_bits is None or encoder_buffer_fullness_bits is None:
                raise ValueError("Supply actual encoder buffer fullness and capacity together")
            vbv_ok = (math.isfinite(encoder_buffer_capacity_bits) and
                      math.isfinite(encoder_buffer_fullness_bits) and encoder_buffer_capacity_bits > 0 and
                      0 <= encoder_buffer_fullness_bits <= encoder_buffer_capacity_bits)
        actual = self.total_bits / self.duration_seconds
        return {"actual_bits_per_second": actual, "target_bits_per_second": self.target_bits_per_second,
                "actual_rate_ratio": actual / self.target_bits_per_second,
                "vbv_buffer_valid": vbv_ok, "actual_buffer_fullness_bits": encoder_buffer_fullness_bits}
