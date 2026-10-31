"""Pre-registered ablations; M1 always uses the stock codec."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Baseline:
    name: str
    delta_qp: bool
    delta_x: bool
    stock_codec: bool = False


BASELINES = {
    "M1": Baseline("stock anchor", False, False, True),
    "M2": Baseline("delta-QP only", True, False),
    "M3": Baseline("delta-x only", False, True),
    "M4": Baseline("hybrid", True, True),
}


def apply_heads(name: str, delta_qp: Any, residual: Any) -> tuple[Any, Any]:
    baseline = BASELINES[name]
    return (delta_qp if baseline.delta_qp else delta_qp * 0,
            residual if baseline.delta_x else residual * 0)
