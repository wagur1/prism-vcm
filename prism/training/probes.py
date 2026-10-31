"""Real-codec calibration probe collection with frozen split and encode accounting."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from prism.data.splits import SplitLedger
from .manifest import EncodeManifest


@dataclass(frozen=True)
class Probe:
    sample_id: str
    base_qp: int
    delta_qp: int
    frame_type: str
    codec_state_sha256: str
    delta_x_sha256: str
    dependent_frame_ids: tuple[str, ...] = ()


def collect_probes(probes: Sequence[Probe], ledger: SplitLedger, manifest: EncodeManifest,
                   encode_from_state: Callable[[Probe], dict]) -> list[dict]:
    groups: dict[tuple[str, int, str, str], list[Probe]] = {}
    for probe in probes:
        key = (probe.sample_id, probe.base_qp, probe.frame_type, probe.delta_x_sha256)
        groups.setdefault(key, []).append(probe)
    for group in groups.values():
        expected = {delta for delta in (-1, 0, 1) if 0 <= group[0].base_qp + delta <= 51}
        if ({probe.delta_qp for probe in group} != expected or len(group) != len(expected) or
                len({probe.codec_state_sha256 for probe in group}) != 1):
            raise ValueError("Pair anchor and every feasible +/-1 probe from the same codec state")
    results = []
    for probe in probes:
        ledger.require(probe.sample_id, "calibration")
        for sample_id in probe.dependent_frame_ids:
            ledger.require(sample_id, "calibration")
        if probe.delta_qp not in {-1, 0, 1} or not 0 <= probe.base_qp + probe.delta_qp <= 51:
            raise ValueError("Calibration probes must be feasible +/-1 QP with a zero anchor")
        if len(probe.codec_state_sha256) != 64 or len(probe.delta_x_sha256) != 64:
            raise ValueError("Record codec-state and delta-x SHA256s for paired probes")
        # Count all dependent encodes before the callback so failed attempts are visible.
        for sample_id in (probe.sample_id, *probe.dependent_frame_ids):
            manifest.record("calibration_probe", sample_id, probe=probe.__dict__)
        results.append(encode_from_state(probe))
    return results
