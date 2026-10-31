"""Extended-only VVC entry point. No VVC claims are enabled by this skeleton."""
from __future__ import annotations


class VTMBridge:
    phase = 3
    patch_timebox_person_weeks = 1

    def encode(self, *args: object, **kwargs: object) -> bytes:
        raise NotImplementedError(
            "Audit stock CuQpDeltaQp* first; see docs/VTM_AUDIT.md. "
            "Bridge and conformance evidence are required for the extended intra subset."
        )
