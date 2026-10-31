import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from prism.codec.gate0 import residual_trace, verify_spatial_effect


def test_average_qp_cannot_substitute_spatial_evidence():
    anchor = {(0, 0, 0): 32., (0, 16, 0): 32.}
    changed = {(0, 0, 0): 38., (0, 16, 0): 26.}
    verify_spatial_effect(anchor, changed, np.array([[6., -6.]]),
                          min_pairs=1, min_qg=1)
    with pytest.raises(ValueError, match="Sign-fidelity violation"):
        verify_spatial_effect(anchor, changed, np.array([[-6., 6.]]),
                              min_pairs=1, min_qg=1)


def test_v14_bands_and_aggregate():
    # I-frame band: median 6.0 passes, 3.0 fails.
    i_anchor = {(0, 0, 0): 32.}
    i_changed = {(0, 0, 0): 38.}
    verify_spatial_effect(i_anchor, i_changed, np.array([[6.]]),
                          min_pairs=1, min_qg=1)
    with pytest.raises(ValueError, match="outside"):
        verify_spatial_effect(i_anchor, {(0, 0, 0): 35.}, np.array([[6.]]),
                              min_pairs=1, min_qg=1)
    # P-frame band: median 3.0 passes at frame 1; 2.6 passes band but fails aggregate.
    p_anchor = {(1, 0, 0): 32., (2, 0, 0): 32.}
    p_changed = {(1, 0, 0): 35., (2, 0, 0): 35.}
    verify_spatial_effect(p_anchor, p_changed, np.array([[6.]]),
                          min_pairs=1, min_qg=1)
    weak = {(1, 0, 0): 34.6, (2, 0, 0): 34.6}
    with pytest.raises(ValueError, match="too weak"):
        verify_spatial_effect(p_anchor, weak, np.array([[6.]]),
                              min_pairs=1, min_qg=1)
    # I-frame treated per i_frames override: frame 1 as intra.
    verify_spatial_effect(p_anchor, p_changed, np.array([[6.]]),
                          min_pairs=1, min_qg=1, i_frames={1, 2})


def test_min_count_guard_fails_loudly():
    anchor = {(0, 0, 0): 32., (0, 16, 0): 32.}
    changed = {(0, 0, 0): 38., (0, 16, 0): 38.}
    with pytest.raises(ValueError, match="Insufficient residual pairs"):
        verify_spatial_effect(anchor, changed, np.array([[6.]]))


@pytest.mark.integration
def test_encode_ab_residual_trace_and_independent_decode(x265_library, tmp_path):
    pin = os.environ.get("PRISM_GATE0_PIN")
    decoder = os.environ.get("PRISM_HEVC_REFERENCE_COMMAND_JSON")
    stock_pin = os.environ.get("PRISM_GATE0_STOCK_PIN")
    stock_shim = os.environ.get("PRISM_X265_STOCK_SHIM")
    stock_library = os.environ.get("PRISM_X265_STOCK_LIBRARY")
    if not all((pin, decoder, stock_pin, stock_shim, stock_library)):
        pytest.skip("GATE 0 needs reviewed patched/stock pins and independent HEVC reference decoder")
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg decoder unavailable")
    from scripts.spike_gate0 import read_build, run_spike
    assert json.loads(Path(pin).read_text()) == x265_library.pin.__dict__
    stock = read_build(Path(stock_shim), Path(stock_library), Path(stock_pin))
    report = run_spike(tmp_path, x265_library, "ffmpeg", json.loads(decoder), stock=stock)
    assert report["status"] == "PASS"
