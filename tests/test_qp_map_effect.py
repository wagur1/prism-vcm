import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from prism.codec.gate0 import residual_trace, verify_spatial_effect


def test_average_qp_cannot_substitute_spatial_evidence():
    anchor = {(0, 0, 0): 32., (0, 16, 0): 32.}
    changed = {(0, 0, 0): 38., (0, 16, 0): 26.}
    verify_spatial_effect(anchor, changed, np.array([[6., -6.]]))
    with pytest.raises(ValueError):
        verify_spatial_effect(anchor, changed, np.array([[-6., 6.]]))


def test_trace_excludes_skip_and_padding(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("frame,x,y,qp,has_residual,skip\n" + "".join(
        f"{frame},0,0,32,1,0\n{frame},0,1080,99,1,0\n{frame},16,0,99,0,1\n"
        for frame in range(10)))
    assert len(residual_trace(path, 1920, 1080)) == 10


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
