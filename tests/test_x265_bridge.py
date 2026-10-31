import pytest

np = pytest.importorskip("numpy")

from prism.codec.x265_bridge import EncoderConfig, X265Encoder, qg_shape


def test_qg_padded_dimensions():
    assert qg_shape(1920, 1080) == (68, 120)
    assert qg_shape(1280, 720) == (48, 80)
    assert qg_shape(66, 66) == (8, 8)


def test_pinned_cqp_parameters(tmp_path):
    config = EncoderConfig()
    options = config.options(tmp_path / "frames.csv")
    for expected in ("qp=32", "aq-mode=1", "cutree=0", "sao=1", "qg-size=16", "csv-log-level=2"):
        assert expected in options


@pytest.mark.integration
def test_x265_buffer_lifetime_and_flush(x265_library, tmp_path):
    grid = np.zeros(qg_shape(64, 64), dtype=np.float32)
    with X265Encoder(x265_library, 64, 64, EncoderConfig(gop=1, bframes=0), tmp_path / "bits.csv") as enc:
        bitstream = enc.encode(bytes([128]) * (64 * 64 * 3 // 2), grid) + enc.flush()
        assert len(bitstream) > 0
        assert not enc._pending
