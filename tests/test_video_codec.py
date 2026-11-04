"""Sequence contracts run without a native shim or an FFmpeg installation."""
import io
from unittest.mock import Mock, call

import numpy as np
import pytest

from prism.codec.x265_bridge import EncoderConfig, X265Encoder, X265Unavailable, qg_shape
from prism.video import VideoCodecRunner
from prism.video import video_codec


WIDTH, HEIGHT = 66, 64
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2


@pytest.fixture
def encoder_stub(monkeypatch):
    encoder = Mock(spec=X265Encoder)
    encoder.encode.return_value = b""
    encoder.flush.return_value = b"delayed"
    # Supply the same lifecycle as the native context manager.
    encoder.__enter__ = Mock(return_value=encoder)

    def close(*exc):
        encoder.close()
        return False

    encoder.__exit__ = Mock(side_effect=close)
    factory = Mock(return_value=encoder)
    monkeypatch.setattr(video_codec, "X265Encoder", factory)
    library = object()
    return VideoCodecRunner(library=library), encoder, factory


def test_default_config_loads_no_native_library(monkeypatch):
    loader = Mock(side_effect=AssertionError("Native library loaded eagerly"))
    monkeypatch.setattr(video_codec, "X265Library", loader)
    runner = VideoCodecRunner()
    assert runner.config == EncoderConfig(
        gop=32, bframes=3, fps=30, preset="medium", rc_mode="CRF", crf=28.0)
    loader.assert_not_called()


@pytest.mark.parametrize("bframes", [0, 3], ids=["low_delay_p", "random_access"])
def test_inter_frame_configuration_reaches_encoder(encoder_stub, tmp_path, bframes):
    runner, encoder, factory = encoder_stub
    config = EncoderConfig(gop=32, bframes=bframes, fps=30, preset="medium",
                           rc_mode="CRF", crf=23.5)
    runner = VideoCodecRunner(config, library=runner.library)
    path = tmp_path / "video.hevc"
    runner.encode_sequence([bytes(FRAME_SIZE)], None, path, WIDTH, HEIGHT)
    factory.assert_called_once_with(runner.library, WIDTH, HEIGHT, config,
                                    tmp_path / "video.hevc.csv")
    options = dict(line.split("=", 1) for line in config.options(path).splitlines())
    assert options["keyint"] == options["min-keyint"] == "32"
    assert options["bframes"] == str(bframes)
    assert options["fps"] == "30"
    assert options["crf"] == "23.5"
    assert options["scenecut"] == "0"
    assert "qp" not in options
    encoder.flush.assert_called_once_with()


@pytest.mark.parametrize("source_kind", ["iterator", "path"])
def test_streams_frames_offsets_and_delayed_packets(encoder_stub, tmp_path, source_kind):
    runner, encoder, factory = encoder_stub
    frames = [bytes([i]) * FRAME_SIZE for i in range(5)]
    grids = [np.full(qg_shape(WIDTH, HEIGHT), i / 4, dtype=np.float32) for i in range(5)]
    grids[2] = None
    offsets = Mock(side_effect=grids)
    packets = [b"", b"", b"I", b"P", b"B"]
    encoder.encode.side_effect = packets
    encoder.flush.return_value = b"BB-tail"
    source = iter(frames)
    if source_kind == "path":
        source = tmp_path / "input.yuv"
        source.write_bytes(b"".join(frames))
    output = tmp_path / "nested" / "video.hevc"

    result = runner.encode_sequence(source, offsets, output, WIDTH, HEIGHT)

    assert output.read_bytes() == b"IPBBB-tail"
    assert result == {"total_bytes": 10, "num_frames": 5,
                      "bitrate_kbps": pytest.approx(0.48), "bitstream_path": output}
    assert offsets.call_args_list == [call(i, WIDTH, HEIGHT) for i in range(5)]
    for i, args in enumerate(encoder.encode.call_args_list):
        assert args.args[0] == frames[i]
        assert args.args[1] is grids[i]
    factory.assert_called_once()
    encoder.flush.assert_called_once_with()
    encoder.close.assert_called_once_with()
    assert [c[0] for c in encoder.method_calls] == ["encode"] * 5 + ["flush", "close"]


@pytest.mark.parametrize("count,packet_size,tail_size,fps,expected_kbps", [
    (1, 0, 100, 30, 24.0),
    (4, 250, 0, 24, 48.0),
    (60, 0, 7500, 30, 30.0),
    (3, 100, 75, 60, 60.0),
])
def test_bitrate_counts_all_bytes_and_input_frames(
        encoder_stub, tmp_path, count, packet_size, tail_size, fps, expected_kbps):
    runner, encoder, factory = encoder_stub
    encoder.encode.return_value = b"x" * packet_size
    encoder.flush.return_value = b"B" * tail_size
    output = tmp_path / "out.hevc"
    result = runner.encode_sequence((bytes(FRAME_SIZE) for _ in range(count)),
                                    None, output, WIDTH, HEIGHT, fps=fps)
    assert result["total_bytes"] == output.stat().st_size == count * packet_size + tail_size
    assert result["num_frames"] == count
    assert result["bitrate_kbps"] == pytest.approx(expected_kbps)
    assert factory.call_args.args[3].fps == fps
    assert runner.config.fps == 30  # Per-call timing never mutates the base config.
    assert all(args.args[1] is None for args in encoder.encode.call_args_list)


@pytest.mark.parametrize("fps", [0, -1, 1.5, float("nan"), True])
def test_rejects_invalid_fps_before_opening_encoder(encoder_stub, tmp_path, fps):
    runner, _, factory = encoder_stub
    with pytest.raises(ValueError, match="fps"):
        runner.encode_sequence([], None, tmp_path / "out.hevc", WIDTH, HEIGHT, fps)
    factory.assert_not_called()


@pytest.mark.parametrize("width,height", [(0, 64), (64, -2), (65, 64), (64, 63)])
def test_rejects_invalid_dimensions(encoder_stub, tmp_path, width, height):
    runner, _, factory = encoder_stub
    with pytest.raises(ValueError, match="dimensions"):
        runner.encode_sequence([], None, tmp_path / "out.hevc", width, height)
    with pytest.raises(ValueError, match="dimensions"):
        list(runner.decode_bitstream(tmp_path / "out.hevc", width, height))
    factory.assert_not_called()


@pytest.mark.parametrize("source_kind", ["iterator", "path"])
def test_rejects_empty_sequences(encoder_stub, tmp_path, source_kind):
    runner, encoder, _ = encoder_stub
    source = iter([])
    if source_kind == "path":
        source = tmp_path / "empty.yuv"
        source.write_bytes(b"")
    with pytest.raises(ValueError, match="empty sequence"):
        runner.encode_sequence(source, None, tmp_path / "out.hevc", WIDTH, HEIGHT)
    encoder.encode.assert_not_called()
    encoder.flush.assert_not_called()
    encoder.close.assert_called_once_with()


@pytest.mark.parametrize("source_kind", ["iterator", "path"])
def test_rejects_partial_yuv_frames(encoder_stub, tmp_path, source_kind):
    runner, encoder, _ = encoder_stub
    frames = [bytes(FRAME_SIZE), b"partial"]
    source = iter(frames)
    if source_kind == "path":
        source = tmp_path / "truncated.yuv"
        source.write_bytes(b"".join(frames))
    offsets = Mock(return_value=None)
    with pytest.raises(ValueError, match="frame size at frame 1"):
        runner.encode_sequence(source, offsets, tmp_path / "out.hevc", WIDTH, HEIGHT)
    offsets.assert_called_once_with(0, WIDTH, HEIGHT)
    encoder.encode.assert_called_once()
    encoder.flush.assert_not_called()
    encoder.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["callback", "input", "encode", "flush"])
def test_encoder_is_closed_on_failure(encoder_stub, tmp_path, failure):
    runner, encoder, _ = encoder_stub
    error = RuntimeError("injected failure")
    callback = None
    source = [bytes(FRAME_SIZE)]
    if failure == "callback":
        callback = Mock(side_effect=error)
    elif failure == "input":
        def broken_source():
            yield bytes(FRAME_SIZE)
            raise error
        source = broken_source()
    else:
        getattr(encoder, failure).side_effect = error
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.encode_sequence(source, callback, tmp_path / "out.hevc", WIDTH, HEIGHT)
    encoder.close.assert_called_once_with()


@pytest.mark.parametrize("filename", ["out.hevc", "out.hevc.csv"])
def test_input_cannot_be_overwritten(encoder_stub, tmp_path, filename):
    runner, _, factory = encoder_stub
    source = tmp_path / filename
    source.write_bytes(bytes(FRAME_SIZE))
    with pytest.raises(ValueError, match="must differ"):
        runner.encode_sequence(source, None, tmp_path / "out.hevc", WIDTH, HEIGHT)
    assert source.read_bytes() == bytes(FRAME_SIZE)
    factory.assert_not_called()


def test_missing_native_library_is_explicit_and_preserves_output(monkeypatch, tmp_path):
    monkeypatch.delenv("PRISM_X265_SHIM", raising=False)
    monkeypatch.delenv("PRISM_X265_LIBRARY", raising=False)
    output = tmp_path / "out.hevc"
    output.write_bytes(b"existing")
    with pytest.raises(X265Unavailable, match="PRISM_X265_SHIM and PRISM_X265_LIBRARY"):
        VideoCodecRunner().encode_sequence([bytes(FRAME_SIZE)], None, output, WIDTH, HEIGHT)
    assert output.read_bytes() == b"existing"


class ShortReads(io.BytesIO):
    """Model a pipe which may return fewer bytes than requested."""

    def read(self, size=-1):
        return super().read(min(size, 5))


@pytest.fixture
def decoder_stub(monkeypatch):
    def install(payload, *, returncode=0, stderr=b"", read_error=None):
        process = Mock()
        process.stdout = ShortReads(payload)
        if read_error:
            process.stdout.read = Mock(side_effect=read_error)
        process.returncode = None

        def wait():
            process.returncode = returncode
            return returncode

        process.wait.side_effect = wait
        process.poll.side_effect = lambda: process.returncode

        def popen(command, **kwargs):
            kwargs["stderr"].write(stderr)
            return process

        factory = Mock(side_effect=popen)
        monkeypatch.setattr(video_codec.subprocess, "Popen", factory)
        return process, factory

    return install


def test_decode_streams_rgb_frames_and_handles_short_reads(decoder_stub, tmp_path):
    frames = np.arange(3 * 4 * 2 * 3, dtype=np.uint8).reshape(3, 2, 4, 3)
    process, factory = decoder_stub(frames.tobytes())
    path = tmp_path / "video with spaces.hevc"
    decoded = list(VideoCodecRunner().decode_bitstream(path, 4, 2))
    assert len(decoded) == 3
    for actual, expected in zip(decoded, frames):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.uint8
        assert actual.flags.writeable and actual.flags.c_contiguous
    command = factory.call_args.args[0]
    assert command[command.index("-i") + 1] == str(path.resolve())
    assert command[command.index("-pix_fmt") + 1] == "rgb24"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert process.stdout.closed
    assert factory.call_args.kwargs["stderr"].closed
    process.kill.assert_not_called()


@pytest.mark.parametrize("trailing_bytes", [1, 23])
def test_decode_rejects_partial_rgb_frame(decoder_stub, tmp_path, trailing_bytes):
    process, _ = decoder_stub(bytes(24 + trailing_bytes))
    frames = VideoCodecRunner().decode_bitstream(tmp_path / "out.hevc", 4, 2)
    assert next(frames).shape == (2, 4, 3)
    with pytest.raises(RuntimeError, match="partial RGB frame"):
        next(frames)
    assert process.stdout.closed


def test_decode_reports_ffmpeg_failure(decoder_stub, tmp_path):
    process, _ = decoder_stub(b"partial", returncode=1, stderr=b"invalid HEVC data")
    with pytest.raises(RuntimeError, match="FFmpeg decode failed.*invalid HEVC data"):
        list(VideoCodecRunner().decode_bitstream(tmp_path / "bad.hevc", 4, 2))
    assert process.stdout.closed


def test_closing_decode_iterator_stops_process(decoder_stub, tmp_path):
    process, _ = decoder_stub(bytes(72))
    frames = VideoCodecRunner().decode_bitstream(tmp_path / "out.hevc", 4, 2)
    next(frames)
    frames.close()
    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with()
    assert process.stdout.closed


def test_decode_read_failure_stops_process(decoder_stub, tmp_path):
    process, _ = decoder_stub(b"", read_error=OSError("pipe failed"))
    with pytest.raises(OSError, match="pipe failed"):
        list(VideoCodecRunner().decode_bitstream(tmp_path / "out.hevc", 4, 2))
    process.kill.assert_called_once_with()
    assert process.stdout.closed


def test_missing_ffmpeg_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(video_codec.subprocess, "Popen", Mock(side_effect=FileNotFoundError))
    with pytest.raises(RuntimeError, match="FFmpeg.*install it on PATH"):
        list(VideoCodecRunner().decode_bitstream(tmp_path / "out.hevc", 4, 2))
