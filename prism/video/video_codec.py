"""Stream YUV420p8 sequences through x265 and decode HEVC to RGB frames."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from prism.codec.ffmpeg_stream import frame_bytes
from prism.codec.x265_bridge import EncoderConfig, X265Encoder, X265Library


class VideoCodecRunner:
    """One encoder per sequence, preserving references and delayed B-frames.

    Defaults to CRF 28, GOP 32, three B-frames, 30 fps and the medium preset.
    Pass ``EncoderConfig(rc_mode="CRF", crf=crf, gop=32, bframes=0)`` for
    Low-Delay P, or ``bframes=3`` for Random-Access. Native x265 is loaded only
    when encoding; missing builds raise the bridge's ``X265Unavailable`` error.
    A preloaded (optionally build-pinned) library can be supplied explicitly.
    """

    def __init__(self, config: EncoderConfig | None = None, *,
                 library: X265Library | None = None):
        self.config = config if config is not None else EncoderConfig(rc_mode="CRF")
        self.library = library

    def encode_sequence(
        self,
        frames_yuv: Iterable[bytes] | Path,
        offset_generator: Callable[[int, int, int], np.ndarray | None] | None,
        output_bitstream: Path,
        width: int,
        height: int,
        fps: int = 30,
    ) -> dict:
        """Encode complete, tightly packed YUV420p8 frames in input order.

        A Path denotes a raw YUV file. The callback receives zero-based
        ``(frame_index, width, height)`` and returns padded QG16 offsets or None.
        The bridge validates and retains the maps until their frames emerge.
        ``fps`` sets both encoder timing and the reported duration, overriding
        config.fps for this call. Empty sequences and partial frames are errors.
        x265 telemetry is written beside the bitstream as ``<filename>.csv``.
        """
        size = frame_bytes(width, height)
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        output_bitstream = Path(output_bitstream)
        csv_path = output_bitstream.with_name(output_bitstream.name + ".csv")
        if isinstance(frames_yuv, Path) and frames_yuv.resolve() in (
                output_bitstream.resolve(), csv_path.resolve()):
            raise ValueError("Input YUV must differ from bitstream and telemetry paths")
        config = replace(self.config, fps=fps)
        config.options(csv_path)
        library = self.library if self.library is not None else X265Library()

        with ExitStack() as stack:
            if isinstance(frames_yuv, Path):
                source = stack.enter_context(frames_yuv.open("rb"))
                frames = iter(lambda: source.read(size), b"")
            else:
                frames = iter(frames_yuv)
            output_bitstream.parent.mkdir(parents=True, exist_ok=True)
            encoder = stack.enter_context(
                X265Encoder(library, width, height, config, csv_path))
            output = stack.enter_context(output_bitstream.open("wb"))
            num_frames = 0
            total_bytes = 0
            for yuv in frames:
                if len(yuv) != size:
                    raise ValueError(f"Incorrect YUV420p8 frame size at frame {num_frames}")
                offsets = (None if offset_generator is None else
                           offset_generator(num_frames, width, height))
                total_bytes += output.write(encoder.encode(yuv, offsets))
                num_frames += 1
            if not num_frames:
                raise ValueError("Cannot encode an empty sequence")
            # flush() drains all delayed frames internally, including B-frames.
            total_bytes += output.write(encoder.flush())

        return {
            "total_bytes": total_bytes,
            "num_frames": num_frames,
            "bitrate_kbps": (total_bytes * 8) / (num_frames / fps) / 1000.0,
            "bitstream_path": output_bitstream,
        }

    def decode_bitstream(self, bitstream_path: Path, width: int,
                         height: int) -> Iterator[np.ndarray]:
        """Yield writable uint8 RGB arrays of shape (height, width, 3).

        Dimensions must match the encoded video. FFmpeg emits display order
        without frame duplication. Frames stay in memory; stderr uses a temporary
        file so a full error pipe cannot deadlock a long decode. Close the
        iterator when stopping early to release the decoder immediately.
        """
        frame_bytes(width, height)
        size = width * height * 3
        command = [
            "ffmpeg", "-v", "error", "-xerror", "-nostdin", "-f", "hevc",
            "-i", str(Path(bitstream_path).resolve()), "-map", "0:v:0",
            "-an", "-sn", "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
            "-f", "rawvideo", "pipe:1",
        ]
        with tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=errors)
            except FileNotFoundError as exc:
                raise RuntimeError("FFmpeg is required for decoding; install it on PATH") from exc
            assert process.stdout is not None
            try:
                while True:
                    payload = bytearray()
                    while len(payload) < size:
                        chunk = process.stdout.read(size - len(payload))
                        if not chunk:
                            break
                        payload.extend(chunk)
                    if len(payload) < size:
                        break
                    yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
                if process.wait():
                    errors.seek(0)
                    detail = errors.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"FFmpeg decode failed (exit {process.returncode}): {detail}")
                if payload:
                    raise RuntimeError(f"FFmpeg returned a partial RGB frame: {len(payload)}/{size} bytes")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdout.close()
