"""Header-compiled C shim for x265 quantOffsets, with explicit buffer ownership."""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .ffmpeg_stream import frame_bytes


class X265Unavailable(RuntimeError):
    """The pinned library/shim pair is unavailable."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qg_shape(width: int, height: int) -> tuple[int, int]:
    """QG16 on the CTU64-padded domain, including partial edge CTUs."""
    frame_bytes(width, height)
    return ((height + 63) // 64 * 4, (width + 63) // 64 * 4)


@dataclass(frozen=True)
class BuildPin:
    library_sha256: str
    shim_sha256: str
    version: str
    abi: int
    picture_size: int
    enable_assembly: bool = True


@dataclass(frozen=True)
class EncoderConfig:
    qp: int = 32
    preset: str = "medium"
    aq_mode: int = 1
    cutree: bool = False
    sao: bool = True
    qg_size: int = 16
    gop: int = 32
    bframes: int = 3
    fps: int = 30
    rc_mode: str = "CQP"
    crf: float = 28.0
    bitrate_kbps: int | None = None
    vbv_maxrate_kbps: int | None = None
    vbv_bufsize_kbits: int | None = None

    def options(self, csv_path: Path) -> str:
        if not 0 <= self.qp <= 51 or self.qg_size != 16:
            raise ValueError("Expected legal 8-bit QP and pinned QG16")
        if self.gop < 1 or self.bframes < 0:
            raise ValueError("Invalid GOP configuration")
        if self.rc_mode not in {"CQP", "CRF", "ABR", "VBV"}:
            raise ValueError("Unknown rate control mode")
        values = {"aq-mode": self.aq_mode, "cutree": int(self.cutree),
                  "sao": int(self.sao), "qg-size": 16, "keyint": self.gop,
                  "min-keyint": self.gop, "scenecut": 0, "bframes": self.bframes,
                  "fps": self.fps, "repeat-headers": 1, "annexb": 1,
                  "frame-threads": 1, "pools": "none", "wpp": 0,
                  "csv": str(csv_path), "csv-log-level": 2}
        if self.rc_mode == "CQP":
            values["qp"] = self.qp
        elif self.rc_mode == "CRF":
            if not 0 <= self.crf <= 51:
                raise ValueError("Invalid CRF")
            values["crf"] = self.crf
        else:
            if self.bitrate_kbps is None or self.bitrate_kbps <= 0:
                raise ValueError("ABR/VBV require positive target bitrate")
            values["bitrate"] = self.bitrate_kbps
            if self.rc_mode == "VBV":
                if not self.vbv_maxrate_kbps or not self.vbv_bufsize_kbits or min(
                        self.vbv_maxrate_kbps, self.vbv_bufsize_kbits) <= 0:
                    raise ValueError("VBV requires positive maxrate and buffer size")
                values["vbv-maxrate"] = self.vbv_maxrate_kbps
                values["vbv-bufsize"] = self.vbv_bufsize_kbits
        if any("\n" in str(value) for value in values.values()):
            raise ValueError("Options cannot contain newlines")
        return "\n".join(f"{key}={value}" for key, value in values.items())


class X265Library:
    def __init__(self, shim: str | Path | None = None,
                 library: str | Path | None = None, pin: BuildPin | None = None):
        shim = shim or os.environ.get("PRISM_X265_SHIM")
        library = library or os.environ.get("PRISM_X265_LIBRARY")
        if not shim or not library or not Path(shim).is_file() or not Path(library).is_file():
            raise X265Unavailable("Set PRISM_X265_SHIM and PRISM_X265_LIBRARY to local builds")
        self.shim_path = Path(shim).resolve()
        self.library_path = Path(library).resolve()
        self._dll_dir = (os.add_dll_directory(str(Path(library).resolve().parent))
                         if os.name == "nt" else None)
        try:
            self._library = ct.CDLL(str(Path(library).resolve()))
            self.api = ct.CDLL(str(Path(shim).resolve()))
        except OSError as exc:
            raise X265Unavailable(str(exc)) from exc
        self.api.prism_version.restype = ct.c_char_p
        self.api.prism_abi.restype = ct.c_int
        self.api.prism_picture_size.restype = ct.c_size_t
        self.api.prism_open.argtypes = [ct.c_int, ct.c_int, ct.c_char_p, ct.c_char_p]
        self.api.prism_open.restype = ct.c_void_p
        self.api.prism_close.argtypes = [ct.c_void_p]
        self.api.prism_close.restype = None
        self.api.prism_encode.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_float), ct.c_int, ct.c_int, ct.c_int64,
            ct.POINTER(ct.POINTER(ct.c_uint8)), ct.POINTER(ct.c_size_t),
            ct.POINTER(ct.c_int64)]
        self.api.prism_encode.restype = ct.c_int
        self.pin = BuildPin(file_sha256(library), file_sha256(shim),
                           self.api.prism_version().decode(), self.api.prism_abi(),
                           self.api.prism_picture_size())
        if pin is not None and pin != self.pin:
            raise ValueError(f"x265 build pin mismatch: {asdict(self.pin)}")


class X265Encoder:
    def __init__(self, library: X265Library, width: int, height: int,
                 config: EncoderConfig, csv_path: Path):
        self.width, self.height = width, height
        self.size = frame_bytes(width, height)
        self.library = library
        self._pending: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        self._next_pts = 0
        self._flushed = False
        self.handle = library.api.prism_open(width, height, config.preset.encode(),
                                            config.options(csv_path).encode())
        if not self.handle:
            raise RuntimeError("x265 rejected pinned parameters/header ABI")

    def encode(self, yuv: bytes, offsets: np.ndarray | None) -> bytes:
        if not self.handle or self._flushed:
            raise RuntimeError("Encoder is closed or flushed")
        if len(yuv) != self.size:
            raise ValueError("Incorrect YUV frame size")
        frame = np.frombuffer(yuv, dtype=np.uint8).copy()
        grid = None
        if offsets is not None:
            grid = np.array(offsets, dtype=np.float32, order="C", copy=True)
            if grid.shape != qg_shape(self.width, self.height) or not np.isfinite(grid).all():
                raise ValueError("Expected finite float32 row-major QG16 offsets on padded domain")
        pts = self._next_pts
        self._next_pts += 1
        self._pending[pts] = (frame, grid)
        return self._call(frame, grid, pts)[1]

    def _call(self, frame: np.ndarray | None, grid: np.ndarray | None,
              pts: int) -> tuple[int, bytes]:
        packet = ct.POINTER(ct.c_uint8)()
        length, out_pts = ct.c_size_t(), ct.c_int64()
        count = self.library.api.prism_encode(self.handle,
            None if frame is None else frame.ctypes.data_as(ct.POINTER(ct.c_uint8)),
            None if grid is None else grid.ctypes.data_as(ct.POINTER(ct.c_float)),
            self.width, self.height, pts, ct.byref(packet), ct.byref(length), ct.byref(out_pts))
        if count < 0:
            raise RuntimeError("x265 encode failed")
        payload = ct.string_at(packet, length.value) if length.value else b""
        if count:
            self._pending.pop(out_pts.value, None)
        return count, payload

    def flush(self) -> bytes:
        if not self.handle:
            raise RuntimeError("Encoder is closed")
        chunks: list[bytes] = []
        while True:
            count, payload = self._call(None, None, 0)
            chunks.append(payload)
            if not count:
                break
        self._flushed = True
        return b"".join(chunks)

    def close(self) -> None:
        if self.handle:
            self.library.api.prism_close(self.handle)
            self.handle = None
            self._pending.clear()

    def __enter__(self) -> X265Encoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
