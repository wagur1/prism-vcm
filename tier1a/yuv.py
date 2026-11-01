"""Lossless gzip storage of BT.709 limited-range YUV420p8; raw bytes stay in RAM."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np

from prism.codec.ffmpeg_stream import frame_bytes, memory_yuv


def letterbox_yuv(rgb: np.ndarray, width: int = 1920, height: int = 1080) -> tuple[bytes, dict]:
    from PIL import Image

    frame_bytes(width, height)  # Validate even dimensions using the codec contract.
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB [H,W,3]")
    source_h, source_w = rgb.shape[:2]
    scale = min(width / source_w, height / source_h)
    # Even extents and offsets align chroma with the active picture.
    resized_w = min(width, max(2, int(source_w * scale) // 2 * 2))
    resized_h = min(height, max(2, int(source_h * scale) // 2 * 2))
    left, top = ((width - resized_w) // 2 // 2 * 2, (height - resized_h) // 2 // 2 * 2)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = np.asarray(
        Image.fromarray(rgb).resize((resized_w, resized_h), Image.Resampling.LANCZOS))
    r, g, b = (canvas[..., index].astype(np.float32) for index in range(3))
    y = 16 + (219 / 255) * (.2126 * r + .7152 * g + .0722 * b)
    u = 128 + (224 / 255) * (-.114572 * r - .385428 * g + .5 * b)
    v = 128 + (224 / 255) * (.5 * r - .454153 * g - .045847 * b)
    planes = [np.clip(np.rint(y), 16, 235).astype(np.uint8)]
    for chroma in (u, v):
        reduced = chroma.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
        planes.append(np.clip(np.rint(reduced), 16, 240).astype(np.uint8))
    payload = b"".join(plane.tobytes() for plane in planes)
    memory_yuv([payload], width, height)  # Same validation as the encode IPC path.
    geometry = {"width": width, "height": height, "pixel_format": "yuv420p",
                "matrix": "bt709", "range": "limited", "chroma": "2x2 box average",
                "source_width": source_w, "source_height": source_h,
                "left": left, "top": top, "active_width": resized_w, "active_height": resized_h,
                "scale_x": resized_w / source_w, "scale_y": resized_h / source_h,
                "raw_bytes": len(payload), "raw_sha256": hashlib.sha256(payload).hexdigest()}
    return payload, geometry


def save_yuv(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6) as compressed:
            compressed.write(payload)
    temporary.replace(path)


def load_yuv(path: Path, width: int = 1920, height: int = 1080) -> bytes:
    with gzip.open(path, "rb") as stream:
        payload = stream.read(frame_bytes(width, height) + 1)
    return memory_yuv([payload], width, height).getvalue()
