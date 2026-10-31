"""Transport YUV420p8 through memory and anonymous OS pipes only."""
from __future__ import annotations

import io
import subprocess
import threading
from collections.abc import Iterable, Sequence


def frame_bytes(width: int, height: int) -> int:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("YUV420 dimensions must be positive and even")
    return width * height * 3 // 2


def memory_yuv(frames: Iterable[bytes], width: int, height: int) -> io.BytesIO:
    stream = io.BytesIO()
    for frame in frames:
        if len(frame) != frame_bytes(width, height):
            raise ValueError("Incorrect YUV420p8 frame size")
        stream.write(frame)
    stream.seek(0)
    return stream


def stream_process(command: Sequence[str], frames: Iterable[bytes],
                   width: int, height: int) -> bytes:
    """Drain stdout/stderr concurrently to avoid pipe deadlocks; return encoded bytes."""
    size = frame_bytes(width, height)
    proc = subprocess.Popen(list(command), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    errors: list[BaseException] = []

    def feed() -> None:
        assert proc.stdin is not None
        try:
            for frame in frames:
                if len(frame) != size:
                    raise ValueError("Incorrect YUV420p8 frame size")
                proc.stdin.write(frame)
        except BaseException as exc:
            errors.append(exc)
        finally:
            proc.stdin.close()

    worker = threading.Thread(target=feed, daemon=True)
    worker.start()
    # communicate must not attempt to flush the producer's stdin.
    output: list[bytes] = []
    assert proc.stdout is not None and proc.stderr is not None
    reader = threading.Thread(target=lambda: output.append(proc.stdout.read()), daemon=True)
    reader.start()
    stderr = proc.stderr.read()
    worker.join()
    reader.join()
    proc.wait()
    proc.stdout.close()
    proc.stderr.close()
    if errors:
        raise errors[0]
    if proc.returncode:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))
    return output[0]
