"""Atomic artifacts, verified resumable downloads and resource guards."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterator
import urllib.request


class StepFailure(RuntimeError):
    def __init__(self, step: str, reason: str):
        self.step = step
        super().__init__(f"step={step}: {reason}")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)


def _hash(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write("Exclusive Step 1 writer. Remove only after confirming the previous process stopped.\n")
    except FileExistsError as exc:
        raise StepFailure("lock", f"Writer lock exists: {path}") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def check_storage(root: Path, reserve_bytes: int = 0) -> int:
    """CONFIG uses decimal GB, including the entire working directory."""
    root.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    if used + reserve_bytes >= 16_000_000_000:
        raise StepFailure("storage", f"{used} bytes + {reserve_bytes} reserved violates working <16 GB")
    if shutil.disk_usage(root).free < reserve_bytes:
        raise StepFailure("storage", f"Insufficient free space for {reserve_bytes} bytes")
    return used


def download(url: str, path: Path, *, expected_sha256: str | None = None,
             retries: int = 3, opener: Callable[..., Any] = urllib.request.urlopen,
             sleep: Callable[[float], None] = time.sleep, offline: bool = False) -> dict:
    """First receipt is TOFU, not an upstream checksum. Subsequent reads verify it.

    Three retries means four total attempts. Partial files are never promoted.
    """
    receipt = path.with_name(path.name + ".sha256.json")
    known = read_json(receipt) if receipt.exists() else None
    if known and known["url"] != url:
        raise StepFailure("download", f"Receipt URL changed for {path}: {url}")
    expected = expected_sha256 or (known["sha256"] if known else None)
    if known and expected != known["sha256"]:
        raise StepFailure("download", f"Frozen/receipt SHA conflict: {url}")
    if path.exists():
        actual = sha256(path)
        if expected and actual != expected:
            raise StepFailure("download", f"SHA-256 mismatch at {path}; URL={url}")
        if not expected:
            raise StepFailure("download", f"Unverified pre-existing file {path}; supply its receipt; URL={url}")
        return {"status": "SKIPPED", "sha256": actual, "url": url}
    if offline:
        raise StepFailure("download", f"Offline cache miss: {path}; URL={url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(retries + 1):
        try:
            digest = hashlib.sha256()
            size = 0
            with opener(url, timeout=60) as response, partial.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    stream.write(block)
                length = response.headers.get("Content-Length")
                if not size or (length is not None and size != int(length)):
                    raise ValueError(f"Incomplete response: {size} bytes, expected {length}")
            actual = digest.hexdigest()
            if sha256(partial) != actual or (expected and actual != expected):
                raise ValueError("SHA-256 verification failed")
            # Receipt first: a crash before rename is safely resumable.
            write_json(receipt, {"url": url, "sha256": actual, "bytes": size,
                                 "verification": "first-receipt SHA256; no published upstream checksum"})
            partial.replace(path)
            return {"status": "CREATED", "sha256": actual, "url": url}
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise StepFailure("download", f"URL={url}; attempts={attempt + 1}; {exc}") from exc
            sleep(2 ** attempt)
    raise AssertionError("unreachable")
