import hashlib
import json
import sys

import pytest

np = pytest.importorskip("numpy")

from prism.codec.ffmpeg_stream import memory_yuv, stream_process
from prism.data.cached_dataset import CachedDataset
from prism.data.splits import SplitEntry, SplitLedger


def test_yuv_memory_pipe_without_raw_files():
    frames = [bytes([index]) * 24 for index in range(3)]
    assert memory_yuv(frames, 4, 4).read() == b"".join(frames)
    output = stream_process([sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
                            iter(frames), 4, 4)
    assert output == b"".join(frames)


def test_pipe_errors_are_propagated():
    with pytest.raises(ValueError):
        stream_process([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], [b"bad"], 4, 4)


def test_npz_cache_provenance_and_shapes(tmp_path):
    shard = tmp_path / "a.npz"
    np.savez(shard, x_low=np.zeros((3, 256, 256)), dino_features=np.zeros((2, 768)),
             sam_masks=np.zeros((2, 16, 16)), rois=np.zeros((2, 5)))
    entries = [SplitEntry("a", "a" * 64, "train")]
    draft = SplitLedger(entries)
    frozen = SplitLedger(entries, draft.digest)
    row = {"sample_id": "a", "source_sha256": "a" * 64, "shard": shard.name,
           "cache_sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"teachers": ["DINOv2-Base", "SAM 2.1-Base"],
                                "split_ledger_sha256": frozen.digest, "samples": [row]}))
    dataset = CachedDataset(index, frozen)
    assert dataset[0]["dino_features"].shape == (2, 768)
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        dataset[0]
