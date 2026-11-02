"""Unit tests for Tier 1b full 5,000-image sharded evaluation pipeline.

Verifies:
  1. Deterministic disjoint partitioning of 5,000 images into 19 shards.
  2. Observation validation and structure integrity.
  3. Dispatch script kernel workspace generation.
  4. Shard merger and aggregation integrity.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from tier1b.run_tier1b_val5k import shard_images, validate_crfs, validate_observations, observation_keys
from scripts.dispatch_pool_5k import generate_shard_kernel, make_shard_runner_script
from scripts.merge_pool_5k import merge_shard_data


def test_shard_images_5000_into_19_disjoint_partitions():
    """Test that 5,000 images partitioned into 19 shards is strictly disjoint and complete."""
    # Generate 5,000 mock COCO val2017 image records
    all_images = [
        {"id": i, "file_name": f"{i:012d}.jpg", "width": 640, "height": 480}
        for i in range(1, 5001)
    ]
    num_shards = 19
    shards = []
    all_shard_ids = []

    for s_idx in range(num_shards):
        shard = shard_images(all_images, s_idx, num_shards)
        shards.append(shard)
        # 5000 / 19 = 263.15 -> each shard must be either 263 or 264
        assert len(shard) in (263, 264), f"Shard {s_idx} had unexpected length {len(shard)}"
        shard_ids = [img["id"] for img in shard]
        all_shard_ids.extend(shard_ids)

    # 1. Complete coverage: exactly 5,000 images
    assert len(all_shard_ids) == 5000

    # 2. Strict pairwise disjointness (no duplicate images across any shard)
    unique_ids = set(all_shard_ids)
    assert len(unique_ids) == 5000
    assert unique_ids == {img["id"] for img in all_images}


def test_shard_images_error_cases():
    """Test boundary conditions and error handling in shard_images."""
    # Invalid shard_idx
    with pytest.raises(ValueError, match="0 <= shard_idx < num_shards"):
        shard_images([{"id": 1}], shard_idx=-1, num_shards=19)
    with pytest.raises(ValueError, match="0 <= shard_idx < num_shards"):
        shard_images([{"id": 1}], shard_idx=19, num_shards=19)

    # Invalid num_shards
    with pytest.raises(ValueError, match="num_shards > 0"):
        shard_images([{"id": 1}], shard_idx=0, num_shards=0)

    # Duplicate IDs
    with pytest.raises(ValueError, match="Duplicate image IDs"):
        shard_images([{"id": 1}, {"id": 1}], shard_idx=0, num_shards=2)


def test_validate_crfs():
    """Test CRF validation logic."""
    assert validate_crfs([36, 20, 24, 28, 32]) == [20, 24, 28, 32, 36]
    with pytest.raises(ValueError, match="CRFs must be distinct"):
        validate_crfs([20, 24, 24])
    with pytest.raises(ValueError, match="CRFs must be distinct"):
        validate_crfs([])
    with pytest.raises(ValueError, match="CRFs must be distinct"):
        validate_crfs([20, 55])


def test_validate_observations_success_and_failure():
    """Test observation validation rejects corrupted or incomplete entries."""
    image_ids = [100]
    crfs = [20]
    category_ids = {1, 2, 3}

    # Valid observation structure
    valid_obs = [
        # M1 (seed=None)
        {
            "image_id": 100, "model": "M1", "seed": None, "crf": 20,
            "whole_bitstream_bytes": 1000, "bits": 8000,
            "yolo11l": {"ap50": 0.85, "rows": [{"image_id": 100, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.9}]},
            "faster_rcnn": {"ap50": 0.80, "rows": []},
        },
    ]
    for m in ("M2", "M3", "M4"):
        for seed in (0, 1, 2):
            valid_obs.append({
                "image_id": 100, "model": m, "seed": seed, "crf": 20,
                "whole_bitstream_bytes": 1000, "bits": 8000,
                "yolo11l": {"ap50": 0.85, "rows": []},
                "faster_rcnn": {"ap50": 0.80, "rows": []},
            })

    validate_observations(valid_obs, image_ids, crfs, category_ids)

    # Corrupt bits != whole_bitstream_bytes * 8
    corrupt_obs = [dict(valid_obs[0], bits=9999)] + valid_obs[1:]
    with pytest.raises(ValueError, match="Invalid whole-bitstream size"):
        validate_observations(corrupt_obs, image_ids, crfs, category_ids)

    # Missing observations
    with pytest.raises(ValueError, match="Missing"):
        validate_observations(valid_obs[:-1], image_ids, crfs, category_ids)


def test_kernel_generation(tmp_path: Path):
    """Test generating kernel workspace metadata and runner script."""
    shard_dir = tmp_path / "shard_3"
    res = generate_shard_kernel(3, 19, "test_user", shard_dir)
    assert res == shard_dir

    metadata_path = shard_dir / "kernel-metadata.json"
    assert metadata_path.is_file()
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["id"] == "test_user/prism-tier1b-shard3"
    assert meta["enable_gpu"] is True
    assert "awsaf49/coco-2017-dataset" in meta["dataset_sources"]
    assert "htran123456/prism-tier1a-step3" in meta["kernel_sources"]

    runner_path = shard_dir / "tier1b_shard.py"
    assert runner_path.is_file()
    content = runner_path.read_text(encoding="utf-8")
    assert "SHARD_IDX = 3" in content
    assert "NUM_SHARDS = 19" in content


def test_merge_shard_data(tmp_path: Path):
    """Test merging multiple shard report files."""
    shard0 = tmp_path / "shard_0_report.json"
    shard1 = tmp_path / "shard_1_report.json"

    shard0_data = {
        "shard_idx": 0,
        "image_ids": [1, 3],
        "observations": [
            {"image_id": 1, "model": "M1", "seed": None, "crf": 20, "whole_bitstream_bytes": 100},
            {"image_id": 3, "model": "M1", "seed": None, "crf": 20, "whole_bitstream_bytes": 200},
        ]
    }
    shard1_data = {
        "shard_idx": 1,
        "image_ids": [2, 4],
        "observations": [
            {"image_id": 2, "model": "M1", "seed": None, "crf": 20, "whole_bitstream_bytes": 150},
            {"image_id": 4, "model": "M1", "seed": None, "crf": 20, "whole_bitstream_bytes": 250},
        ]
    }
    shard0.write_text(json.dumps(shard0_data), encoding="utf-8")
    shard1.write_text(json.dumps(shard1_data), encoding="utf-8")

    reports = {0: shard0, 1: shard1}
    summary, all_obs, image_ids = merge_shard_data(reports, expected_num_shards=2)
    assert summary["shards_found"] == 2
    assert summary["total_unique_images"] == 4
    assert image_ids == [1, 2, 3, 4]
    assert len(all_obs) == 4

    # Overlapping images should raise ValueError
    shard1_data_dup = {
        "shard_idx": 1,
        "image_ids": [1, 4],  # 1 is already in shard 0
        "observations": []
    }
    shard1.write_text(json.dumps(shard1_data_dup), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping images"):
        merge_shard_data(reports, expected_num_shards=2)
