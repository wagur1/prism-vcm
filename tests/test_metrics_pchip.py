from dataclasses import asdict
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from prism.evaluation.metrics import (RDPoint, bd_rate, cluster_bootstrap,
    holm_bonferroni, isotonic_pava_sensitivity, whole_bitstream_bytes)


def curve(scores=(0., 1., 2., 3., 4.), scale=1.):
    return [RDPoint(qp, score, int(rate * scale)) for qp, score, rate
            in zip((47, 42, 37, 32, 27), scores, (100, 200, 400, 800, 1600))]


def test_pchip_log_rate_known_reduction():
    result = bd_rate(curve(), curve(scale=0.9))
    assert result.valid and result.percent == pytest.approx(-10.)


def test_overlap_single_70_percent_threshold():
    assert bd_rate(curve(), curve((1.2, 2.2, 3.2, 4.2, 5.2))).valid
    invalid = bd_rate(curve(), curve((1.21, 2.21, 3.21, 4.21, 5.21)))
    assert not invalid.valid and invalid.percent is None
    assert not bd_rate(curve(), curve((10, 11, 12, 13, 14))).valid


@pytest.mark.parametrize("scores", [(0, 1, .5, 3, 4), (0, 1, 1, 3, 4), (0, 1, float('nan'), 3, 4)])
def test_invalid_raw_is_table_and_flag_not_nan(scores):
    result = bd_rate(curve(), curve(scores))
    assert result.flag == "[BD-INVALID-RAW]" and result.percent is None
    assert len(result.per_qp["candidate"]) == 5
    json.dumps(asdict(result), allow_nan=False)


def test_full_stream_byte_count(tmp_path):
    path = tmp_path / "stream.hevc"
    path.write_bytes(b"headerpayloadfooter")
    assert whole_bitstream_bytes(path) == whole_bitstream_bytes(path.read_bytes()) == 19


def test_cluster_bootstrap_and_holm():
    data = {seed: {"a": np.array([-3., -4.]), "b": np.array([-2., -5.])} for seed in (1, 2, 3)}
    result = cluster_bootstrap(data, unit="sequence")
    assert result["ci95"][1] < 0 and result["resamples"] == 1000
    with pytest.raises(ValueError):
        cluster_bootstrap(data, unit="sequence", resamples=999)
    with pytest.raises(ValueError):
        cluster_bootstrap({1: data[1]}, unit="sequence")
    results = holm_bonferroni([.001, .03, .04])
    assert [item["reject"] for item in results] == [True, False, False]


def test_pava_is_sensitivity_only():
    assert isotonic_pava_sensitivity([0., 2., 1., 3.]) == [0., 1.5, 1.5, 3.]
    assert not bd_rate(curve(), curve((0, 2, 1, 3, 4))).valid
