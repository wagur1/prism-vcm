"""Synthetic MOT17 contracts; no dataset, detector, codec or TrackEval needed."""
import math

import numpy as np
from PIL import Image
import pytest

from prism.video.mot_eval import (
    MOT17_VALIDATION_SEQUENCES, MOTSequence, SimpleByteTracker,
    compute_mot_metrics, format_mot_line,
)


EMPTY = np.empty((0, 5))
METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA")


@pytest.fixture
def sequence_dir(tmp_path):
    sequence = tmp_path / "MOT17-02-FRCNN"
    (sequence / "frames").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=frames\nframeRate=30\n"
        "seqLength=3\nimWidth=8\nimHeight=4\nimExt=.png\n", encoding="utf-8")
    for index, color in enumerate(("black", "white", "red"), 1):
        Image.new("RGB", (8, 4), color).save(sequence / "frames" / f"{index:06d}.png")
    return sequence


def test_seqinfo_and_frame_order(sequence_dir):
    sequence = MOTSequence(sequence_dir)
    assert (sequence.seq_name, sequence.im_width, sequence.im_height,
            sequence.seq_length, sequence.frame_rate) == ("MOT17-02-FRCNN", 8, 4, 3, 30)
    # Extra images do not alter the declared temporal interval.
    (sequence_dir / "frames" / "000004.png").touch()
    assert [p.name for p in sequence.iter_frame_paths()] == [
        "000001.png", "000002.png", "000003.png"]
    assert MOT17_VALIDATION_SEQUENCES == (
        "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13")


def test_seqinfo_missing_file_and_invalid_metadata(sequence_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        MOTSequence(tmp_path / "absent")
    path = sequence_dir / "seqinfo.ini"
    path.write_text(path.read_text().replace("seqLength=3", "seqLength=0"))
    with pytest.raises(ValueError, match="metadata"):
        MOTSequence(sequence_dir)


def test_missing_frame_does_not_silently_shorten_sequence(sequence_dir):
    (sequence_dir / "frames" / "000002.png").unlink()
    frames = MOTSequence(sequence_dir).iter_frame_paths()
    assert next(frames).name == "000001.png"
    with pytest.raises(FileNotFoundError, match="Missing frame 2"):
        next(frames)


def test_yuv_frame_size_plane_order_range_and_resize(sequence_dir):
    frames = list(MOTSequence(sequence_dir).iter_yuv_frames(4, 2))
    assert len(frames) == 3
    assert all(isinstance(frame, bytes) and len(frame) == 12 for frame in frames)
    assert frames[0] == bytes([16] * 8 + [128] * 4)
    assert frames[1] == bytes([235] * 8 + [128] * 4)
    # BT.709 limited-range red: Y=63, Cb=102, Cr=240 (not BT.601).
    assert frames[2] == bytes([63] * 8 + [102] * 2 + [240] * 2)


@pytest.mark.parametrize("width,height", [(3, 2), (4, 3), (0, 2), (-2, 2)])
def test_yuv_invalid_dimensions(sequence_dir, width, height):
    with pytest.raises(ValueError, match="dimensions"):
        list(MOTSequence(sequence_dir).iter_yuv_frames(width, height))


def test_yuv_rejects_metadata_dimension_mismatch(sequence_dir):
    Image.new("RGB", (4, 4)).save(sequence_dir / "frames" / "000001.png")
    with pytest.raises(ValueError, match="seqinfo.ini"):
        next(MOTSequence(sequence_dir).iter_yuv_frames(4, 2))


def detection(x=0, score=0.9):
    return np.array([[x, 0, x + 12, 12, score]], dtype=float)


def test_track_creation_order_independence_and_continuity():
    tracker = SimpleByteTracker()
    first = tracker.update(np.concatenate([detection(), detection(100)]))
    assert first[:, 4].tolist() == [1, 2]
    second = tracker.update(np.concatenate([detection(102), detection(2)]))
    assert second[:, 4].tolist() == [1, 2]
    np.testing.assert_array_equal(second[:, 0], [2, 102])
    assert SimpleByteTracker().update(detection())[0, 4] == 1


def test_low_confidence_rescues_active_track_without_creating_ids():
    tracker = SimpleByteTracker()
    assert tracker.update(detection(score=0.2)).shape == (0, 5)
    assert tracker.update(detection())[0, 4] == 1
    recovered = tracker.update(np.concatenate([detection(2, 0.2), detection(100, 0.2)]))
    np.testing.assert_array_equal(recovered, [[2, 0, 14, 12, 1]])
    assert tracker.update(detection(4))[0, 4] == 1


def test_high_confidence_association_has_priority():
    tracker = SimpleByteTracker()
    tracker.update(detection())
    actual = tracker.update(np.concatenate([detection(2, 0.8), detection(0, 0.4)]))
    np.testing.assert_array_equal(actual, [[2, 0, 14, 12, 1]])


def test_motion_prediction_and_occlusion_buffer():
    tracker = SimpleByteTracker(max_age=2)
    tracker.update(detection(0))
    tracker.update(detection(3))
    assert tracker.update(EMPTY).shape == (0, 5)
    assert tracker.update(EMPTY).shape == (0, 5)
    # Last observed box at x=3 has no gated overlap at x=12. Motion is needed.
    assert tracker.update(detection(12))[0, 4] == 1


def test_termination_never_reuses_ids():
    tracker = SimpleByteTracker(max_age=2)
    tracker.update(detection())
    for _ in range(3):
        assert tracker.update(EMPTY).shape == (0, 5)
    assert tracker.update(detection())[0, 4] == 2


def test_lost_track_cannot_be_reactivated_by_low_confidence():
    tracker = SimpleByteTracker(max_age=2)
    tracker.update(detection())
    tracker.update(EMPTY)
    assert tracker.update(detection(score=0.3)).shape == (0, 5)
    assert tracker.update(detection())[0, 4] == 1


def test_strict_iou_threshold_and_confidence_boundaries():
    tracker = SimpleByteTracker()
    assert tracker.update(detection(score=0.6))[0, 4] == 1
    # A four-pixel displacement of a 12-pixel box gives exactly IoU=.5.
    assert tracker.update(detection(4))[0, 4] == 2
    tracker = SimpleByteTracker()
    tracker.update(detection())
    assert tracker.update(detection(score=0.1))[0, 4] == 1
    assert tracker.update(detection(score=0.099)).shape == (0, 5)


@pytest.mark.parametrize("bad", [np.empty((0, 4)), np.zeros(5),
                                    [[0, 0, 0, 1, 0.9]], [[0, 0, 1, 1, np.nan]],
                                    [[0, 0, 1, 1, 1.1]]])
def test_invalid_detections_do_not_advance_tracker(bad):
    tracker = SimpleByteTracker(max_age=0)
    tracker.update(detection())
    with pytest.raises(ValueError):
        tracker.update(bad)
    assert tracker.update(detection())[0, 4] == 1


@pytest.mark.parametrize("kwargs", [{"max_age": -1}, {"max_age": 0.5},
                                        {"high_thresh": 0.1}, {"match_thresh": 1},
                                        {"low_thresh": float("nan")}])
def test_tracker_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SimpleByteTracker(**kwargs)


def test_format_mot_line():
    line = format_mot_line(3, 7, [10.25, 20.5, 40.75, 60], 0.875)
    assert line == "3,7,10.250000,20.500000,30.500000,39.500000,0.875000,-1,-1,-1"
    assert len(line.split(",")) == 10
    assert format_mot_line(1, 1, np.array([0, 0, 10, 20])).split(",")[6] == "1.000000"


@pytest.mark.parametrize("frame,identity,box", [(0, 1, [0, 0, 1, 1]),
                                                  (1, 1.5, [0, 0, 1, 1]),
                                                  (1, 1, [0, 0, -1, 1])])
def test_format_invalid_inputs(frame, identity, box):
    with pytest.raises(ValueError):
        format_mot_line(frame, identity, box)


def row(frame, identity=1, x=0, mark=1, cls=1):
    return (frame, identity, x, 0, 12, 12, mark, cls, 1)


def ground_truth(tmp_path, rows):
    path = tmp_path / "gt.txt"
    path.write_text("\n".join(",".join(map(str, item)) for item in rows), encoding="utf-8")
    return path


def test_perfect_metrics_from_file_and_unsorted_rows(tmp_path):
    gt = ground_truth(tmp_path, [row(1), row(2)])
    predictions = [row(2, 19), row(1, 19)]
    from_rows = compute_mot_metrics(predictions, gt)
    path = tmp_path / "predictions.txt"
    path.write_text("\n".join(format_mot_line(t, 19, [0, 0, 12, 12]) for t in (1, 2)))
    assert compute_mot_metrics(path, gt) == from_rows
    assert all(from_rows[key] == pytest.approx(100) for key in METRICS)
    assert (from_rows["TP"], from_rows["FP"], from_rows["FN"], from_rows["IDSW"]) == (2, 0, 0, 0)


def test_identity_switch_penalizes_association_metrics(tmp_path):
    gt = ground_truth(tmp_path, [row(1), row(2)])
    result = compute_mot_metrics([row(1, 10), row(2, 20)], gt)
    assert result["MOTA"] == 50
    assert result["IDF1"] == 50
    assert result["DetA"] == 100
    assert result["AssA"] == 50
    assert result["HOTA"] == pytest.approx(100 / math.sqrt(2))
    assert result["IDSW"] == 1


def test_misses_false_positives_and_id_switch_across_gap(tmp_path):
    gt = ground_truth(tmp_path, [row(1), row(2), row(3)])
    result = compute_mot_metrics([row(1, 10), row(3, 20), row(4, 30, 100)], gt)
    assert (result["TP"], result["FP"], result["FN"], result["IDSW"]) == (2, 1, 1, 1)
    assert result["MOTA"] == 0
    assert result["IDF1"] == pytest.approx(100 / 3)
    assert result["DetA"] == 50
    assert result["AssA"] == pytest.approx(100 / 3)
    assert result["HOTA"] == pytest.approx(100 / math.sqrt(6))


def test_hota_averages_all_thresholds_and_clear_includes_iou_half(tmp_path):
    gt = ground_truth(tmp_path, [row(1)])
    result = compute_mot_metrics([row(1, x=4)], gt)  # Exactly .5 IoU.
    assert result["MOTA"] == result["IDF1"] == 100
    for key in ("HOTA", "DetA", "AssA"):
        assert result[key] == pytest.approx(100 * 10 / 19)


def test_clear_preserves_previous_matches_even_when_other_iou_is_larger(tmp_path):
    gt = ground_truth(tmp_path, [row(t, identity, x) for t in (1, 2)
                               for identity, x in ((1, 0), (2, 2))])
    predictions = [row(1, 10, 0), row(1, 20, 2), row(2, 10, 2), row(2, 20, 0)]
    result = compute_mot_metrics(predictions, gt)
    assert result["MOTA"] == 100
    assert result["IDSW"] == 0
    # Identity uses all threshold-valid pairs, not just per-frame CLEAR matches.
    assert result["IDF1"] == 100


def test_mot17_distractors_and_zero_marked_pedestrians(tmp_path):
    truth = [row(1)] + [row(1, cls, cls * 100, cls=cls) for cls in (2, 7, 8, 12)]
    truth += [row(1, 50, 5000, mark=0), row(1, 60, 6000, cls=3)]
    gt = ground_truth(tmp_path, truth)
    result = compute_mot_metrics(truth, gt)
    # Distractor matches disappear; zero-marked people and vehicles only remove
    # GT, so their predictions remain FPs under MOT17/TrackEval preprocessing.
    assert (result["num_gt"], result["num_predictions"]) == (1, 3)
    assert (result["TP"], result["FP"], result["FN"]) == (1, 2, 0)
    assert result["MOTA"] == -100
    assert result["IDF1"] == 50


def test_distractor_suppression_is_one_to_one(tmp_path):
    gt = ground_truth(tmp_path, [row(1, cls=8)])
    result = compute_mot_metrics([row(1, 10), row(1, 20)], gt)
    assert result["num_gt"] == 0
    assert result["FP"] == 1


@pytest.mark.parametrize("truth,predictions", [([], []), ([row(1)], []), ([], [row(1)])])
def test_empty_inputs_have_finite_zero_scores(tmp_path, truth, predictions):
    result = compute_mot_metrics(predictions, ground_truth(tmp_path, truth))
    assert all(result[key] == 0 for key in METRICS)
    assert result["FN"] == len(truth)
    assert result["FP"] == len(predictions)


def test_missing_ground_truth_is_explicit(tmp_path):
    result = compute_mot_metrics([row(1)], None)
    assert result["evaluated"] is False
    assert all(result[key] is None for key in METRICS)
    assert result["num_predictions"] == 1
    with pytest.raises(FileNotFoundError):
        compute_mot_metrics([], tmp_path / "missing.txt")


@pytest.mark.parametrize("bad", [[row(1), row(1)], [(0, 1, 0, 0, 12, 12)],
                                    [(1, 1, 0, 0, -12, 12)], [(1, 1, np.nan, 0, 12, 12)],
                                    [(1, 1, 0)]])
def test_metrics_reject_invalid_predictions(tmp_path, bad):
    with pytest.raises(ValueError):
        compute_mot_metrics(bad, ground_truth(tmp_path, [row(1)]))


def test_sparse_large_ids_and_classless_mot_rows(tmp_path):
    gt = ground_truth(tmp_path, [(1, 1000000000, 0, 0, 12, 12)])
    result = compute_mot_metrics([(1, 2000000000, 0, 0, 12, 12)], gt)
    assert result["HOTA"] == result["IDF1"] == 100


def test_tracker_to_mot_to_metrics_pipeline(tmp_path):
    tracker = SimpleByteTracker()
    gt = ground_truth(tmp_path, [row(t, x=2 * (t - 1)) for t in range(1, 5)])
    lines = []
    for frame, score in enumerate((0.9, 0.9, 0.2, 0.9), 1):
        for track in tracker.update(detection(2 * (frame - 1), score)):
            lines.append(format_mot_line(frame, int(track[4]), track[:4]))
    prediction_path = tmp_path / "tracks.txt"
    prediction_path.write_text("\n".join(lines))
    result = compute_mot_metrics(prediction_path, gt)
    assert all(result[key] == 100 for key in METRICS)
