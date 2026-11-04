"""MOT17 frame loading, two-stage tracking, and single-sequence evaluation.

Metrics follow MOTChallenge/TrackEval pedestrian preprocessing and metric
definitions. Scores use the MOTChallenge percentage scale, not fractions.
Evaluate each sequence independently (with a fresh tracker) before aggregating
the seven validation sequences for bitrate curves.
"""
from __future__ import annotations

from collections.abc import Iterator
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from prism.codec.ffmpeg_stream import frame_bytes


MOT17_VALIDATION_SEQUENCES = tuple(
    f"MOT17-{number:02d}" for number in (2, 4, 5, 9, 10, 11, 13))
_EPS = np.finfo(float).eps


class MOTSequence:
    """Read one MOT sequence, including detector-suffixed directory names.

    Frame paths are one-based, six-digit filenames in ``imDir`` with ``imExt``.
    Missing frames raise instead of silently shortening the evaluated sequence.
    Pillow (the existing ``cache`` extra) is needed only for YUV conversion.
    """

    def __init__(self, sequence_dir: Path | str):
        self.sequence_dir = Path(sequence_dir)
        info = self.sequence_dir / "seqinfo.ini"
        parser = ConfigParser(interpolation=None)
        with info.open(encoding="utf-8-sig") as stream:
            parser.read_file(stream)
        try:
            section = parser["Sequence"]
            self.seq_name = section["name"]
            self.im_width = section.getint("imWidth")
            self.im_height = section.getint("imHeight")
            self.seq_length = section.getint("seqLength")
            self.frame_rate = section.getint("frameRate")
            self.im_dir = section.get("imDir", "img1")
            self.im_ext = section.get("imExt", ".jpg")
            if (not self.seq_name or not self.im_ext.startswith(".") or
                    any(value is None or value <= 0 for value in (
                        self.im_width, self.im_height, self.seq_length, self.frame_rate))):
                raise ValueError("name, extension and positive dimensions/timing are required")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid sequence metadata in {info}: {exc}") from exc

    def iter_frame_paths(self) -> Iterator[Path]:
        for index in range(1, self.seq_length + 1):
            path = self.sequence_dir / self.im_dir / f"{index:06d}{self.im_ext}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frame {index}: {path}")
            yield path

    def iter_yuv_frames(self, width: int, height: int) -> Iterator[bytes]:
        """Yield packed BT.709 limited-range YUV420p8 for VideoCodecRunner.

        Images are resized directly to the requested even dimensions. Map
        decoded tracking boxes back by (im_width/width, im_height/height) before
        comparing against original-coordinate GT. Chroma uses 2x2 box averaging.
        """
        frame_bytes(width, height)
        from PIL import Image

        for path in self.iter_frame_paths():
            with Image.open(path) as image:
                if image.size != (self.im_width, self.im_height):
                    raise ValueError(f"Frame dimensions disagree with seqinfo.ini: {path}")
                rgb = np.asarray(image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS), dtype=np.float32)
            r, g, b = rgb.transpose(2, 0, 1)
            y = 16 + (219 / 255) * (.2126 * r + .7152 * g + .0722 * b)
            u = 128 + (224 / 255) * (-.114572 * r - .385428 * g + .5 * b)
            v = 128 + (224 / 255) * (.5 * r - .454153 * g - .045847 * b)
            planes = [np.clip(np.rint(y), 16, 235).astype(np.uint8)]
            for chroma in (u, v):
                reduced = chroma.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
                planes.append(np.clip(np.rint(reduced), 16, 240).astype(np.uint8))
            yield b"".join(plane.tobytes() for plane in planes)


def _ious(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Continuous-coordinate box IoU (no inclusive-pixel +1)."""
    intersection = np.maximum(0, np.minimum(left[:, None, 2:], right[None, :, 2:]) -
                              np.maximum(left[:, None, :2], right[None, :, :2])).prod(axis=2)
    areas_left = np.maximum(0, left[:, 2:] - left[:, :2]).prod(axis=1)
    areas_right = np.maximum(0, right[:, 2:] - right[:, :2]).prod(axis=1)
    union = areas_left[:, None] + areas_right[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    velocity: np.ndarray
    last_seen: int

    def predict(self, frame: int) -> np.ndarray:
        return self.bbox + self.velocity * (frame - self.last_seen)

    def observe(self, bbox: np.ndarray, frame: int) -> None:
        # Translation-only constant velocity preserves positive box dimensions.
        old_center = (self.bbox[:2] + self.bbox[2:]) / 2
        new_center = (bbox[:2] + bbox[2:]) / 2
        self.velocity = np.tile((new_center - old_center) / (frame - self.last_seen), 2)
        self.bbox = bbox.copy()
        self.last_seen = frame


class SimpleByteTracker:
    """ByteTrack-style high/low association with constant-velocity motion.

    High-confidence detections match active and buffered lost tracks first;
    low-confidence detections rescue only still-active unmatched tracks and
    never create IDs. New high-confidence detections activate immediately.
    This lightweight variant uses linear motion rather than ByteTrack's Kalman
    filter and omits appearance features, score fusion and tentative tracks.

    ``max_age`` is the number of consecutive missed frames to retain a track.
    Only tracks observed on the current frame are returned. Call update even on
    empty frames; IDs start at 1 and are never reused within a tracker instance.
    """

    def __init__(self, high_thresh: float = 0.6, low_thresh: float = 0.1,
                 match_thresh: float = 0.5, max_age: int = 30):
        if not 0 <= low_thresh < high_thresh <= 1:
            raise ValueError("Require 0 <= low_thresh < high_thresh <= 1")
        if not 0 <= match_thresh < 1:
            raise ValueError("match_thresh must be in [0, 1)")
        if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 0:
            raise ValueError("max_age must be a nonnegative integer")
        self.high_thresh, self.low_thresh = high_thresh, low_thresh
        self.match_thresh, self.max_age = match_thresh, max_age
        self._tracks: list[_Track] = []
        self._frame = 0
        self._next_id = 1

    def _associate(self, tracks: list[_Track], detections: np.ndarray) -> set[int]:
        if not tracks or not len(detections):
            return set()
        overlaps = _ious(np.array([track.predict(self._frame) for track in tracks]),
                         detections[:, :4])
        valid = overlaps > self.match_thresh
        # Prioritize the number of valid matches, then total IoU. Gating before
        # assignment prevents invalid edges from stealing valid associations.
        scores = np.where(valid, min(overlaps.shape) + 1 + overlaps, 0)
        rows, cols = linear_sum_assignment(-scores)
        matched = set()
        for row, col in zip(rows, cols):
            if valid[row, col]:
                tracks[row].observe(detections[col, :4], self._frame)
                matched.add(int(col))
        return matched

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Accept [N,5] xyxy/score, return [M,5] xyxy/integer-valued ID.

        Output is float64 to hold coordinates and IDs together, including an
        empty (0,5) array. Invalid detections raise before advancing tracker time.
        """
        detections = np.asarray(detections, dtype=float)
        if detections.ndim != 2 or detections.shape[1] != 5:
            raise ValueError("detections must have shape [N, 5]")
        if (not np.isfinite(detections).all() or
                np.any(detections[:, 2:4] <= detections[:, :2]) or
                np.any((detections[:, 4] < 0) | (detections[:, 4] > 1))):
            raise ValueError("detections require finite positive-area boxes and scores in [0, 1]")
        self._frame += 1
        high = detections[detections[:, 4] >= self.high_thresh]
        low = detections[(detections[:, 4] >= self.low_thresh) &
                         (detections[:, 4] < self.high_thresh)]
        matched_high = self._associate(self._tracks, high)
        active_unmatched = [track for track in self._tracks
                            if track.last_seen == self._frame - 1]
        self._associate(active_unmatched, low)
        self._tracks = [track for track in self._tracks
                        if self._frame - track.last_seen <= self.max_age]
        for index, detection in enumerate(high):
            if index not in matched_high:
                self._tracks.append(_Track(self._next_id, detection[:4].copy(),
                                           np.zeros(4), self._frame))
                self._next_id += 1
        return np.array([np.r_[track.bbox, track.track_id] for track in self._tracks
                         if track.last_seen == self._frame], dtype=float).reshape(-1, 5)


def format_mot_line(frame_idx: int, track_id: int,
                    bbox_xyxy: tuple | list | np.ndarray, score: float = 1.0) -> str:
    """Return a ten-column MOT row without a newline or coordinate-origin shift.

    Frame numbers start at 1; the caller must use the same spatial origin as GT.
    Coordinates/confidence are written with six decimal places.
    """
    box = np.asarray(bbox_xyxy, dtype=float)
    if (not np.isfinite(frame_idx) or int(frame_idx) != frame_idx or frame_idx < 1 or
            not np.isfinite(track_id) or int(track_id) != track_id or track_id < 0):
        raise ValueError("Require a positive integer frame and a nonnegative integer ID")
    if (box.shape != (4,) or not np.isfinite(box).all() or
            np.any(box[2:] <= box[:2]) or not np.isfinite(score)):
        raise ValueError("Require a finite positive-area xyxy box and finite score")
    x, y = box[:2]
    width, height = box[2:] - box[:2]
    return (f"{int(frame_idx)},{int(track_id)},{x:.6f},{y:.6f},{width:.6f},{height:.6f},"
            f"{score:.6f},-1,-1,-1")


def _read_mot(source: Path | str | list[tuple], *, ground_truth: bool) -> dict[int, np.ndarray]:
    """Normalize MOT xywh rows to [id,x1,y1,x2,y2,mark,class] per frame."""
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8-sig") as stream:
            rows = [line.strip().split(",") for line in stream if line.strip()]
    else:
        rows = source
    frames: dict[int, list] = {}
    seen = set()
    for row in rows:
        if len(row) < 6:
            raise ValueError("MOT rows require at least frame,id,x,y,w,h")
        values = np.asarray(row[:8], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("MOT rows must be finite")
        frame, identity, x, y, width, height = values[:6]
        if (frame < 1 or frame != int(frame) or identity < 0 or identity != int(identity) or
                width <= 0 or height <= 0):
            raise ValueError("MOT rows require integer frame/ID and positive width/height")
        key = int(frame), int(identity)
        if key in seen:
            raise ValueError(f"Duplicate MOT ID {key[1]} in frame {key[0]}")
        seen.add(key)
        mark = values[6] if len(values) > 6 else 1
        cls = values[7] if ground_truth and len(values) > 7 else 1
        # A -1 placeholder denotes classless MOT rows (e.g. formatter output).
        if cls == -1:
            cls = 1
        if ground_truth and (cls != int(cls) or not 1 <= cls <= 13):
            raise ValueError(f"Invalid MOT17 ground-truth class: {cls}")
        frames.setdefault(int(frame), []).append([identity, x, y, x + width, y + height, mark, cls])
    return {frame: np.asarray(rows, dtype=float) for frame, rows in frames.items()}


def _prepare_frames(gt: dict, predictions: dict) -> tuple[list, np.ndarray, np.ndarray]:
    empty = np.empty((0, 7))
    frames = []
    # Compact ID maps avoid allocations proportional to arbitrary input IDs.
    gt_map, pred_map = {}, {}
    for frame in sorted(gt.keys() | predictions.keys()):
        truth, pred = gt.get(frame, empty), predictions.get(frame, empty)
        similarity = _ious(truth[:, 1:5], pred[:, 1:5])
        scores = np.where(similarity >= 0.5 - _EPS, similarity, 0)
        rows, cols = linear_sum_assignment(-scores)
        ignored = cols[(scores[rows, cols] > _EPS) &
                       np.isin(truth[rows, 6], (2, 7, 8, 12))]
        keep_pred = np.ones(len(pred), dtype=bool)
        keep_pred[ignored] = False
        keep_gt = (truth[:, 5] != 0) & (truth[:, 6] == 1)
        truth, pred = truth[keep_gt], pred[keep_pred]
        gt_ids = np.array([gt_map.setdefault(int(row[0]), len(gt_map)) for row in truth], dtype=int)
        pred_ids = np.array([pred_map.setdefault(int(row[0]), len(pred_map)) for row in pred], dtype=int)
        frames.append((gt_ids, pred_ids, similarity[keep_gt][:, keep_pred]))
    gt_counts, pred_counts = np.zeros(len(gt_map)), np.zeros(len(pred_map))
    for gt_ids, pred_ids, _ in frames:
        gt_counts[gt_ids] += 1
        pred_counts[pred_ids] += 1
    return frames, gt_counts, pred_counts


def compute_mot_metrics(pred_tracks: Path | list[tuple], gt_path: Path | None) -> dict:
    """Evaluate one complete sequence, returning HOTA/MOTA/IDF1/DetA/AssA in %.

    Predictions are a MOT-format file or rows (frame,id,x,y,w,h[,score,...]);
    these are xywh, unlike tracker inputs/outputs. GT normally has the MOT17
    mark, class and visibility columns; six/seven-column classless rows also
    work. Predictions matched to distractor classes 2/7/8/12 are removed at
    IoU >= .5; GT keeps marked pedestrians only. Visibility is not thresholded.

    CLEAR uses IoU >= .5 and previous association priority; identity switches
    survive gaps. IDF1 uses optimal global identity assignment. HOTA, DetA and
    AssA average thresholds .05:.05:.95, with global alignment before matching,
    following https://github.com/JonathonLuiten/TrackEval. HOTA is averaged after
    the square root at each threshold, not computed from averaged DetA/AssA.

    ``gt_path=None`` returns None scores with evaluated=False, never fabricated
    scores. A missing supplied file raises. Empty evaluated inputs score zero.
    Counts refer to pedestrian detections after preprocessing. Frames present
    only in predictions count as false positives; inputs must span the same
    intended evaluation interval. This function does not aggregate sequences.
    """
    predictions = _read_mot(pred_tracks, ground_truth=False)
    if gt_path is None:
        return {**dict.fromkeys(("HOTA", "MOTA", "IDF1", "DetA", "AssA")),
                "evaluated": False, "reason": "ground truth not supplied",
                "num_predictions": sum(map(len, predictions.values()))}
    frames, gt_counts, pred_counts = _prepare_frames(
        _read_mot(gt_path, ground_truth=True), predictions)
    num_gt, num_pred = int(gt_counts.sum()), int(pred_counts.sum())
    shape = len(gt_counts), len(pred_counts)
    potential = np.zeros(shape)
    identity_matches = np.zeros(shape)
    last_match = np.full(len(gt_counts), -1, dtype=int)
    previous = last_match.copy()
    tp = switches = 0
    for gt_ids, pred_ids, similarity in frames:
        index = np.ix_(gt_ids, pred_ids)
        denom = similarity.sum(axis=0)[None, :] + similarity.sum(axis=1)[:, None] - similarity
        potential[index] += np.divide(similarity, denom, out=np.zeros_like(similarity), where=denom > _EPS)
        identity_matches[index] += similarity >= 0.5
        if not len(gt_ids) or not len(pred_ids):
            # TrackEval retains continuity across completely empty-sided frames.
            continue
        scores = similarity + 1000 * (previous[gt_ids, None] == pred_ids[None, :])
        scores[similarity < 0.5 - _EPS] = 0
        rows, cols = linear_sum_assignment(-scores)
        valid = scores[rows, cols] > _EPS
        matched_gt, matched_pred = gt_ids[rows[valid]], pred_ids[cols[valid]]
        switches += int(np.count_nonzero((last_match[matched_gt] != -1) &
                                        (last_match[matched_gt] != matched_pred)))
        last_match[matched_gt] = matched_pred
        previous.fill(-1)
        previous[matched_gt] = matched_pred
        tp += len(matched_gt)
    fp, fn = num_pred - tp, num_gt - tp
    rows, cols = linear_sum_assignment(-identity_matches)
    idtp = int(identity_matches[rows, cols].sum())

    totals = gt_counts[:, None] + pred_counts[None, :]
    alignment = np.divide(potential, totals - potential, out=np.zeros(shape), where=totals > potential)
    thresholds = np.arange(0.05, 0.96, 0.05)
    # Store only assigned pairs per frame; one count matrix at a time keeps
    # memory independent of the number of HOTA thresholds.
    assignments = []
    for gt_ids, pred_ids, similarity in frames:
        rows, cols = linear_sum_assignment(-alignment[np.ix_(gt_ids, pred_ids)] * similarity)
        assignments.append((gt_ids[rows], pred_ids[cols], similarity[rows, cols]))
    det_a, ass_a = [], []
    for threshold in thresholds:
        matches = np.zeros(shape)
        for gt_ids, pred_ids, similarities in assignments:
            valid = similarities >= threshold - _EPS
            matches[gt_ids[valid], pred_ids[valid]] += 1
        matched = matches.sum()
        det_a.append(matched / max(1, num_gt + num_pred - matched))
        association = matches / np.maximum(1, totals - matches)
        ass_a.append(float((matches * association).sum() / max(1, matched)))
    return {
        "HOTA": float(100 * np.sqrt(np.array(det_a) * ass_a).mean()),
        "MOTA": float(100 * (1 - (fn + fp + switches) / num_gt)) if num_gt else 0.0,
        "IDF1": float(200 * idtp / max(1, num_gt + num_pred)),
        "DetA": float(100 * np.mean(det_a)), "AssA": float(100 * np.mean(ass_a)),
        "TP": tp, "FP": fp, "FN": fn, "IDSW": switches,
        "IDTP": idtp, "IDFP": num_pred - idtp, "IDFN": num_gt - idtp,
        "num_gt": num_gt, "num_predictions": num_pred, "evaluated": True,
    }
