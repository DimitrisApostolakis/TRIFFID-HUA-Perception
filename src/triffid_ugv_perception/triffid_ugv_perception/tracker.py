"""
ByteTrack-style Tracker for TRIFFID Perception

Kalman-predicted bboxes, two-pass Hungarian association (high-confidence
first), tentative->confirmed gate, 3D position as a fallback cost. IDs
are persistent and never reused/reset.
"""

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


from .tracking.kalman import _KalmanBBox
from .tracking.state import _Track
from .tracking.metrics import _iou_batch, _pos_distance_batch

class ByteTracker:
    """ByteTrack-style multi-object tracker with Kalman prediction.

    iou_threshold/iou_threshold_low gate the two association passes
    (split by conf_threshold_high); pos_gate (metres) is an auxiliary
    3D-distance gate for when IoU is unreliable (small boxes); n_init
    confirmations are required before a track is published; max_age
    frames without a match before a track is dropped.
    """

    def __init__(
        self,
        iou_threshold=0.3,
        iou_threshold_low=0.15,
        conf_threshold_high=0.4,
        max_age=30,
        n_init=3,
        pos_gate=2.0,
    ):
        self.iou_threshold = iou_threshold
        self.iou_threshold_low = iou_threshold_low
        self.conf_threshold_high = conf_threshold_high
        self.max_age = max_age
        self.n_init = n_init
        self.pos_gate = pos_gate

        self.next_id = 1
        self.tracks = []   # list of _Track objects

    def update(self, detections):
        """Update with new detections (bbox/class_id/class_name/confidence/
        position dicts); returns confirmed tracks with a track_id added."""
        for track in self.tracks:
            track.predict()

        if not detections:
            # No detections → all tracks become lost
            for track in self.tracks:
                track.mark_lost()
            self._remove_dead_tracks()
            return self._output()

        # Split detections by confidence
        dets_high = []
        dets_low = []
        for det in detections:
            if det['confidence'] >= self.conf_threshold_high:
                dets_high.append(det)
            else:
                dets_low.append(det)

        # First pass: high-confidence vs confirmed + lost tracks
        confirmed_tracks = [t for t in self.tracks
                           if t.state != _Track.TENTATIVE]
        tentative_tracks = [t for t in self.tracks
                           if t.state == _Track.TENTATIVE]

        matched_t, matched_d, unmatched_tracks, unmatched_dets = \
            self._associate(confirmed_tracks, dets_high,
                            self.iou_threshold)

        for ti, di in zip(matched_t, matched_d):
            confirmed_tracks[ti].update(dets_high[di])
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_tracks]

        # Second pass: low-confidence vs remaining tracks
        if dets_low and remaining_tracks:
            matched_t2, matched_d2, unmatched_tracks2, _ = \
                self._associate(remaining_tracks, dets_low,
                                self.iou_threshold_low)
            for ti, di in zip(matched_t2, matched_d2):
                remaining_tracks[ti].update(dets_low[di])
            for i in unmatched_tracks2:
                remaining_tracks[i].mark_lost()
        else:
            for t in remaining_tracks:
                t.mark_lost()

        # Tentative tracks: match with remaining unmatched high-conf dets
        unmatched_high_dets = [dets_high[i] for i in unmatched_dets]
        if tentative_tracks and unmatched_high_dets:
            mt, md, ut, ud = self._associate(
                tentative_tracks, unmatched_high_dets,
                self.iou_threshold)
            for ti, di in zip(mt, md):
                tentative_tracks[ti].update(unmatched_high_dets[di])
            for i in ut:
                tentative_tracks[i].mark_lost()
            unmatched_high_dets = [unmatched_high_dets[i] for i in ud]
        else:
            for t in tentative_tracks:
                t.mark_lost()

        # Create new tentative tracks for remaining unmatched high-conf dets
        for det in unmatched_high_dets:
            self._create_track(det)

        self._remove_dead_tracks()
        return self._output()

    # ------------------------------------------------------------------
    #  Association
    # ------------------------------------------------------------------

    def _associate(self, tracks, detections, iou_thresh):
        """Match via IoU + 3D position gate; returns (matched_track_idx,
        matched_det_idx, unmatched_track_idx, unmatched_det_idx)."""
        if not tracks or not detections:
            return ([], [],
                    list(range(len(tracks))),
                    list(range(len(detections))))

        track_bboxes = [t.bbox for t in tracks]
        det_bboxes = [d['bbox'] for d in detections]
        iou_matrix = _iou_batch(track_bboxes, det_bboxes)

        track_pos = [t.position for t in tracks]
        det_pos = [d.get('position') for d in detections]
        dist_matrix = _pos_distance_batch(track_pos, det_pos)

        # Cost = 1 - IoU (lower is better)
        cost = 1.0 - iou_matrix

        # Class gate: penalise cross-class matches
        track_cls = [t.class_name for t in tracks]
        det_cls = [d['class_name'] for d in detections]
        for i, tc in enumerate(track_cls):
            for j, dc in enumerate(det_cls):
                if tc != dc:
                    cost[i, j] += 1e5

        # Gate: reject pairs that are both low-IoU AND far in 3D
        gate_mask = (iou_matrix < iou_thresh) & (dist_matrix > self.pos_gate)
        cost[gate_mask] = 1e5

        no_info = (iou_matrix == 0) & np.isinf(dist_matrix)
        cost[no_info] = 1e5

        if _HAS_SCIPY:
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = self._greedy_assignment(cost)

        matched_t = []
        matched_d = []
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= 1e5:
                continue
            if iou_matrix[r, c] < iou_thresh:
                # Below IoU threshold — accept only if 3D match is close
                if dist_matrix[r, c] > self.pos_gate:
                    continue
            matched_t.append(r)
            matched_d.append(c)

        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(detections)) if i not in matched_d]

        return matched_t, matched_d, unmatched_t, unmatched_d

    @staticmethod
    def _greedy_assignment(cost):
        """Fallback greedy assignment when scipy is not available."""
        used_r = set()
        used_c = set()
        pairs_r = []
        pairs_c = []

        indices = np.unravel_index(
            np.argsort(cost, axis=None), cost.shape
        )
        for r, c in zip(indices[0], indices[1]):
            if r in used_r or c in used_c:
                continue
            pairs_r.append(r)
            pairs_c.append(c)
            used_r.add(r)
            used_c.add(c)

        return np.array(pairs_r), np.array(pairs_c)

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _create_track(self, det):
        track = _Track(self.next_id, det, self.n_init)
        self.next_id += 1
        self.tracks.append(track)
        return track

    def _remove_dead_tracks(self):
        self.tracks = [
            t for t in self.tracks
            if not (t.state == _Track.LOST
                    and t.time_since_update > self.max_age)
            and not (t.state == _Track.TENTATIVE
                     and t.time_since_update > 0)
        ]

    def _output(self):
        """Return list of confirmed-track dicts (compatible with old API)."""
        results = []
        for t in self.tracks:
            if not t.is_confirmed:
                continue
            results.append({
                'bbox': t.bbox,
                'class_name': t.class_name,
                'class_id': t.class_id,
                'confidence': t.confidence,
                'position': t.position,
                'extent': t.extent,
                'n_depth_pts': t.n_depth_pts,
                'track_id': t.id,
            })
        return results
