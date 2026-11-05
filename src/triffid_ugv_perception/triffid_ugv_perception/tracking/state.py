import numpy as np
from .kalman import _KalmanBBox

class _Track:
    """Single tracked object with state machine."""

    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2

    def __init__(self, track_id, detection, n_init):
        self.id = track_id
        self.kf = _KalmanBBox(detection['bbox'])
        self.bbox = detection['bbox']
        self.position = detection.get('position')
        self.extent = detection.get('extent', (0.0, 0.0, 0.0))
        self.n_depth_pts = detection.get('n_depth_pts', 0)
        self.class_id = detection.get('class_id', -1)
        self.class_name = detection['class_name']
        self.confidence = detection['confidence']
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.n_init = n_init
        self.state = self.TENTATIVE if n_init > 1 else self.CONFIRMED
        # Class vote histogram: {class_name: count}
        self._class_votes = {self.class_name: 1}

    @property
    def is_confirmed(self):
        return self.state == self.CONFIRMED

    @property
    def is_lost(self):
        return self.state == self.LOST

    def predict(self):
        """Kalman predict; returns predicted bbox."""
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.kf.predict()
        return self.bbox

    def update(self, detection):
        """Kalman update with matched detection."""
        self.kf.update(detection['bbox'])
        self.bbox = detection['bbox']
        self.position = detection.get('position')
        self.extent = detection.get('extent', (0.0, 0.0, 0.0))
        self.n_depth_pts = detection.get('n_depth_pts', 0)
        self.confidence = detection['confidence']
        self.hits += 1
        self.time_since_update = 0
        # Majority-vote class: accumulate votes, pick the winner
        det_cls = detection['class_name']
        self._class_votes[det_cls] = self._class_votes.get(det_cls, 0) + 1
        best_cls = max(self._class_votes, key=self._class_votes.get)
        self.class_name = best_cls
        det_cid = detection.get('class_id', -1)
        if best_cls == det_cls:
            self.class_id = det_cid

        if self.state == self.TENTATIVE and self.hits >= self.n_init:
            self.state = self.CONFIRMED
        elif self.state == self.LOST:
            self.state = self.CONFIRMED

    def mark_lost(self):
        self.state = self.LOST
