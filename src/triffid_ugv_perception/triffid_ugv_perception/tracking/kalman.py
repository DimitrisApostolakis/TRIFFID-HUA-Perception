import numpy as np

class _KalmanBBox:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h) — the same
    8-state model used in SORT/DeepSORT/ByteTrack."""

    _STD_WEIGHT_POS = 1.0 / 20.0
    _STD_WEIGHT_VEL = 1.0 / 160.0

    def __init__(self, bbox):
        """Initialise from (x1, y1, x2, y2)."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        a = w / h if h > 0 else 0.0

        self.x = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64)
        self.P[4:, 4:] *= 10.0  # high initial velocity uncertainty

        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4)

        self.H = np.eye(4, 8, dtype=np.float64)

    def predict(self):
        """Advance state by one time step.  Returns predicted bbox."""
        std = [
            self._STD_WEIGHT_POS * self.x[3],
            self._STD_WEIGHT_POS * self.x[3],
            1e-2,
            self._STD_WEIGHT_POS * self.x[3],
            self._STD_WEIGHT_VEL * self.x[3],
            self._STD_WEIGHT_VEL * self.x[3],
            1e-5,
            self._STD_WEIGHT_VEL * self.x[3],
        ]
        Q = np.diag(np.square(std))

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        self.x[2] = max(self.x[2], 1e-4)
        self.x[3] = max(self.x[3], 1.0)

        return self._state_to_bbox()

    def update(self, bbox):
        """Correct state with a measurement (x1, y1, x2, y2)."""
        z = self._bbox_to_measurement(bbox)

        std = [
            self._STD_WEIGHT_POS * self.x[3],
            self._STD_WEIGHT_POS * self.x[3],
            1e-1,
            self._STD_WEIGHT_POS * self.x[3],
        ]
        R = np.diag(np.square(std))

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def _state_to_bbox(self):
        """Convert internal state → (x1, y1, x2, y2)."""
        cx, cy, a, h = self.x[:4]
        w = a * h
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @staticmethod
    def _bbox_to_measurement(bbox):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        a = w / h if h > 0 else 0.0
        return np.array([cx, cy, a, h], dtype=np.float64)
