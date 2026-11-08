"""Thin adapter to the existing pretrained MASA engine; no UGV policy changes."""


class MasaImageTracker:
    def __init__(self, memory_seconds=900):
        try:
            from triffid_ugv_perception.tracking.masa import MasaTracker
        except ImportError as error:
            raise ImportError('MASA requires the UGV package on PYTHONPATH and the existing '
                              'ugv_models/opus_env runtime. Use scripts/run_uav_tracking.sh.') from error
        self.engine = MasaTracker(memory_seconds=memory_seconds)

    def update(self, detections, bgr, timestamp_ns):
        tracks = self.engine.update(detections, bgr, timestamp_ns)
        self.engine.torch.cuda.synchronize()
        return tracks

    def metadata(self):
        return self.engine.metadata()

    def reset(self):
        """Start an independent sequence while retaining only the loaded model weights."""
        self.engine.model.tracker.reset()
        self.engine.first_timestamp = self.engine.last_timestamp = None
        self.engine.calls = self.engine.image_encodings = 0
        self.engine.embeddings = None
