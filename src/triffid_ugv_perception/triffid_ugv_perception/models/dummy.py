"""Explicit offline/debug backend; never selected as a failed-model fallback."""
from ..ontology import TARGET_CLASSES

class DummyDetector:
    def detect(self, bgr):
        height, width = bgr.shape[:2]
        return [dict(bbox=(0., 0., float(width), float(height)), class_id=0,
                     class_name=TARGET_CLASSES[0], confidence=0.99, mask=None)]

    def metadata(self):
        return {'name': 'dummy', 'inference': 'disabled'}
