"""Explicit semantic-mask to component adapter; replaceable with an instance backend."""
import cv2
import numpy as np

class SemanticDetector:
    def __init__(self, backend, classes, supports, confidence_threshold=0.35, min_area_px=1):
        if not 0 <= confidence_threshold <= 1 or min_area_px < 1:
            raise ValueError('Invalid confidence threshold or component area')
        self.backend, self.classes = backend, tuple(classes)
        self.confidence_threshold, self.min_area_px = confidence_threshold, min_area_px
        self.support_metadata = supports.metadata()
        self.backend.prepare(self.classes, supports)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        by_id = {item.id: item for item in self.classes}
        detections, seen = [], set()
        for prediction in self.backend.predict(rgb, self.classes):
            if prediction.class_id not in by_id or prediction.class_id in seen:
                raise ValueError('Backend returned an unexpected or repeated class')
            seen.add(prediction.class_id)
            mask = np.asarray(prediction.mask)
            if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
                raise ValueError('Backend mask must be boolean and match the original image')
            scores = prediction.probabilities
            if scores is None or scores.shape != mask.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError('Semantic backend must provide finite pixel probabilities in [0, 1]')
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            for index in range(1, count):
                x, y, width, height, area = stats[index]
                if area < self.min_area_px:
                    continue
                # Reduce scores in the crop, avoiding a full-image scan per component.
                component_crop = labels[y:y+height, x:x+width] == index
                score = float(scores[y:y+height, x:x+width][component_crop].mean())
                if score < self.confidence_threshold:
                    continue
                component = np.zeros(mask.shape, dtype=bool)
                component[y:y+height, x:x+width] = component_crop
                item = by_id[prediction.class_id]
                detections.append(dict(bbox=(float(x), float(y), float(x+width), float(y+height)),
                                       class_id=item.id, class_name=item.name,
                                       confidence=score, mask=component))
        if seen != set(by_id):
            raise ValueError('Backend omitted requested classes')
        return detections

    def metadata(self):
        return dict(self.backend.metadata(), support_bank=self.support_metadata,
                    instance_method='8-connected semantic components; touching objects may join',
                    score_type='mean_foreground_probability; not calibrated detection confidence')
