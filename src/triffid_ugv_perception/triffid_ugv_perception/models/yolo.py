"""Optional legacy YOLO adapter, retaining its original prediction/mask conversion."""
import cv2
import numpy as np
from ..ontology import TARGET_CLASSES

class YoloDetector:
    def __init__(self, config):
        from ultralytics import YOLO
        self.model = YOLO(config['model_path'])
        self.conf_thresh = config['confidence_threshold']
        self.yolo_imgsz = config['yolo_imgsz']

    def metadata(self):
        return {'name': 'yolo', 'image_size': self.yolo_imgsz}

    def detect(self, cv_image):
        """Run YOLO-seg, return dicts with bbox/class_id/class_name/confidence/mask."""
        if self.model is None:
            return []

        results = self.model(
            cv_image, conf=self.conf_thresh, imgsz=self.yolo_imgsz, verbose=False,
        )
        detections = []
        for r in results:
            masks_data = r.masks  # may be None if model produced no masks
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Extract per-instance binary mask (resized to input image)
                mask = None
                if masks_data is not None and i < len(masks_data.data):
                    mask = masks_data.data[i].cpu().numpy().astype(bool)
                    # Resize mask to match the input image dimensions
                    h, w = cv_image.shape[:2]
                    if mask.shape != (h, w):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'class_id': cls_id,
                    'class_name': TARGET_CLASSES[cls_id],
                    'confidence': float(box.conf[0]),
                    'mask': mask,
                })
        return detections
