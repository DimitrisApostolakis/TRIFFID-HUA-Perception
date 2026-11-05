"""Lazy detector factories. Register another adapter without touching RGB-D/ROS stages."""
from .base import DetectionBackend

_FACTORIES = {}

def register_detector(name, factory):
    if not name or name in _FACTORIES:
        raise ValueError(f'Detector already registered or unnamed: {name!r}')
    _FACTORIES[name] = factory

def _sansa(config):
    from ..ontology import classes_for
    from ..supports import SupportBank
    from .sansa import SansaBackend
    from .semantic import SemanticDetector
    classes = classes_for(config['classes'])
    supports = SupportBank(config['support_bank']).select(classes, shots=config['shots'])
    backend = SansaBackend(config['sansa_repo'], config['checkpoint'], config['sam2_checkpoint'],
                           config['device'], config['image_size'], config['mask_threshold'])
    return SemanticDetector(backend, classes, supports, config['confidence_threshold'], config['min_area_px'])

def _yolo(config):
    from .yolo import YoloDetector
    return YoloDetector(config)

def create_detector(config) -> DetectionBackend:
    if config['use_dummy_detections']:
        from .dummy import DummyDetector
        return DummyDetector()
    name = config['backend']
    if name not in _FACTORIES:
        raise ValueError(f'Unknown detector {name!r}; registered: {sorted(_FACTORIES)}')
    return _FACTORIES[name](config)

register_detector('sansa', _sansa)
register_detector('yolo', _yolo)
