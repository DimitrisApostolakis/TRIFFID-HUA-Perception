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


def _devit(config):
    from .devit import DevitDetector
    return DevitDetector(config)

register_detector("devit", _devit)


def _sine(config):
    from .sine import SineDetector
    return SineDetector(config)

register_detector("sine", _sine)


def _cosine(config):
    from .cosine import CosineDetector
    return CosineDetector(config)

register_detector('cosine', _cosine)

def _pet(config):
    from .pet_dino import PetDinoDetector
    return PetDinoDetector(config)

register_detector('pet_dino', _pet)

def _opus(config):
    from .opus import OpusDetector
    return OpusDetector(config)

register_detector('opus', _opus)

def _yoloe(config):
    from .yoloe import YoloeDetector
    return YoloeDetector(config)

register_detector('yoloe', _yoloe)
