"""Backend contracts. Images and masks use original query dimensions."""
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence, TypedDict
import numpy as np
from ..ontology import UgvClass
from ..supports import SupportSelection

@dataclass
class ClassPrediction:
    class_id: int
    mask: np.ndarray
    probabilities: np.ndarray | None = None

class Detection2D(TypedDict):
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray | None

class SegmentationBackend(Protocol):
    def prepare(self, classes: Sequence[UgvClass], supports: SupportSelection) -> None: ...
    def predict(self, rgb: np.ndarray, classes: Sequence[UgvClass]) -> Iterable[ClassPrediction]: ...
    def metadata(self) -> dict: ...

class DetectionBackend(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Detection2D]: ...
    def metadata(self) -> dict: ...
