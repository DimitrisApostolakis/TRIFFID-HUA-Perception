from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

from ..ontology import UavClass
from ..supports import SupportSelection


@dataclass
class ClassPrediction:
    class_id: int
    mask: np.ndarray  # boolean semantic foreground, original query H x W
    probabilities: np.ndarray | None = None  # not calibrated detection confidence


class SegmentationBackend(Protocol):
    def prepare(self, classes: Sequence[UavClass], supports: SupportSelection) -> None: ...
    def predict(self, rgb: np.ndarray, classes: Sequence[UavClass]) -> Iterable[ClassPrediction]: ...
    def metadata(self) -> dict: ...
