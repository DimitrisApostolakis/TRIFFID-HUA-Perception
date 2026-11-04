"""Small, testable video/image IO boundary."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import cv2


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    def to_dict(self):
        return asdict(self)


class VideoReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError(f'Video not found: {path}')
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f'Cannot open video: {path}')
        self.info = VideoInfo(str(self.path), self.cap.get(cv2.CAP_PROP_FPS),
                              int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not math.isfinite(self.info.fps) or self.info.fps <= 0 or min(
                self.info.width, self.info.height, self.info.frame_count) <= 0:
            self.cap.release()
            raise ValueError('Video has invalid timing/dimensions; cannot align SRT reliably')

    def read_rgb(self, index):
        if not 0 <= index < self.info.frame_count:
            raise ValueError(f'Frame outside video: {index}')
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f'Cannot seek to frame {index}')
        ok, bgr = self.cap.read()
        if not ok or abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - index - 1) > 0.5:
            raise ValueError(f'Failed to decode requested frame {index}')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cap.release()


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read RGB image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Cannot save frame: {path}')
