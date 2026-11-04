"""Explicit per-class visual supports, independent of model implementation."""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .media import read_rgb


@dataclass(frozen=True)
class Support:
    image: Path
    mask: Path
    foreground_value: int | None = None

    def load(self):
        rgb = read_rgb(self.image)
        mask = cv2.imread(str(self.mask), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.shape != rgb.shape[:2]:
            raise ValueError(f'Support mask must be single-channel and match image: {self.mask}')
        foreground = mask > 0 if self.foreground_value is None else mask == self.foreground_value
        if not foreground.any():
            raise ValueError(f'Support mask has no foreground: {self.mask}')
        return rgb, foreground.astype(np.uint8)


class SupportBank:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.data = json.loads(self.path.read_text())
        if not isinstance(self.data, dict) or self.data.get('schema_version') != 'triffid.uav.supports.v1':
            raise ValueError('Unsupported support manifest schema')
        if not isinstance(self.data.get('classes'), dict):
            raise ValueError('Support manifest must contain a classes object')

    def for_class(self, item):
        entries = self.data.get('classes', {}).get(item.name)
        if not isinstance(entries, list) or len(entries) not in (1, 5):
            raise ValueError(f'{item.name}: provide exactly 1 or 5 support image/mask pairs')
        supports = []
        for entry in entries:
            if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in ('image', 'mask')):
                raise ValueError('Each support needs image and mask paths')
            value = entry.get('foreground_value')
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError('foreground_value must be a nonnegative integer')
            supports.append(Support((self.path.parent / entry['image']).resolve(),
                                    (self.path.parent / entry['mask']).resolve(), value))
        return supports
