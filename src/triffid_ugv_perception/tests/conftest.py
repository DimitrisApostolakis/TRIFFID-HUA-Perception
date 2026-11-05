import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import cv2
import numpy as np
import pytest

from triffid_ugv_perception.supports import BANK_SCHEMA, SupportBank


@pytest.fixture
def bank_selection(tmp_path):
    def select(classes, shots=5):
        classes = tuple(classes)
        root = tmp_path / 'ugv_bank'
        root.mkdir(exist_ok=True)
        entries = {}
        for item in classes:
            entries[item.name] = []
            for slot in range(1, 6):
                stem = f'{item.id}_{slot}'
                image = root / f'{stem}_image.png'
                mask = root / f'{stem}_mask.png'
                cv2.imwrite(str(image), np.full((20, 30, 3), slot * 20, np.uint8))
                foreground = np.zeros((20, 30), np.uint8)
                foreground[4:16, 5:25] = 255
                cv2.imwrite(str(mask), foreground)
                entries[item.name].append({'id': f'example_{slot:02d}',
                                           'image': image.name, 'mask': mask.name})
        manifest = root / 'bank.json'
        manifest.write_text(json.dumps({'schema_version': BANK_SCHEMA, 'classes': entries}))
        return SupportBank(root).select(classes, shots=shots)
    return select
