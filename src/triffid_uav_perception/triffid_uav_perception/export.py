"""Versioned pixel-space JSON. Geographic geometry is created downstream."""
import json
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = 'triffid.uav.observations.v1'


def document(task, sources, model, classes):
    return {
        'schema_version': SCHEMA_VERSION, 'task': task,
        'coordinate_space': 'original_image_pixels',
        'pixel_origin': 'top_left', 'axes': {'x': 'right', 'y': 'down'},
        'bbox_format': 'xyxy_max_exclusive',
        'class_catalogue': [{'id': c.id, 'name': c.name, 'target_geometry': c.geometry,
                             'source_class_ids': list(c.source_ids)} for c in classes],
        'model': model, 'sources': sources,
        'deduplication': 'none; observation IDs are not persistent world-object IDs',
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return path
