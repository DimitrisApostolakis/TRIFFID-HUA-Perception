"""Atomic GeoJSON snapshot publication."""
import json
import os
from pathlib import Path

def _atomic_write_geojson(path, collection: dict) -> None:
    """Write a GeoJSON snapshot atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(collection, indent=2))
    os.replace(tmp, path)
