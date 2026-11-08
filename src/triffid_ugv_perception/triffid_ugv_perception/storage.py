"""Bounded optional snapshots; durable delivery state belongs in SQLite."""
import json
import os
from pathlib import Path


def _atomic_write_geojson(path, collection: dict, max_bytes=None) -> None:
    payload = json.dumps(collection, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if max_bytes is not None and len(payload) > max_bytes:
        raise ValueError('Snapshot exceeds its configured byte budget')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp, path)


class BoundedImageWriter:
    """Rotate only this writer's PNG files, before writing another bounded image."""
    def __init__(self, directory, prefix, max_files=100, max_bytes=64*1024*1024):
        if max_files < 1 or max_bytes < 1 or not prefix.isidentifier():
            raise ValueError('Invalid debug storage limits/prefix')
        self.directory, self.prefix = Path(directory), prefix
        self.max_files, self.max_bytes = max_files, max_bytes

    def write(self, filename, image):
        import cv2
        if Path(filename).name != filename or not filename.startswith(self.prefix+'_') or not filename.endswith('.png'):
            raise ValueError('Invalid managed debug filename')
        success, encoded = cv2.imencode('.png', image)
        if not success or encoded.nbytes > self.max_bytes:
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        files = sorted((p for p in self.directory.glob(self.prefix+'_*.png') if p.is_file()),
                       key=lambda p:p.stat().st_mtime_ns)
        total = sum(p.stat().st_size for p in files)
        # A same-stamp overwrite is also safe: remove the old managed copy.
        target = self.directory/filename
        for path in list(files):
            if path == target:
                total -= path.stat().st_size
                path.unlink()
                files.remove(path)
        while files and (len(files) >= self.max_files or total+encoded.nbytes > self.max_bytes):
            path = files.pop(0)
            total -= path.stat().st_size
            path.unlink()
        try:
            target.write_bytes(encoded.tobytes())
        except OSError:
            return False
        return True
