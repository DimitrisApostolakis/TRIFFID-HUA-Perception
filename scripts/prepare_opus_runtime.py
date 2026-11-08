"""Apply the one version-guard fix needed by OPUS's pinned dependencies.

OPUS explicitly pins MMCV 2.2.0, but pinned MMDetection 3.3.0 still rejects
that version on import. Widen that guard, preserving the actual MMCV version.
The launcher imports this source tree, rather than a second MMDetection copy.
"""
from pathlib import Path


def prepare(root):
    path = root / 'ugv_models/opus/third_party/mmdetection/mmdet/__init__.py'
    before = "mmcv_maximum_version = '2.2.0'"
    after = "mmcv_maximum_version = '2.3.0'"
    source = path.read_text()
    if before in source:
        path.write_text(source.replace(before, after, 1))
    elif after not in source:
        raise RuntimeError('Unexpected MMDetection compatibility guard; inspect before patching')


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[1])
