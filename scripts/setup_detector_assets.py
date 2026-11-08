"""Download official detector checkpoints only; never construct/run a model."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).absolute().parents[1]
WEIGHTS = ROOT/'ugv_models/weights'
ASSETS = [
    ('FSOD-VFM/sam2.1_hiera_large.pt', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'),
    ('FSOD-VFM/upn_large.pth', 'https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth'),
    ('DINOv/model_swinT.pth', 'https://github.com/UX-Decoder/DINOv/releases/download/checkpoint/model_swinT.pth'),
    ('DINOv/model_swinL.pth', 'https://github.com/UX-Decoder/DINOv/releases/download/checkpoint/model_swinL.pth'),
    ('YOLOE/yoloe-v8s-seg.pt', 'https://huggingface.co/jameslahm/yoloe/resolve/main/yoloe-v8s-seg.pt'),
    ('YOLOE/yoloe-v8l-seg.pt', 'https://huggingface.co/jameslahm/yoloe/resolve/main/yoloe-v8l-seg.pt'),
]


def obtain(item):
    relative, url = item
    path = WEIGHTS/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(path.suffix+'.partial')
        subprocess.run(['curl','-L','--fail','--silent','--show-error','--retry','3',
                        '--continue-at','-','--output',str(temporary),url], check=True)
        if temporary.stat().st_size < 1_000_000:
            raise ValueError(f'Checkpoint is unexpectedly small: {temporary}')
        temporary.rename(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(chunk)
    print(f'Ready: {relative} ({path.stat().st_size/1024**2:.1f} MiB)',flush=True)
    return dict(path=str(path.relative_to(ROOT)),url=url,bytes=path.stat().st_size,sha256=digest.hexdigest())


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(obtain, ASSETS))
    (WEIGHTS/'assets.json').write_text(json.dumps(records,indent=2)+'\n')
