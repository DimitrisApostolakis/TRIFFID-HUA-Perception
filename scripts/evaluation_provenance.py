"""Small shared provenance contract; no model imports or GPU work."""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def capture(split,backend,script):
    inputs={}
    def add(path):
        path=Path(path)
        inputs[str(path)]=sha256(path)
    add(split['bank'])
    for q in split['queries']:
        add(q['image'])
        for path in q['masks'].values():add(path)
    sources={str(Path(script).resolve()):sha256(script)}
    for folder in ('src/triffid_uav_perception/triffid_uav_perception/models',
                   'src/triffid_ugv_perception/triffid_ugv_perception/models'):
        for path in (ROOT/folder).glob('*.py'):sources[str(path.relative_to(ROOT))]=sha256(path)
    weights={}
    checkpoint=backend.get('checkpoint')
    if checkpoint and Path(checkpoint).is_file():weights[checkpoint]=sha256(checkpoint)
    return dict(python=sys.version,executable=sys.executable,inputs_sha256=inputs,
        source_files_sha256=sources,weights_sha256=weights,
        repository_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        repository_dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()))
