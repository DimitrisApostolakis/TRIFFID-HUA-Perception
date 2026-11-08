"""Reversible PUI directory migration; no deletion and no model execution."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
ARCHIVE=ROOT/'archive/samples_legacy_20261106'


def main():
    from triffid_ugv_perception.supports import SupportBank
    from triffid_ugv_perception.ontology import classes_for
    original=ROOT/'ugv_supports'
    report=json.loads((original/'import_report.json').read_text())
    classes=classes_for(','.join(report['one_shot_classes']))
    before=SupportBank(original).select_available(classes)
    manifest_sha=hashlib.sha256((original/'bank.json').read_bytes()).hexdigest()
    holding=ROOT/'ugv_supports_before_PUI'
    if holding.exists() or (original/'PUI').exists() or ARCHIVE.exists():
        raise FileExistsError('Migration has already started or completed; inspect before repeating')
    original.rename(holding)
    original.mkdir()
    holding.rename(original/'PUI')
    after=SupportBank(original/'PUI').select_available(classes)
    assert before.fingerprints==after.fingerprints
    assert hashlib.sha256((original/'PUI/bank.json').read_bytes()).hexdigest()==manifest_sha
    moves=[]
    for source,destination in [
        ('samples/pui_rgb_frames','data/PUI/rgb_frames'),
        ('samples/pui_run','data/PUI/original_rosbag'),
        ('samples/pui_scenarios','data/PUI/scenarios'),
        ('samples/scenario2_pair_20261106/rgb_frames','data/PUI/scenario2_pair_rgb_frames'),
    ]:
        src,dst=ROOT/source,ROOT/destination
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True,exist_ok=True)
        src.rename(dst)
        moves.append(dict(source=source,destination=destination))
    ARCHIVE.parent.mkdir(exist_ok=True)
    (ROOT/'samples').rename(ARCHIVE)
    models=['OPUS','PET-DINO','SANSA','DEVIT','SINE','COSINE','FSOD-VFM','DINOv','YOLOE']
    for model in models:
        (ROOT/'samples'/model).mkdir(parents=True)
    record=dict(bank='ugv_supports/PUI',classes=len(classes),
        supports=sum(len(x) for _,x in after.examples),bank_sha256=manifest_sha,
        moves=moves,previous_samples=str(ARCHIVE.relative_to(ROOT)),
        deleted_files=0,model_runs=0,
        note='Renames preserve all data; archiving does not reclaim existing disk usage.')
    (ARCHIVE.parent/'PUI_MIGRATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
