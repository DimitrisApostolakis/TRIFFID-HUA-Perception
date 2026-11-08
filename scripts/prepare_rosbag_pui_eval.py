"""Freeze ROSBAG-PUI queries and a separate support bank, excluding pixel-identical queries."""
import hashlib
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pixels(path):
    a = np.asarray(Image.open(path).convert('RGB'))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

def freeze_bank(original, excluded):
    bank = json.loads(original.read_text())
    removals = []
    for name, entries in bank['classes'].items():
        kept = []
        for row in entries:
            if not row or not row.get('image') or not row.get('mask'):
                continue
            row = dict(row)
            image = (original.parent/row['image']).resolve()
            mask = (original.parent/row['mask']).resolve()
            if not image.is_file() or not mask.is_file():
                continue
            if pixels(image) in excluded:
                removals.append(dict(class_name=name, image=str(image)))
                continue
            row.update(image=str(image), mask=str(mask))
            kept.append(row)
        capacity=bank['examples_per_class']
        for i in range(len(kept),capacity):
            kept.append(dict(id=f'empty_slot_{i+1:02d}',image=f'unpopulated/{name}/{i}/image.png',mask=f'unpopulated/{name}/{i}/mask.png'))
        bank['classes'][name] = kept
    counts = {k:sum(Path(r['image']).is_file() for r in v) for k,v in bank['classes'].items()}
    return bank, removals, {k:v for k,v in counts.items() if v}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/ROSBAG-PUI/evaluation')
    args=parser.parse_args()
    source = ROOT/'data/ROSBAG-PUI/snapshot_20261107'
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    detection = json.loads((source/'annotations/object_detection.json').read_text())
    semantic = json.loads((source/'annotations/semantic_segmentation.json').read_text())
    queries = []
    for image in detection['images']:
        path = source/image['file_name']
        queries.append(dict(task_id=image['id'], annotation_id=image['label_studio_annotation_id'],
            image=str(path), image_sha256=digest(path), pixel_sha256=pixels(path),
            masks={str(a['category_id']):str(source/a['mask_file']) for a in semantic['annotations'] if a['image_id']==image['id']},
            boxes=[a for a in detection['annotations'] if a['image_id']==image['id']]))
    excluded = {q['pixel_sha256'] for q in queries}
    original = ROOT/'ugv_supports/PUI/bank.json'
    bank, removals, counts = freeze_bank(original, excluded)
    (out/'bank.json').write_text(json.dumps(bank,indent=2)+'\n')
    result = dict(schema='triffid.rosbag-pui.evaluation.v1', project='ROSBAG-PUI',
        queries=queries, bank=str(out/'bank.json'), bank_sha256=digest(out/'bank.json'),
        source_bank_sha256=digest(original), classes=list(counts), support_counts=counts,
        excluded_supports=removals,
        limitations=['Two annotated frames; no temporal identity ground truth.',
            'Pixel equality excludes re-encodings with identical pixels; other views of the same physical objects may remain.',
            'Annotations may be incomplete; semantic primary region is the union of valid saved masks.'])
    (out/'split.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(tasks=[q['task_id'] for q in queries],counts=counts,excluded=len(removals)),indent=2))

if __name__ == '__main__':
    main()
