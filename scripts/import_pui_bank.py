"""Import original-ID semantic annotations into a portable UGV support bank."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from triffid_ugv_perception.ontology import CLASSES
from triffid_ugv_perception.supports import bank_template, SupportBank, EXAMPLES_PER_CLASS

root = Path('ugv_supports/PUI')
source = root / 'source_annotations'
manifest = root / 'bank.json'
if manifest.exists():
    raise SystemExit('Bank already exists; refusing to overwrite')
categories = json.loads((source / 'categories.json').read_text())
assert categories == {str(c.id): c.name for c in CLASSES}
data = bank_template()
candidates = {c.id: [] for c in CLASSES}
for item in json.loads((source / 'manifest.json').read_text())['items']:
    image = source / item['image']
    mask = source / item['semantic_mask']
    rgb = cv2.imread(str(image))
    labels = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
    assert labels.ndim == 2 and labels.shape == rgb.shape[:2]
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    for class_id in np.unique(labels):
        if class_id != 255:
            assert int(class_id) in candidates
            candidates[int(class_id)].append((item, digest, int((labels == class_id).sum())))
report = {}
for c in CLASSES:
    seen = set()
    rows = []
    # Prefer ground photographs over drone views, then stable source filename order.
    for item, digest, pixels in sorted(candidates[c.id], key=lambda r: ('DJI_' in r[0]['image'] or 'view_' in r[0]['image'], r[0]['image'])):
        if digest in seen:
            continue
        seen.add(digest)
        if len(rows) < EXAMPLES_PER_CLASS:
            slot = len(rows)
            data['classes'][c.name][slot] = dict(id=Path(item['image']).stem,
                image='source_annotations/' + item['image'],
                mask='source_annotations/' + item['semantic_mask'], foreground_value=c.id)
            rows.append(dict(source=item['image'], foreground_pixels=pixels, sha256=digest))
    report[c.name] = dict(available_distinct=len(seen), imported=len(rows), examples=rows)
manifest.write_text(json.dumps(data, indent=2) + '\n')
bank = SupportBank(root)
ready = [c for c in CLASSES if report[c.name]['imported']]
bank.select(ready, shots=1)
complete = [c for c in ready if report[c.name]['imported'] >= 5]
assert bank.validate(complete,shots=5)['valid']
output = dict(classes=report, one_shot_classes=[c.name for c in ready], five_shot_classes=[c.name for c in complete])
(root / 'import_report.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({k: v for k, v in output.items() if k != 'classes'}, indent=2))
