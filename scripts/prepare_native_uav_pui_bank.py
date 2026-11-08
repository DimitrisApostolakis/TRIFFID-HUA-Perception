"""Convert imported PUI masks to the current UAV ontology and native five-slot bank.

Run with the latest UAV checkout's src/triffid_uav_perception on PYTHONPATH.
Original portable inputs and the ZIP remain unchanged.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def main():
    from triffid_uav_perception.ontology import CLASSES
    from triffid_uav_perception.supports import SupportBank, bank_template, EXAMPLES_PER_CLASS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--exclude-task-ids', type=int, nargs='*', default=[],
                        help='Hold out these images and identical image-content copies from support selection')
    args=parser.parse_args();root=args.bank.resolve()
    previous=json.loads((root/'import_report.json').read_text())
    revision=root/('portable_'+previous['export_sha256'][:12])
    data=json.loads((revision/'annotations/semantic_segmentation.json').read_text())
    images={r['id']:r for r in data['images']}
    requested_exclusions = set(args.exclude_task_ids)
    if requested_exclusions - images.keys():
        raise ValueError('Held-out task IDs are absent from the portable inputs')
    image_hashes = {tid: hashlib.sha256((revision/row['file_name']).read_bytes()).hexdigest()
                    for tid,row in images.items()} if requested_exclusions else {}
    excluded_hashes = {image_hashes[tid] for tid in requested_exclusions}
    excluded_images = requested_exclusions | {tid for tid,digest in image_hashes.items()
                                              if digest in excluded_hashes}
    mapping={source:c for c in CLASSES for source in c.source_ids}
    masks={};source_classes=defaultdict(set);excluded=set()
    names={c['id']:c['name'] for c in data['categories']}
    for ann in data['annotations']:
        if ann['image_id'] in excluded_images:continue
        item=mapping.get(ann['category_id'])
        if item is None:
            excluded.add(names[ann['category_id']]);continue
        key=(ann['image_id'],item.id)
        mask=cv2.imread(str(revision/ann['mask_file']),cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim!=2 or not np.isin(mask,[0,255]).all():
            raise ValueError('Invalid original binary mask')
        masks[key]=(masks.get(key,np.zeros_like(mask))|mask)
        source_classes[key].add(names[ann['category_id']])
    derived=root/'native_masks';derived.mkdir(exist_ok=True)
    candidates=defaultdict(list)
    for (tid,cid),mask in sorted(masks.items()):
        image=revision/images[tid]['file_name']
        with image.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(handle.read()).hexdigest()
        destination=derived/f'task_{tid:06d}_class_{cid:02d}.png'
        if not cv2.imwrite(str(destination),mask):raise OSError('Cannot write native mask')
        candidates[cid].append(dict(task_id=tid,image=str(image.relative_to(root)),
            mask=str(destination.relative_to(root)),sha256=sha,pixels=int((mask>0).sum()),
            source_classes=sorted(source_classes[(tid,cid)])))
    manifest=bank_template();manifest.update(source_platform='uav',site='PUI')
    counts={};ready=[];five=[]
    for item in CLASSES:
        distinct={}
        for row in candidates[item.id]:distinct.setdefault(row['sha256'],row)
        available=list(distinct.values());selected=available[:EXAMPLES_PER_CLASS]
        for index,row in enumerate(selected):
            manifest['classes'][item.name][index]=dict(id=f"task_{row['task_id']}",image=row['image'],mask=row['mask'])
        counts[item.name]=dict(available_distinct=len(available),imported=len(selected),examples=selected)
        if selected:ready.append(item)
        if len(selected)==EXAMPLES_PER_CLASS:five.append(item)
    staged=root/'bank.native.staged.json'
    staged.write_text(json.dumps(manifest,indent=2)+'\n')
    bank=SupportBank(staged)
    one=bank.select(ready,shots=1)
    if five:bank.select(five,shots=5)
    history=root/'history'/('before_native_uav_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    history.mkdir(parents=True,exist_ok=False)
    for name in ('bank.json','import_report.json'):shutil.copyfile(root/name,history/name)
    report=dict(previous,classes=counts,one_shot_classes=[c.name for c in ready],
        five_shot_classes=[c.name for c in five],sixteen_shot_classes=[],
        ways=len(ready),supports=sum(r['imported'] for r in counts.values()),
        native_schema=manifest['schema_version'],examples_per_class=EXAMPLES_PER_CLASS,
        human_policy={str(source):mapping[source].name for source in (6,23,43)},
        excluded_from_native_ontology=sorted(excluded),
        unavailable_slots='Unfilled slots reference absent files; no examples duplicated or invented',
        native_uav_module=str(Path(__import__('triffid_uav_perception.supports',fromlist=['x']).__file__)),
        mask_policy='Original annotated masks; per-image union only for native class mapping')
    if requested_exclusions:
        report.update(included_task_ids=sorted(set(images)-excluded_images),
                      excluded_task_ids=sorted(excluded_images),
                      held_out_task_ids=sorted(requested_exclusions),
                      held_out_image_sha256=sorted(excluded_hashes),
                      holdout_policy='Exclude source task IDs and all identical image-content copies')
    staged.replace(root/'bank.json')
    (root/'import_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(ways=report['ways'],supports=report['supports'],
        counts={c.name:counts[c.name]['imported'] for c in ready},five_shot_classes=report['five_shot_classes'],
        excluded=sorted(excluded),native_schema=manifest['schema_version']),indent=2),flush=True)


if __name__=='__main__':main()
