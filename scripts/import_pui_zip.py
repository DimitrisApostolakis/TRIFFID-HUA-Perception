"""Import portable PUI original class masks, without box rasterization.

The existing UGV manifest schema is also used as the compatible mask-bank format
for UAV inputs; source_platform explicitly distinguishes the two independent banks.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

import cv2
import numpy as np

from triffid_ugv_perception.supports import bank_template, SupportBank, EXAMPLES_PER_CLASS
from triffid_ugv_perception.ontology import CLASSES, BY_ID


def import_zip(archive, root, platform='ugv'):
    if platform not in ('ugv','uav'):raise ValueError('Expected ugv or uav platform')
    if platform=='uav' and root.resolve().name!='PUI':raise ValueError('The UAV bank must be under a PUI folder')
    if platform=='uav' and (root/'bank.json').exists():
        active=json.loads((root/'bank.json').read_text())
        if active.get('schema_version')=='triffid.uav.support-bank.v1':
            raise ValueError('Preserve the native UAV bank: import new raw inputs to a staging PUI folder, then convert with prepare_native_uav_pui_bank.py')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    revision = root/('portable_'+digest[:12])
    if revision.exists():
        raise ValueError(f'Already imported: {revision}')
    with ZipFile(archive) as bundle:
        prefix = platform.upper()+'-PUI/'
        data = json.loads(bundle.read(prefix+'annotations/semantic_segmentation.json'))
        categories = {c['id']:c['name'].lower() for c in data['categories']}
        if any(k not in BY_ID or BY_ID[k].name != v for k,v in categories.items()):
            raise ValueError('Export category IDs do not match the shared PUI ontology')
        images = {r['id']:r for r in data['images']}
        excluded = {29,31,33,34} if platform=='ugv' else set()
        if excluded.intersection(images):
            raise ValueError('Export includes previously excluded UAV tasks')
        candidates = defaultdict(list)
        # Validate the complete support inputs before changing the active bank.
        for ann in data['annotations']:
            row=images[ann['image_id']]
            image_raw=bundle.read(prefix+row['file_name'])
            mask_raw=bundle.read(prefix+ann['mask_file'])
            image=cv2.imdecode(np.frombuffer(image_raw,np.uint8),cv2.IMREAD_COLOR)
            mask=cv2.imdecode(np.frombuffer(mask_raw,np.uint8),cv2.IMREAD_UNCHANGED)
            if image is None or mask is None or mask.ndim!=2 or image.shape[:2]!=mask.shape:
                raise ValueError(f'Invalid support image/mask: {ann["id"]}')
            if not np.isin(mask,[0,255]).all() or not mask.any():
                raise ValueError(f'Invalid binary foreground: {ann["id"]}')
            if image.shape[:2]!=(row['height'],row['width']) or int((mask>0).sum())!=ann['area']:
                raise ValueError(f'Annotation dimensions/area mismatch: {ann["id"]}')
            candidates[ann['category_id']].append(dict(task_id=row['id'],image=row['file_name'],
                mask=ann['mask_file'],sha256=hashlib.sha256(image_raw).hexdigest(),pixels=ann['area']))
        revision.mkdir(parents=True)
        # Exclude visual previews and reproducibility script copies; retain original inputs/provenance.
        for name in bundle.namelist():
            relative=PurePosixPath(name).relative_to(prefix.rstrip('/'))
            if relative.parts[0] not in ('images','annotations','metadata','semantic_masks_by_class'):
                continue
            if '..' in relative.parts or name.endswith('/'):
                continue
            target=revision/Path(relative)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(bundle.read(name))
    manifest=bank_template();counts={}
    if platform=='uav':manifest.update(source_platform='uav',site='PUI')
    for item in CLASSES:
        seen=set();selected=[]
        for row in sorted(candidates[item.id],key=lambda r:r['task_id']):
            if row['sha256'] in seen:continue
            seen.add(row['sha256'])
            if len(selected)<EXAMPLES_PER_CLASS:
                row=dict(row,image=str((revision/row['image']).relative_to(root)),
                         mask=str((revision/row['mask']).relative_to(root)))
                selected.append(row)
        for index,row in enumerate(selected):
            manifest['classes'][item.name][index]=dict(id=f"task_{row['task_id']}",image=row['image'],mask=row['mask'])
        counts[item.name]=dict(available_distinct=len(seen),imported=len(selected),examples=selected)
    staged=root/('bank_'+digest[:12]+'.json')
    staged.write_text(json.dumps(manifest,indent=2)+'\n')
    ready=[item for item in CLASSES if counts[item.name]['imported']]
    selection=SupportBank(staged).select_available(ready)
    report=dict(export_sha256=digest,source_archive=str(archive),format=platform.upper()+'-PUI portable semantic class masks',
        included_task_ids=sorted(images),excluded_task_ids=sorted(excluded),classes=counts,
        one_shot_classes=[c.name for c in ready],five_shot_classes=[c.name for c in ready if counts[c.name]['imported']>=5],
        sixteen_shot_classes=[c.name for c in ready if counts[c.name]['imported']>=16],
        mask_policy='Original per-class binary PNGs; no inferred masks or filled boxes',
        ways=len(ready),supports=sum(counts[c.name]['imported'] for c in ready))
    if platform=='uav':report.update(source_platform='uav',site='PUI')
    history=root/'history'/('before_'+digest[:12]);history.mkdir(parents=True)
    for name in ('bank.json','import_report.json'):
        if (root/name).exists():shutil.copyfile(root/name,history/name)
    (root/'bank.json').write_bytes(staged.read_bytes())
    for destination in (root/'import_report.json',revision/'import_report.json'):
        destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(ways=report['ways'],supports=report['supports'],
        counts={c.name:counts[c.name]['imported'] for c in ready},bank_sha256=selection.manifest_sha256)),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--platform',choices=['ugv','uav'],default='ugv')
    args=parser.parse_args()
    import_zip(args.archive,args.bank,args.platform)
