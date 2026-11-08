"""Rebuild the support bank from Label Studio brush masks, preserving originals.

Explicit exclusions override the export: deleted tasks can remain in stale JSON.
Boxes are never silently converted into segmentation masks.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from triffid_ugv_perception.io import write_json
from triffid_ugv_perception.ontology import CLASSES, BY_NAME
from triffid_ugv_perception.supports import bank_template, SupportBank, EXAMPLES_PER_CLASS


def decode_brush(rle, height, width):
    """Decode Label Studio's rle-pack RGBA stream; return its alpha mask.

    Format: HumanSignal/label-studio-converter, brush.py. Length and bounds
    checks prevent malformed exports from allocating arbitrary-sized arrays.
    """
    bits=''.join(f'{int(value):08b}' for value in rle)
    position=0
    def read(count):
        nonlocal position
        end=position+count
        if end>len(bits):raise ValueError('Truncated brush RLE')
        value=int(bits[position:end],2);position=end
        return value
    count=read(32);word=read(5)+1;sizes=[read(4)+1 for _ in range(4)]
    if count!=height*width*4 or word!=8:
        raise ValueError('Brush RLE dimensions or word size mismatch')
    values=np.empty(count,dtype=np.uint8)
    start=0
    while start<count:
        repeat=read(1);end=start+1+read(sizes[read(2)])
        if end>count:raise ValueError('Brush RLE exceeds image dimensions')
        if repeat:values[start:end]=read(word)
        else:
            for index in range(start,end):values[index]=read(word)
        start=end
    return values.reshape(height,width,4)[:,:,3]>0


def import_export(export, bank_root, excluded, image_root=None, validate_box_tasks=()):
    export,root=Path(export),Path(bank_root)
    raw=export.read_bytes();tasks=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    revision=root/('labelstudio_'+digest[:12])
    if revision.exists():raise ValueError(f'Import already exists: {revision}')
    excluded={int(value) for value in excluded}
    validate_box_tasks=set(map(int,validate_box_tasks))
    source=root/'source_annotations'
    source_manifest=json.loads((source/'manifest.json').read_text())
    source_images={Path(x['image']).name:source/x['image'] for x in source_manifest['items']}
    previous_images={}
    if (root/'import_report.json').is_file():
        previous=json.loads((root/'import_report.json').read_text())
        manifest=root/('labelstudio_'+previous['export_sha256'][:12])/'manifest.json'
        if manifest.is_file():
            previous_images={int(x['task_id']):root/x['image'] for x in json.loads(manifest.read_text())['items']}
    external={}
    if image_root:
        for path in Path(image_root).rglob('*'):
            if path.is_file():external.setdefault(path.name.lower(),[]).append(path)
    prepared=[];counts=Counter();ids=set();skipped=[]
    for task in tasks:
        task_id=int(task['id'])
        if task_id in ids:raise ValueError(f'Repeated task ID: {task_id}')
        ids.add(task_id)
        if task_id in excluded:continue
        filename=Path(unquote(urlparse(task['data']['image']).path)).name
        image_path=source_images.get(filename)
        if image_path is None and task_id in previous_images:
            candidate=previous_images[task_id]
            if candidate.name.endswith(filename) or candidate.name.endswith(filename.split('-',1)[-1]):image_path=candidate
        if image_path is None:
            original=filename.split('-',1)[-1]
            matches=external.get(filename.lower(),external.get(original.lower(),[]))
            if len(matches)!=1:raise ValueError(f'Missing or ambiguous original image: {filename}')
            image_path=matches[0]
        image=cv2.imread(str(image_path))
        if image is None:raise ValueError(f'Unreadable image: {image_path}')
        height,width=image.shape[:2]
        annotations=[a for a in task['annotations'] if not a.get('was_cancelled')]
        if len(annotations)!=1:raise ValueError(f'Task {task_id}: expected one completed annotation')
        semantic=np.full((height,width),255,dtype=np.uint8)
        masks={}
        box_masks={}
        if task_id in validate_box_tasks:
            for result in annotations[0]['result']:
                value=result['value']
                if result['type']!='rectanglelabels' or 'x' not in value:continue
                name=value['rectanglelabels'][0].lower()
                a=max(0,int(np.floor(value['x']*width/100))-2)
                b=max(0,int(np.floor(value['y']*height/100))-2)
                c=min(width,int(np.ceil((value['x']+value['width'])*width/100))+2)
                d=min(height,int(np.ceil((value['y']+value['height'])*height/100))+2)
                box_masks.setdefault(name,np.zeros((height,width),bool))[b:d,a:c]=True
        for result in annotations[0]['result']:
            if result['type'] not in ('brushlabels','rectanglelabels'):continue
            if result['type']=='rectanglelabels' and 'rle' not in result['value']:continue
            if result.get('image_rotation',0)!=0:raise ValueError('Rotated masks require explicit conversion')
            if (result['original_height'],result['original_width'])!=(height,width):
                raise ValueError(f'Task {task_id}: annotation/image dimensions differ')
            labels=result['value'].get('brushlabels',result['value'].get('rectanglelabels'))
            if len(labels)!=1 or labels[0].lower() not in BY_NAME:
                raise ValueError(f'Unknown or ambiguous label: {labels}')
            item=BY_NAME[labels[0].lower()]
            if 'rle' not in result['value']:
                skipped.append(dict(task_id=task_id,region_id=result['id'],class_name=item.name,
                    reason='brushlabels region contains rectangle coordinates but no pixel RLE'))
                continue
            binary=decode_brush(result['value']['rle'],height,width)
            if item.name in box_masks and binary.any():
                agreement=float((binary&box_masks[item.name]).sum()/binary.sum())
                if agreement<.5:
                    skipped.append(dict(task_id=task_id,region_id=result['id'],class_name=item.name,
                        reason='RLE mask mostly outside same-class rectangles',foreground_inside_boxes=agreement))
                    continue
            masks[item.id]=masks.get(item.id,np.zeros_like(binary))|binary
            counts[item.name]+=1
            # Preserve export ordering for the preview, but bank references
            # use independent binary masks so overlapping classes lose no pixels.
            semantic[binary]=item.id
        if not masks:raise ValueError(f'Task {task_id}: no brush masks')
        prepared.append((task_id,image_path,semantic,masks))
    if not prepared:raise ValueError('No included tasks')
    revision.mkdir(parents=True)
    (revision/'masks').mkdir()
    (revision/'images').mkdir()
    shutil.copyfile(export,revision/'export.json')
    data=bank_template();candidates={item.id:[] for item in CLASSES};items=[]
    for task_id,image_path,semantic,masks in prepared:
        try:image_path.relative_to(root)
        except ValueError:
            destination=revision/'images'/f'task_{task_id}_{image_path.name}'
            shutil.copyfile(image_path,destination)
            image_path=destination
        image_hash=hashlib.sha256(image_path.read_bytes()).hexdigest()
        preview=revision/'masks'/f'task_{task_id}_semantic.png'
        if not cv2.imwrite(str(preview),semantic):raise OSError(preview)
        items.append(dict(task_id=task_id,image=str(image_path.relative_to(root)),
            semantic_mask=str(preview.relative_to(root)),class_ids=sorted(masks)))
        for class_id,binary in masks.items():
            if not binary.any():continue
            path=revision/'masks'/f'task_{task_id}_class_{class_id}.png'
            if not cv2.imwrite(str(path),binary.astype(np.uint8)*255):raise OSError(path)
            candidates[class_id].append(dict(task_id=task_id,image=str(image_path.relative_to(root)),
                mask=str(path.relative_to(root)),sha256=image_hash,pixels=int(binary.sum())))
    report={}
    for item in CLASSES:
        seen=set();selected=[]
        for row in sorted(candidates[item.id],key=lambda r:('view_' in r['image'],r['image'])):
            if row['sha256'] in seen:continue
            seen.add(row['sha256'])
            if len(selected)<EXAMPLES_PER_CLASS:selected.append(row)
        for index,row in enumerate(selected):
            data['classes'][item.name][index]=dict(id=f"task_{row['task_id']}",image=row['image'],mask=row['mask'])
        report[item.name]=dict(available_distinct=len(seen),imported=len(selected),examples=selected)
    provenance=dict(export_sha256=digest,excluded_task_ids=sorted(excluded),
        excluded_ids_present_in_export=sorted(ids&excluded),included_task_ids=[x[0] for x in prepared],
        classes=report,brush_regions=dict(counts),skipped_regions=skipped,
        overlap_policy='independent binary mask per class',box_consistency_task_ids=sorted(validate_box_tasks),
        one_shot_classes=[c.name for c in CLASSES if report[c.name]['imported']],
        five_shot_classes=[c.name for c in CLASSES if report[c.name]['imported']>=5],
        sixteen_shot_classes=[c.name for c in CLASSES if report[c.name]['imported']>=16])
    # Validate staged manifest with paths relative to the actual bank root.
    staged=root/('bank_'+digest[:12]+'.json')
    write_json(staged,data)
    bank=SupportBank(staged)
    bank.select([BY_NAME[name] for name in provenance['one_shot_classes']],shots=1)
    if provenance['five_shot_classes']:
        bank.select([BY_NAME[name] for name in provenance['five_shot_classes']],shots=5)
    history=root/'history'/digest[:12];history.mkdir(parents=True)
    for name in ('bank.json','import_report.json'):
        if (root/name).exists():shutil.copyfile(root/name,history/name)
    write_json(revision/'manifest.json',dict(items=items))
    write_json(revision/'import_report.json',provenance)
    write_json(root/'import_report.json',provenance)
    write_json(root/'bank.json',data)
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export')
    parser.add_argument('--bank',default='ugv_supports/PUI')
    parser.add_argument('--exclude-task-ids',type=int,nargs='*',default=[])
    parser.add_argument('--image-root',help='Read-only directory containing newly annotated original images')
    parser.add_argument('--validate-box-tasks',type=int,nargs='*',default=[])
    args=parser.parse_args()
    report=import_export(args.export,args.bank,args.exclude_task_ids,args.image_root,args.validate_box_tasks)
    print(json.dumps({k:v for k,v in report.items() if k not in ('classes','brush_regions')},indent=2))
