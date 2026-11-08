"""Audit exported masks, stable IDs and sampling equivalence after a replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from triffid_ugv_perception.io import write_json


def audit(directory,reference=None):
    root=Path(directory)
    records=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
    summary=json.loads((root/'summary.json').read_text())
    events=[];masks_checked=0;classes=Counter()
    for record in records:
        objects=[t['object_id'] for t in record['tracks'] if t.get('object_id')]
        assert len(objects)==len(set(objects)),f'Duplicate stable UUID in frame {record["frame"]}'
        for detection in record['detections']:
            classes[detection['class_name']]+=1
            if 'mask_file' not in detection:continue
            with np.load(root/detection['mask_file']) as data:
                shape=tuple(data['shape'])
                masks=np.unpackbits(data['packed_masks'],axis=-1,count=shape[-1]).astype(bool)
            mask=masks[detection['mask_index']]
            assert mask.shape==(record['source']['height'],record['source']['width'])
            ys,xs=np.where(mask)
            assert len(xs)>0
            assert detection['bbox']==[float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]
            masks_checked+=1
        events.extend(record['new_objects'])
    ids=[event['object_id'] for event in events]
    assert len(ids)==len(set(ids)),'Repeated creation UUID'
    assert len(records)==summary['frames']==summary['tracker_updates']
    assert summary['detector']['query_encodings']==len(records)
    if summary['backend']=='cosine':assert summary['detector']['clip_query_encodings']==len(records)
    if reference:
        other=[json.loads(line)['source'] for line in (Path(reference)/'frames.jsonl').read_text().splitlines()]
        assert [record['source'] for record in records]==other,'Different source frames'
    registry=json.loads(Path(summary['registry_path']).read_text())
    for event in events:
        stored=json.loads((Path(summary['outbox_path'])/(event['object_id']+'.json')).read_text())
        assert stored==event==registry['objects'][event['object_id']]['created_event']
    warm=[record['total_s'] for record in records[1:]]
    result=dict(passed=True,frames=len(records),masks_checked=masks_checked,creation_events=len(ids),
        class_detections=dict(classes),same_source_frames_as=str(reference) if reference else None,
        warm_mean_s=float(np.mean(warm)),warm_p95_s=float(np.percentile(warm,95)),
        limits='Checks export consistency and UUID uniqueness; does not verify accuracy or physical deduplication')
    write_json(root/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('--reference')
    args=parser.parse_args();print(json.dumps(audit(args.directory,args.reference),indent=2))
