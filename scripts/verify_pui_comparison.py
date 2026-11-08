"""Verify separate comparison views and queue truth; no GT accuracy evaluation."""
import csv
import json
from pathlib import Path

import cv2
from pui_comparison_views import VIDEO_NAMES,VIEW_SCHEMA,rows_for_view,geojson_collection

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
MODELS=('PET-DINO','OPUS','FSOD-VFM','DINOv','YOLOE')
AVAILABLE_MODELS=MODELS+('YOLOE-26-M','YOLOE-26-L')


def verify(model,run='ugv_pui_20261106'):
    directory=ROOT/'samples'/model/'original'/run
    audit=json.loads((directory/'audit.json').read_text())
    assert audit['status']=='complete' and audit['artifact_schema']==VIEW_SCHEMA,model
    assert audit['api_enabled'] is False and audit['accuracy_evaluated'] is False
    count=audit['frames']
    rows=list(csv.DictReader((directory/'timings.csv').open()))
    packets=[json.loads(line) for line in (directory/'observations.jsonl').open()]
    assert len(rows)==len(packets)==count
    assert [int(row['frame']) for row in rows]==list(range(count))
    assert [p['frame'] for p in packets]==list(range(count))
    stamps=[int(row['timestamp_ns']) for row in rows]
    assert all(a<b for a,b in zip(stamps,stamps[1:]))
    assert stamps==[p['source']['bag_timestamp_ns'] for p in packets]
    seen=set();new_seen=set()
    for row,p in zip(rows,packets):
        ready=set(p['api_ready_event_ids'])
        assert not ready&seen;seen.update(ready)
        new=set(p['new_object_ids'])
        assert not new&new_seen;new_seen.update(new)
        assert len(p['detections'])==int(row['detections']) and len(p['tracks'])==int(row['tracks'])
        assert len(ready)==int(row['api_ready_creations'])
        assert p['api_sent']==0 and p['api_enabled'] is False
        assert not p['world_coordinates_available'] and not ready  # Strict current RGB-only run.
        updated=set(p.get('updated_object_ids',[]))
        assert updated<=new_seen and not updated&new
        assert {r['object_id'] for r in rows_for_view(p,'exports')}==new|updated
        assert set(p['geojson_export_ids'])==new
        assert int(row['geojson_exports'])==len(new)
    assert len(seen)==audit['api_ready_creations']==0 and audit['api_sent']==0
    assert len(new_seen)==audit['new_objects']==audit['geojson_features']
    if 'local_update_events' in audit:
        assert audit['local_update_events']==sum(len(p.get('updated_object_ids',[])) for p in packets)
    if 'class_updates' in audit:
        assert audit['class_updates']==sum(bool(t.get('class_update') and t['class_update']['kind']=='class') for p in packets for t in p['tracks'])
    collection=json.loads((directory/'detections.geojson').read_text())
    assert collection['type']=='FeatureCollection'
    assert {f['id'] for f in collection['features']}==new_seen
    assert len(collection['features'])==len(new_seen)
    assert collection==geojson_collection(packets,model)
    assert all(f['geometry'] is None and 'bbox_px' in f['properties'] and 'bbox' not in f for f in collection['features'])
    for name in VIDEO_NAMES:
        assert len(list((directory/'overlays'/name).glob('*.jpg')))==count
        capture=cv2.VideoCapture(str(directory/f'{name}.mp4'))
        assert capture.isOpened() and abs(capture.get(cv2.CAP_PROP_FPS)-1.)<.01
        decoded=0
        while True:
            ok,image=capture.read()
            if not ok:break
            assert image.shape[:2]==(720,1280)
            decoded+=1
        capture.release()
        assert decoded==count,(model,name,decoded,count)
    expected={'audit.json','timings.csv','observations.jsonl','detections.geojson','overlays'}|{f'{n}.mp4' for n in VIDEO_NAMES}
    assert {p.name for p in directory.iterdir()}==expected
    assert {p.name for p in (directory/'overlays').iterdir()}==set(VIDEO_NAMES)
    print(f'{model}: 3 videos × {count} frames; 3 × {count} overlays; GeoJSON objects={len(new_seen)}, API-ready=0; old combined video removed',flush=True)
    return audit


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=AVAILABLE_MODELS,default=list(MODELS))
    parser.add_argument('--run',default='ugv_pui_20261106')
    args=parser.parse_args()
    audits=[verify(model,args.run) for model in args.models]
    assert len({a['bank_sha256'] for a in audits})==1
    assert len({a['dataset_index_sha256'] for a in audits})==1
    assert len({json.dumps(a['class_support_counts'],sort_keys=True) for a in audits})==1
