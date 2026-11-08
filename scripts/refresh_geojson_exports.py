"""Rebuild only the GeoJSON export view from saved observations. CPU only, no API."""
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import cv2
from pui_comparison_views import VIEW_SCHEMA,geojson_collection,render,rows_for_view
from pui_comparison_detectors import digest

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
MODELS=('PET-DINO','OPUS','FSOD-VFM','DINOv','YOLOE')


def refresh(model,run):
    directory=ROOT/'samples'/model/'original'/run
    audit=json.loads((directory/'audit.json').read_text())
    packets=[json.loads(line) for line in (directory/'observations.jsonl').open()]
    before={name:digest(directory/f'{name}.mp4') for name in ('detections','tracking')}
    for packet in packets:packet['geojson_export_ids']=list(packet['new_object_ids'])
    collection=geojson_collection(packets,model)
    assert len(collection['features'])==audit['new_objects']
    work=ROOT/'ugv_models/comparison_work';work.mkdir(exist_ok=True)
    started=time.perf_counter()
    with TemporaryDirectory(prefix='geojson-view-',dir=work) as temporary:
        staged=Path(temporary);(staged/'overlays').mkdir()
        writer=None;drawn=[]
        try:
            for packet in packets:
                image=cv2.imread(str(Path(audit['input_dataset'])/packet['source']['file']))
                if image is None:raise ValueError('Missing original RGB image')
                canvas=render(image,packet,'exports',model,len(packets))
                if writer is None:
                    writer=cv2.VideoWriter(str(staged/'exports.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1.,(image.shape[1],image.shape[0]))
                    if not writer.isOpened():raise OSError('Cannot create replacement export video')
                writer.write(canvas)
                if not cv2.imwrite(str(staged/'overlays'/f"{packet['frame']:04d}.jpg"),canvas):raise OSError('Cannot save export overlay')
                drawn.extend(row['object_id'] for row in rows_for_view(packet,'exports'))
        finally:
            if writer:writer.release()
        capture=cv2.VideoCapture(str(staged/'exports.mp4'));decoded=0
        while True:
            ok,_=capture.read()
            if not ok:break
            decoded+=1
        capture.release()
        assert decoded==audit['frames']==len(packets)
        assert len(drawn)==len(set(drawn))==len(collection['features'])
        assert set(drawn)=={f['id'] for f in collection['features']}
        audit.update(artifact_schema=VIEW_SCHEMA,geojson_features=len(drawn),geojson_file='detections.geojson',
            export_video_policy='Newly confirmed GeoJSON UUIDs once per discovery; geometry may be null; independent of the API queue; never sends',
            export_rerendered_from_saved_observations=True,export_cpu_render_s=time.perf_counter()-started)
        audit['timing_note']+=' Export view was subsequently regenerated on CPU; model/tracker timings are retained from the original run.'
        with (directory/'timings.csv').open() as handle:
            reader=csv.DictReader(handle);columns=list(reader.fieldnames);timings=list(reader)
        if 'geojson_exports' not in columns:columns.append('geojson_exports')
        for row,packet in zip(timings,packets):row['geojson_exports']=len(packet['new_object_ids'])
        with (staged/'timings.csv').open('w',newline='') as handle:
            out=csv.DictWriter(handle,fieldnames=columns);out.writeheader();out.writerows(timings)
        (staged/'observations.jsonl').write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in packets))
        for name,value in [('detections.geojson',collection),('audit.json',audit)]:
            (staged/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        # Validate completely before replacing the old empty view. Other videos are untouched.
        target=directory/'overlays/exports';target.mkdir(parents=True,exist_ok=True)
        for previous in target.glob('*.jpg'):previous.unlink()
        for source in (staged/'overlays').glob('*.jpg'):os.replace(source,target/source.name)
        for name in ('exports.mp4','detections.geojson','observations.jsonl','timings.csv','audit.json'):
            os.replace(staged/name,directory/name)
    assert before=={name:digest(directory/f'{name}.mp4') for name in before}
    print(f'{model}: replaced export view; {len(drawn)} unique GeoJSON objects, {decoded} frames; detection/tracking videos unchanged',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='ugv_pui_20261106')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    args=parser.parse_args()
    for model in args.models:refresh(model,args.run)
