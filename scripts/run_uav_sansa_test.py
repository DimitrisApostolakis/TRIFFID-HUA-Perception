"""Timed UAV few-shot baselines on preselected angular views. API/tracking off."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from triffid_uav_perception.ontology import CLASSES
from triffid_uav_perception.supports import SupportBank
from triffid_uav_perception.models.sansa import SansaBackend
from triffid_uav_perception.pipeline import infer_source
from triffid_uav_perception.export import document,write_json
from triffid_uav_perception.media import read_rgb
from triffid_uav_perception.visualize import render_observations

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
CLIP='DJI_20261019155500_0001_S'
MANIFEST=ROOT/f'data/UAV-PUI/{CLIP}/selected_views/selected_frames.json'
OUTPUT=ROOT/f'samples/SANSA/UAV/{CLIP}/one_shot_640'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class TimedBackend:
    def __init__(self,backend):self.backend=backend;self.rows=[];self.frame=None
    def predict(self,rgb,classes):
        iterator=iter(self.backend.predict(rgb,classes))
        for item in classes:
            torch.cuda.synchronize();start=time.perf_counter()
            prediction=next(iterator)
            torch.cuda.synchronize()
            self.rows.append(dict(view=self.frame,class_id=item.id,class_name=item.name,
                backend_s=time.perf_counter()-start))
            yield prediction
        try:next(iterator)
        except StopIteration:return
        raise ValueError('Backend returned an extra class')


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['sansa','fss-sam3','fss-sam3-text'],default='sansa')
    args=parser.parse_args()
    model_name='SANSA' if args.backend=='sansa' else 'FSS-SAM3'
    if args.backend.startswith('fss-sam3'):
        mode='text_visual' if args.backend=='fss-sam3-text' else 'visual'
        OUTPUT=ROOT/f'samples/FSS-SAM3/UAV/{CLIP}/one_shot_1008_{mode}'
    if OUTPUT.exists():raise ValueError('Output already exists; preserve the completed run')
    payload=json.loads(MANIFEST.read_text())
    report=json.loads((ROOT/'uav_supports/PUI/import_report.json').read_text())
    classes=tuple(c for c in CLASSES if c.geometry=='Point' and c.name in report['one_shot_classes'])
    assert {c.name for c in classes}=={'first responder','citizen','window','fire truck'}
    selection=SupportBank(ROOT/'uav_supports/PUI').select(classes,shots=1)
    torch.manual_seed(0);torch.set_num_threads(4)
    if args.backend=='sansa':
        backend=SansaBackend('/mnt/d/Models/ThirdParty/SANSA',
            '/mnt/d/Models/Checkpoints/SANSA/adapter_generalist.pth',
            '/mnt/d/Models/Checkpoints/SANSA/sam2_hiera_large.pt',device='cuda',image_size=640,threshold=.5)
    else:
        from triffid_uav_perception.models.fss_sam3 import FssSam3Backend
        backend=FssSam3Backend('/mnt/d/Models/ThirdParty/FSS-SAM3',
            '/mnt/d/Models/Checkpoints/FSS-SAM3/sam3.pt',
            '/mnt/d/Models/Checkpoints/FSS-SAM3/bpe_simple_vocab_16e6.txt.gz',use_text=args.backend=='fss-sam3-text')
    start=time.perf_counter();backend.prepare(classes,selection);torch.cuda.synchronize()
    preparation=time.perf_counter()-start
    OUTPUT.mkdir(parents=True);(OUTPUT/'overlays').mkdir()
    timed=TimedBackend(backend);sources=[];times=[];video=None
    torch.cuda.reset_peak_memory_stats()
    try:
        # Temporal order for viewing. Source IDs, telemetry and selected pixels are unchanged.
        for index,original in enumerate(sorted(payload['sources'],key=lambda s:s['timestamp_s'])):
            begin=time.perf_counter();source=dict(original);source['observations']=[]
            path=(MANIFEST.parent/source['image_file']).resolve();rgb=read_rgb(path)
            load=time.perf_counter()-begin;source['image_file']=str(path)
            timed.frame=source['source_id']
            torch.cuda.synchronize();start=time.perf_counter()
            infer_source(timed,rgb,classes,source,min_area_px=1)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            sources.append(source)
            canvas,_=render_observations(rgb,source['observations'])
            bgr=cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
            title=f"{model_name} | {index+1}/8 angular views | t={source['timestamp_s']:.2f}s | inference {elapsed:.3f}s | API OFF"
            cv2.rectangle(bgr,(0,0),(bgr.shape[1],42),(20,20,20),-1)
            cv2.putText(bgr,title,(12,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            if not cv2.imwrite(str(OUTPUT/'overlays'/f'{index:04d}.jpg'),bgr):raise OSError('Cannot save overlay')
            if video is None:
                video=cv2.VideoWriter(str(OUTPUT/'detections.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1.,(rgb.shape[1],rgb.shape[0]))
                if not video.isOpened():raise OSError('Cannot create preview')
            video.write(bgr)
            row=dict(view=source['source_id'],frame_index=source['frame_index'],timestamp_s=source['timestamp_s'],
                load_s=load,inference_components_s=elapsed,
                backend_s=sum(r['backend_s'] for r in timed.rows if r['view']==source['source_id']),
                detections=len(source['observations']),total_with_render_s=time.perf_counter()-begin)
            times.append(row)
            print(f"{index+1}/8 {source['source_id']}: {elapsed:.3f}s, {row['detections']} observations",flush=True)
    finally:
        if video is not None:video.release()
    result=document('point_detections',sources,backend.metadata(),classes)
    result.update(video=payload['video'],srt_file=payload['srt_file'],selection=payload['selection'],
        support_bank=selection.metadata(),postprocessing=dict(min_area_px=1,connectivity=8))
    write_json(OUTPUT/'point_detections.json',result)
    for name,rows in [('timings.csv',times),('class_timings.csv',timed.rows)]:
        with (OUTPUT/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    elapsed=np.array([r['inference_components_s'] for r in times])
    audit=dict(status='complete',model=backend.metadata(),model_preparation_s=preparation,
        ways=len(classes),shots=1,classes=[c.name for c in classes],selected_views=len(sources),
        source_manifest=str(MANIFEST),source_manifest_sha256=digest(MANIFEST),
        bank_sha256=digest(ROOT/'uav_supports/PUI/bank.json'),
        checkpoint_sha256=digest(backend.checkpoint),
        base_uav_commit='285c68dc05865ba55ebfcbe25d55dd0957304786',
        local_changes=['Separate human roles in ontology and support bank; original source IDs preserved'],
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),precision='float32'),
        mean_inference_components_s=float(elapsed.mean()),p95_inference_components_s=float(np.percentile(elapsed,95)),
        first_inference_s=float(elapsed[0]),warm_mean_inference_components_s=float(elapsed[1:].mean()),
        per_class_backend_mean_s={c.name:float(np.mean([r['backend_s'] for r in timed.rows if r['class_id']==c.id])) for c in classes},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),observations=sum(len(s['observations']) for s in sources),
        api_enabled=False,tracking_enabled=False,accuracy_evaluated=False,
        preview_note='Eight sparse angular views in timestamp order at 1 preview FPS; not a full-frame-rate video.',
        timing_note='CUDA-synchronized class inference includes probability resize/transfer; frame inference also includes connected components. Loading/rendering and preparation are separate.',
        limitations=['Independent one-class episodes, not native joint K-way inference',
            'Semantic connected components are not learned object instances',
            'Probability threshold .5; no extra component-score filter; minimum component area 1 original-image pixel',
            'Only supported Point classes; polygon classes require the separate top-view workflow',
            'No tracking, duplicate suppression, geographic projection or API export in this run'])
    write_json(OUTPUT/'audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
