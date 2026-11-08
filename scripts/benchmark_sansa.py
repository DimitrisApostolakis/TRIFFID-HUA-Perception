"""Benchmark the existing SANSA adapter on the exact offline comparison frames."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from triffid_ugv_perception.models.sansa import SansaBackend
from triffid_ugv_perception.models.semantic import SemanticDetector
from triffid_ugv_perception.ontology import CLASSES, BY_NAME
from triffid_ugv_perception.supports import SupportBank
from triffid_ugv_perception.io import write_json

ROOT=Path(__file__).absolute().parents[1]
output=ROOT/'archive/samples_legacy_20261106/sansa_baseline'
output.mkdir(exist_ok=True)
bank=SupportBank(ROOT/'ugv_supports/PUI')
reference=ROOT/'archive/samples_legacy_20261106/cosine_baseline/efficient_644'
frames=json.loads((reference/'selected_frames.json').read_text())
backend=SansaBackend('/mnt/d/Models/ThirdParty/SANSA',
    '/mnt/d/Models/Checkpoints/SANSA/adapter_generalist.pth',
    '/mnt/d/Models/Checkpoints/SANSA/sam2_hiera_large.pt',device='cuda',image_size=640)
ready=[]
for cls in CLASSES:
    try:bank.select([cls],shots=1)
    except ValueError:continue
    ready.append(cls)
for name,classes in [('one_way',[BY_NAME['first responder']]),('all_classes',ready)]:
    destination=output/name
    if (destination/'summary.json').exists():continue
    destination.mkdir(exist_ok=True);(destination/'overlays').mkdir(exist_ok=True)
    start=time.perf_counter()
    detector=SemanticDetector(backend,classes,bank.select(classes,shots=1),confidence_threshold=.7,min_area_px=400)
    torch.cuda.synchronize();preparation=time.perf_counter()-start
    times=[];totals=[];count=0
    selected=frames if name=='one_way' else [frames[index] for index in (0,10,20,30,40,50,64)]
    torch.cuda.reset_peak_memory_stats()
    with (destination/'frames.jsonl').open('w') as handle:
        for number,record in enumerate(selected):
            start=time.perf_counter()
            image=cv2.imread(str(ROOT/'data/PUI/rgb_frames'/record['file']))
            torch.cuda.synchronize();infer_start=time.perf_counter()
            detections=detector.detect(image)
            torch.cuda.synchronize();elapsed=time.perf_counter()-infer_start
            for detection in detections:
                x1,y1,x2,y2=map(int,detection['bbox'])
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)
                cv2.putText(image,detection['class_name'],(x1,max(y1-4,14)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
            cv2.imwrite(str(destination/'overlays'/f'{number:04d}.jpg'),image)
            total=time.perf_counter()-start
            times.append(elapsed);totals.append(total);count+=len(detections)
            handle.write(json.dumps(dict(frame=number,source=record,inference_s=elapsed,total_s=total,
                detections=[{k:v for k,v in d.items() if k!='mask'} for d in detections]))+'\n');handle.flush()
            print(f'{name} {number+1}/{len(selected)}: {elapsed:.3f}s inference, {len(detections)} components',flush=True)
    write_json(destination/'summary.json',dict(frames=len(selected),ways=len(classes),shots=1,precision='float32',
        classes=[c.name for c in classes],device=torch.cuda.get_device_name(),preparation_s=preparation,
        inference_mean_s=float(np.mean(times)),inference_p95_s=float(np.percentile(times,95)),
        warm_inference_mean_s=float(np.mean(times[1:])),warm_inference_p95_s=float(np.percentile(times[1:],95)),
        first_inference_s=times[0],inference_frames_over_1s=sum(t>1 for t in times),
        total_mean_s=float(np.mean(totals)),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        detections=count,detector=detector.metadata(),api_enabled=False,
        limitations=['Independent per-class episodes; supports and query re-encoded for every class/frame',
            'Timing includes segmentation and semantic-component conversion; excludes tracking and registry',
            'Components are not native instance predictions; accuracy not ground-truth evaluated']))
