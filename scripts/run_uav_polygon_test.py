"""Polygon-class segmentation of the user-selected original frame 0; pixel output."""
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from triffid_uav_perception.ontology import CLASSES
from triffid_uav_perception.supports import SupportBank
from triffid_uav_perception.models.fss_sam3 import FssSam3Backend
from triffid_uav_perception.masks import observations_from_prediction,decode_rle
from triffid_uav_perception.visualize import render_observations,class_color
from triffid_uav_perception.export import document,write_json
from triffid_uav_perception.media import read_rgb
from run_uav_sansa_test import digest

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
INPUT=ROOT/'data/UAV-PUI/DJI_20261019155500_0001_S/frame_000000'
OUT=ROOT/'samples/FSS-SAM3/UAV/DJI_20261019155500_0001_S/frame_000000_polygons_text_visual'


def main():
    if OUT.exists():raise ValueError('Output exists; preserve the completed test')
    image=INPUT/'image.png';rgb=read_rgb(image);context=json.loads((INPUT/'source.json').read_text())
    report=json.loads((ROOT/'uav_supports/PUI/import_report.json').read_text())
    classes=tuple(c for c in CLASSES if c.geometry=='Polygon' and c.name in report['one_shot_classes'])
    assert len(classes)==6 and any(c.name=='green tree' for c in classes)
    supports=SupportBank(ROOT/'uav_supports/PUI').select(classes,shots=1)
    torch.manual_seed(0);torch.set_num_threads(4)
    backend=FssSam3Backend('/mnt/d/Models/ThirdParty/FSS-SAM3',
        '/mnt/d/Models/Checkpoints/FSS-SAM3/sam3.pt',
        '/mnt/d/Models/Checkpoints/FSS-SAM3/bpe_simple_vocab_16e6.txt.gz',use_text=True)
    start=time.perf_counter();backend.prepare(classes,supports);torch.cuda.synchronize()
    preparation=time.perf_counter()-start
    OUT.mkdir(parents=True);(OUT/'overlays').mkdir();(OUT/'masks').mkdir()
    source=dict(source_id='frame_000000',source_type='top_view_image',input_origin='video_frame',image_file=str(image),
        width=rgb.shape[1],height=rgb.shape[0],frame_index=0,timestamp_s=0.,
        telemetry=context['telemetry'],view_geometry='oblique original frame; polygon segmentation task',
        projection_context=None,srt_file=str(INPUT/'image.srt'),observations=[])
    rows=[];iterator=iter(backend.predict(rgb,classes));torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();all_start=time.perf_counter()
    for item in classes:
        torch.cuda.synchronize();start=time.perf_counter();prediction=next(iterator);torch.cuda.synchronize()
        backend_s=time.perf_counter()-start
        if prediction.class_id!=item.id:raise ValueError('Prediction order/class mismatch')
        start=time.perf_counter()
        observations=observations_from_prediction(prediction,item,source['source_id'],min_area_px=1)
        component_s=time.perf_counter()-start;source['observations'].extend(observations)
        rows.append(dict(class_id=item.id,class_name=item.name,backend_s=backend_s,components_s=component_s,
            inference_components_s=backend_s+component_s,components=len(observations),
            foreground_pixels=int(prediction.mask.sum())))
        print(f'{item.name}: {backend_s+component_s:.3f}s, {len(observations)} components, {int(prediction.mask.sum())} pixels',flush=True)
    torch.cuda.synchronize();inference=time.perf_counter()-all_start
    try:next(iterator)
    except StopIteration:pass
    else:raise ValueError('Unexpected extra class')
    for item in classes:
        selected=[o for o in source['observations'] if o['class_id']==item.id]
        mask=np.zeros(rgb.shape[:2],np.uint8)
        for o in selected:
            crop=decode_rle(o['mask']);x,y=o['mask']['offset_xy']
            mask[y:y+crop.shape[0],x:x+crop.shape[1]]|=crop.astype(np.uint8)*255
        name=item.name.replace(' ','_')
        if not cv2.imwrite(str(OUT/'masks'/f'{name}.png'),mask):raise OSError('Cannot write class mask')
        canvas,_=render_observations(rgb,selected,boxes=False,centroids=False,labels=False)
        bgr=cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
        cv2.putText(bgr,f'FSS-SAM3 | frame 0 | {item.name} | {len(selected)} components',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        if not cv2.imwrite(str(OUT/'overlays'/f'{name}.jpg'),bgr):raise OSError('Cannot write class overlay')
    canvas,_=render_observations(rgb,source['observations'],boxes=False,centroids=False,labels=False)
    bgr=cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr,(0,0),(690,55+len(classes)*28),(20,20,20),-1)
    cv2.putText(bgr,'FSS-SAM3 | frame 0 | pixel masks | pitch -58.1 deg',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    for index,item in enumerate(classes):
        y=52+index*28;color=tuple(reversed(class_color(item.name)))
        cv2.rectangle(bgr,(12,y-10),(28,y+5),color,-1)
        n=sum(o['class_id']==item.id for o in source['observations'])
        cv2.putText(bgr,f'{item.name}: {n} components',(38,y+3),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    if not cv2.imwrite(str(OUT/'overlay.jpg'),bgr):raise OSError('Cannot write combined overlay')
    result=document('polygon_segmentation',[source],backend.metadata(),classes)
    result.update(support_bank=supports.metadata(),source_context=context,
        postprocessing=dict(min_area_px=1,connectivity=8))
    write_json(OUT/'polygon_segmentations.json',result)
    with (OUT/'timings.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    audit=dict(status='complete',model=backend.metadata(),ways=len(classes),shots=1,
        classes=[c.name for c in classes],image=str(image),image_sha256=digest(image),
        source_video=context['video'],frame_index=0,timestamp_s=0.,
        gimbal_pitch_deg=context['telemetry']['gimbal_pitch'],view_geometry=source['view_geometry'],
        srt_file=str(INPUT/'image.srt'),srt_sha256=digest(INPUT/'image.srt'),
        bank_sha256=digest(ROOT/'uav_supports/PUI/bank.json'),checkpoint_sha256=digest(backend.checkpoint),
        model_support_preparation_s=preparation,inference_components_s=inference,
        per_class_timings=rows,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        observations=len(source['observations']),api_enabled=False,tracking_enabled=False,
        geographic_projection=False,accuracy_evaluated=False,
        limitations=['User-selected frame is oblique (-58.1 degrees), not orthorectified',
            'Six independent one-class collage episodes; semantic components are not individual tree instances',
            'Single image timing includes cold first prediction; no warm-frame average measured',
            'Pixel-space masks only; no world-coordinate polygon or footprint estimation'])
    write_json(OUT/'audit.json',audit)
    print(json.dumps(dict(output=str(OUT),preparation_s=preparation,inference_s=inference,components=len(source['observations'])),indent=2),flush=True)


if __name__=='__main__':main()
