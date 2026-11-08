"""Five-model PUI comparison: videos, JPG overlays, timings and audit only. API off."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
MODELS=('PET-DINO','OPUS','FSOD-VFM','DINOv','YOLOE')
AVAILABLE_MODELS=MODELS+('YOLOE-26-M','YOLOE-26-L')


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def frames(dataset):
    from run_fewshot_baseline import sample_frames
    records=[json.loads(line) for line in (dataset/'frames.jsonl').open()]
    selected=list(sample_frames(records))
    if len({(r['bag'],r['topic']) for r in selected})!=1:raise ValueError('One RGB stream required')
    return selected


def detect(args):
    args.work.mkdir(parents=True,exist_ok=True)
    import torch
    from pui_comparison_detectors import build_detector
    if not torch.cuda.is_available():raise RuntimeError('GPU is unavailable')
    torch.manual_seed(0)
    print(f'{args.model}: initializing model and visual support memory',flush=True)
    begin=time.perf_counter();detector=build_detector(args.model);torch.cuda.synchronize()
    initialization=time.perf_counter()-begin
    selected=frames(args.dataset)
    if args.limit:selected=selected[:args.limit]
    (args.work/'mask_overlays').mkdir(exist_ok=True)
    with (args.work/'detections.jsonl').open('w') as output:
        for index,source in enumerate(selected):
            start=time.perf_counter();image=cv2.imread(str(args.dataset/source['file']))
            if image is None:raise ValueError(f'Unreadable frame: {source["file"]}')
            torch.cuda.synchronize();begin=time.perf_counter()
            accepted=detector.detect(image);torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            candidates=detector.last_tracking_detections
            # Keep mask visualization, never raw masks/tensors in the experiment folder.
            overlay=image.copy();has_masks=False
            for row in accepted:
                mask=row.get('mask')
                if mask is None:continue
                if mask.shape!=image.shape[:2]:raise ValueError('Mask is not at original image resolution')
                color=np.array(color_for(row['class_id']),dtype=np.float32)
                overlay[mask]=(overlay[mask]*.78+color*.22).astype(np.uint8);has_masks=True
            if has_masks:
                if not cv2.imwrite(str(args.work/'mask_overlays'/f'{index:04d}.jpg'),overlay):raise OSError('Could not write mask preview')
            clean=lambda row:{k:v for k,v in row.items() if k not in ('mask','appearance')}
            output.write(json.dumps(dict(frame=index,source=source,detector_s=elapsed,
                detection_stage_s=time.perf_counter()-start,mask_overlay=has_masks,
                detections=[clean(r) for r in accepted],candidates=[clean(r) for r in candidates]))+'\n');output.flush()
            print(f'{args.model} detect {index+1}/{len(selected)}: {len(accepted)} detections, {elapsed:.3f}s',flush=True)
    save_json(args.work/'detector.json',dict(model=detector.metadata(),initialization_s=initialization,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),frames=len(selected),
        runtime=dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())))


def draw_label(canvas,text,point,scale=.47):
    cv2.putText(canvas,text,point,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(canvas,text,point,cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)


def color_for(identity):
    return (int(60+identity*71%195),int(60+identity*113%195),int(60+identity*43%195))


def summary(values):
    return dict(mean_s=float(np.mean(values)),median_s=float(np.median(values)),
                p95_s=float(np.percentile(values,95)),max_s=float(max(values))) if values else None


def track(args):
    import torch
    from triffid_ugv_perception.tracking.factory import TrackingPipeline
    from triffid_ugv_perception.discovery import DiscoveryService, DiscoverySettings
    from pui_comparison_detectors import digest
    correction=getattr(args,'class_correction',False)
    begin=time.perf_counter();tracker=TrackingPipeline('masa',person_roles=True,class_correction=correction);torch.cuda.synchronize()
    tracker_init=time.perf_counter()-begin
    settings=DiscoverySettings(namespace=f'PUI/original/{args.model}',embedding_space='MASA-R50:'+tracker.engine.checkpoint_hash,class_correction=correction)
    service=DiscoveryService(args.work/'discoveries.sqlite3',settings)
    metadata=json.loads((args.work/'detector.json').read_text())
    rows=[json.loads(line) for line in (args.work/'detections.jsonl').open()]
    from pui_comparison_views import (VideoSet,competing_labels,new_ready_events,
        summarize_packets,install_replacement,geojson_collection,VIEW_SCHEMA)
    staged=args.work/'rendered'
    staged.mkdir(parents=True,exist_ok=True)
    videos=VideoSet(staged,args.model,len(rows))
    packets=[];seen_exports=set()
    times=[];counts=dict(detections=0,track_observations=0,new_objects=0,local_update_events=0,class_updates=0,api_ready_creations=0,api_sent=0,geojson_features=0)
    columns=['frame','timestamp_ns','detector_s','tracker_s','registry_s','tracking_stage_s','staged_sum_s','detections','tracks','new_objects','local_updates','api_ready_creations','geojson_exports']
    try:
        with (staged/'timings.csv').open('w',newline='') as csvfile, (staged/'observations.jsonl').open('w') as ledger:
            writer=csv.DictWriter(csvfile,fieldnames=columns);writer.writeheader()
            for row in rows:
                start=time.perf_counter();image=cv2.imread(str(args.dataset/row['source']['file']))
                if image is None:raise ValueError('Unreadable RGB frame')
                torch.cuda.synchronize();begin=time.perf_counter()
                tracked=tracker.update(row['candidates'],image,row['source']['bag_timestamp_ns'])
                torch.cuda.synchronize();tracker_s=time.perf_counter()-begin
                begin=time.perf_counter();result,_=service.process(tracked,[None]*len(tracked),row['source']['bag_timestamp_ns'])
                registry_s=time.perf_counter()-begin
                clean=lambda item:{k:v for k,v in item.items() if k not in ('mask','appearance')}
                ready=new_ready_events(service.store.ready_event_ids(),seen_exports)
                packet=dict(frame=row['frame'],source=row['source'],detections=row['detections'],
                    tracks=[clean(item) for item in result.observations],new_object_ids=result.created_ids,
                    updated_object_ids=result.updated_ids or [],class_correction=correction,
                    geojson_export_ids=list(result.created_ids),
                    api_ready_event_ids=ready,api_queue_pending=service.status()['pending'],api_sent=0,
                    api_enabled=False,world_coordinates_available=False,
                    competing_labels=competing_labels(row['detections'],result.observations))
                mask_base=(cv2.imread(str(args.work/'mask_overlays'/f"{row['frame']:04d}.jpg"))
                           if row['mask_overlay'] else None)
                videos.write(image,packet,mask_base)
                packets.append(packet)
                ledger.write(json.dumps(packet,allow_nan=False)+'\n');ledger.flush()
                stage=time.perf_counter()-start
                timing=dict(frame=row['frame'],timestamp_ns=row['source']['bag_timestamp_ns'],detector_s=row['detector_s'],
                    tracker_s=tracker_s,registry_s=registry_s,tracking_stage_s=stage,
                    staged_sum_s=row['detection_stage_s']+stage,detections=len(row['detections']),
                    tracks=len(tracked),new_objects=len(result.created_ids),local_updates=len(result.updated_ids),api_ready_creations=len(ready),geojson_exports=len(result.created_ids))
                times.append(timing);writer.writerow(timing);csvfile.flush()
                counts['detections']+=len(row['detections']);counts['track_observations']+=len(tracked)
                counts['new_objects']+=len(result.created_ids);counts['api_ready_creations']+=len(ready);counts['geojson_features']+=len(result.created_ids)
                counts['local_update_events']+=len(result.updated_ids)
                counts['class_updates']+=sum(bool(t.get('class_update') and t['class_update']['kind']=='class') for t in result.observations)
                print(f"{args.model} views {row['frame']+1}/{len(rows)}: tracks={len(tracked)} new={len(result.created_ids)} API-ready={len(ready)}",flush=True)
    finally:
        videos.close()
        service.close()
    bank=json.loads((ROOT/'ugv_supports/PUI/import_report.json').read_text())
    audit=dict(status='complete',artifact_schema=VIEW_SCHEMA,model=args.model,model_metadata=metadata['model'],class_correction=correction,
        videos=dict(detections='detections.mp4',tracking='tracking.mp4',exports='exports.mp4'),
        export_video_policy=('NEW creates a local UUID once; UPDATED revises that same local feature. API OFF.' if correction else
                             'Newly confirmed GeoJSON UUIDs once per discovery; geometry may be null; independent of the API queue; never sends'),
        export_location_available=False,
        class_competition=summarize_packets(packets),
        detector_runtime=metadata.get('runtime'),
        tracker_runtime=dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
        support_export_sha256=bank['export_sha256'],bank_sha256=digest(ROOT/'ugv_supports/PUI/bank.json'),
        class_support_counts={k:v['imported'] for k,v in bank['classes'].items() if v['imported']},
        classes=bank['one_shot_classes'],ways=len(bank['one_shot_classes']),frames=len(rows),**counts,
        input_dataset=str(args.dataset),dataset_index_sha256=digest(args.dataset/'frames.jsonl'),
        sampling_hz=1,detector_initialization_s=metadata['initialization_s'],tracker_initialization_s=tracker_init,
        tracker=tracker.metadata(),thresholds=dict(detection=.35,recovery=.1),
        timing_summary={key:summary([r[key] for r in times]) for key in ('detector_s','tracker_s','registry_s','staged_sum_s')},
        compute_only_summary=summary([r['detector_s']+r['tracker_s']+r['registry_s'] for r in times]),
        cold_detector_s=times[0]['detector_s'],warm_detector=summary([r['detector_s'] for r in times[1:]]),
        timing_note='Detector and tracker run in separate environments; staged_sum_s includes both stages and duplicated I/O. It is not measured live end-to-end latency.',
        detector_peak_cuda_allocated_bytes=metadata['peak_cuda_allocated_bytes'],api_enabled=False,
        accuracy_evaluated=False,ground_truth_available=False,support_query_overlap='Site supports; image-byte overlap checked below; no held-out accuracy claim',
        limitations=['No accuracy or tracking metrics evaluated','Unpaced execution processes every selected 1 Hz frame',
                     'RGB-only identities; no geographic output or pose accuracy evaluated',
                     'Displayed confidences are native model scores, not calibrated probabilities; a shared threshold does not equalize precision'])
    query_hashes={digest(args.dataset/r['source']['file']) for r in rows}
    support_hashes={r['sha256'] for c in bank['classes'].values() for r in c['examples']}
    audit['support_query_exact_image_overlap_count']=len(query_hashes&support_hashes)
    if correction:
        from dataclasses import asdict
        from triffid_ugv_perception.tracking.class_correction import AssociationSettings
        audit['association_settings']=asdict(AssociationSettings())
        audit['class_evidence_policy']=dict(window_seconds=5,reliable_threshold=.35,
            minimum_support=2,recent_reliable_observations=3,normalized_vote_lead=.15,
            decay_per_second=.8,confidence_update_step=.10)
    save_json(staged/'detections.geojson',geojson_collection(packets,args.model))
    save_json(staged/'audit.json',audit)
    install_replacement(staged,args.output,len(rows))


def environment(model):
    base=os.environ.copy();base.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',YOLO_AUTOINSTALL='false',
        MPLCONFIGDIR=str(ROOT/'ugv_models/runtime_config/matplotlib'),YOLO_CONFIG_DIR=str(ROOT/'ugv_models/runtime_config/yoloe'))
    paths=['src/triffid_ugv_perception']
    if model=='PET-DINO':
        python='/home/oracle/miniforge3/envs/fewshot-dev/bin/python';paths+=['ugv_models/pet_runtime','ugv_models/cosine_runtime','ugv_models/devit_runtime','ugv_models/pet_dino']
    elif model=='MASA' or model=='OPUS':
        python=str(ROOT/'ugv_models/opus_env/bin/python');paths+=['ugv_models/opus/third_party/mmdetection','ugv_models/opus']
    elif model in ('YOLOE-26-M','YOLOE-26-L'):
        python=str(ROOT/'ugv_models/yoloe26_env/bin/python')
    elif model=='YOLOE':
        python=str(ROOT/'ugv_models/yoloe_env/bin/python');paths+=['ugv_models/yoloe']
    elif model=='DINOv':
        python=str(ROOT/'ugv_models/dinov_env/bin/python');paths+=['ugv_models/dinov','ugv_models/msda_bridge','ugv_models/pet_runtime','ugv_models/devit_runtime']
    else:
        python=str(ROOT/'ugv_models/fsod_vfm_env/bin/python');paths+=['ugv_models/fsod_vfm','ugv_models/msda_bridge']
    base['PYTHONPATH']=':'.join(str(ROOT/p) for p in paths)
    return python,base


def run(args):
    work_root=ROOT/'ugv_models/comparison_work';work_root.mkdir(exist_ok=True)
    models=args.models or list(MODELS)
    failures=0
    from pui_comparison_detectors import digest
    active_bank=digest(ROOT/'ugv_supports/PUI/bank.json')
    active_dataset=digest(args.dataset/'frames.jsonl')
    run_root=ROOT/'samples'
    for model in models:
        output=run_root/model/'original'/args.run
        if not args.force and output.exists() and (output/'audit.json').exists():
            saved=json.loads((output/'audit.json').read_text())
            from pui_comparison_views import VIEW_SCHEMA
            if saved.get('status')=='complete' and saved.get('artifact_schema')==VIEW_SCHEMA:
                if saved.get('class_correction',False)!=getattr(args,'class_correction',False):
                    raise ValueError('Existing run has a different identity policy; choose a new --run or use --force')
                if saved.get('bank_sha256')!=active_bank or saved.get('dataset_index_sha256')!=active_dataset:
                    raise ValueError('Existing run has a different bank/dataset; choose a new --run or use --force')
                print(f'{model}: already complete, skipping',flush=True);continue
        work=work_root/args.run/model;work.mkdir(parents=True,exist_ok=True)
        command=[str(ROOT/'scripts/run_pui_comparison.py'),'--model',model,'--dataset',str(args.dataset),
                 '--output',str(output),'--work',str(work),'--limit',str(args.limit)]
        if getattr(args,'class_correction',False):command.append('--class-correction')
        try:
            for stage,environment_name in [('detect',model),('track','MASA')]:
                if stage=='detect' and (work/'detector.json').exists():continue
                python,env=environment(environment_name)
                subprocess.run([python,*command,'--stage',stage],env=env,cwd=str(ROOT),check=True)
            # Only the four requested artifact groups remain in samples; temporary predictions are removed.
            shutil.rmtree(work)
        except Exception as error:
            failures+=1
            output.mkdir(parents=True,exist_ok=True)
            failure=dict(status='failed',model=model,error=str(error),api_enabled=False,accuracy_evaluated=False)
            if not (output/'audit.json').exists():save_json(output/'audit.json',failure)
            else:save_json(work/'failed_attempt.json',failure)
            print(f'{model}: FAILED: {error}; intermediate state retained at {work}',flush=True)
    return 1 if failures else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['run','detect','track'],default='run')
    parser.add_argument('--models',nargs='+',choices=AVAILABLE_MODELS)
    parser.add_argument('--model',choices=AVAILABLE_MODELS)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/PUI/rgb_frames')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--run',default='ugv_pui_20261106')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--class-correction',action='store_true',help='Separate identity from temporal class evidence; update local GeoJSON UUIDs')
    parser.add_argument('--force',action='store_true',help='Regenerate selected current-run artifacts')
    args=parser.parse_args()
    if args.limit<0:parser.error('limit must be nonnegative')
    if args.stage!='run' and (not args.model or not args.output or not args.work):parser.error('stages require --model, --output, --work')
    if args.stage=='run':raise SystemExit(run(args))
    elif args.stage=='detect':detect(args)
    else:track(args)


if __name__=='__main__':main()
