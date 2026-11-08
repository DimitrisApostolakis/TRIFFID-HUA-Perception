"""Run a released few-shot detector + selected tracker on one-Hz RGB; no API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
from triffid_ugv_perception.config import NODE_PARAMETERS
from triffid_ugv_perception.models import create_detector
from triffid_ugv_perception.discoveries import ObjectRegistry
from triffid_ugv_perception.tracking.factory import TrackingPipeline
from triffid_ugv_perception.io import write_json


def sample_frames(records):
    """First recorded frame in each second, anchored per source bag/topic."""
    first,last={},{}
    for row in sorted(records,key=lambda r:r['bag_timestamp_ns']):
        key=(row['bag'],row['topic'])
        stamp=int(row['bag_timestamp_ns'])
        first.setdefault(key,stamp)
        bucket=(stamp-first[key])//1_000_000_000
        if bucket>last.get(key,-1):
            last[key]=bucket
            yield row


def main(default_backend="cosine"):
    p=argparse.ArgumentParser(description=__doc__)
    workspace=Path(__file__).resolve().parents[1]
    p.add_argument('--dataset',default=str(workspace/'data/PUI/rgb_frames'))
    p.add_argument('--backend',choices=['opus','pet_dino','cosine','sine','devit'],default=default_backend)
    p.add_argument('--tracker',choices=['botsort','udc'],default='botsort')
    p.add_argument('--udc-proof-threshold',type=float,default=.35,help='UDC lifetime confidence gate (default: 0.35)')
    p.add_argument('--person-roles',action='store_true',help='Track one person identity with temporal role evidence')
    p.add_argument('--save-tracking-cache',action='store_true',help='PET/OPUS: save >=0.1 candidates and appearance for controlled tracker replay')
    p.add_argument('--repo')
    p.add_argument('--precision',choices=['float16','float32'],default='float16')
    p.add_argument('--checkpoint')
    p.add_argument('--background')
    p.add_argument('--output',required=True)
    p.add_argument('--registry',help='Persistent registry path shared by repeated runs of the same experiment/model')
    p.add_argument('--outbox',help='Durable local creation-event directory; defaults beside a shared registry')
    p.add_argument('--bank',default=str(workspace/'ugv_supports/PUI'))
    p.add_argument('--classes',default='')
    p.add_argument('--shots',type=int,choices=range(1,17),default=1)
    p.add_argument('--use-all-supports',action='store_true',help='PET/OPUS: use each class\'s available supports, up to 16')
    p.add_argument('--pet-backbone',choices=['swin-t','swin-l'],default='swin-t')
    p.add_argument('--pet-prompt-mode',choices=['visual','text'],default='visual')
    p.add_argument('--opus-prompt-mode',choices=['visual','mixed'],default='visual')
    p.add_argument('--device',default='cuda')
    p.add_argument('--image-size',type=int,default=640)
    p.add_argument('--max-size',type=int,default=1024)
    p.add_argument('--proposals',type=int,default=100)
    p.add_argument('--confidence',type=float)
    p.add_argument('--nms-iou',type=float,default=.3)
    p.add_argument('--mask-nms-iou',type=float,default=.5)
    p.add_argument('--min-box-area',type=float,default=400)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if not .35<=args.udc_proof_threshold<=1:p.error('--udc-proof-threshold must be between 0.35 and 1')
    if args.use_all_supports and args.backend not in ('pet_dino','opus'):p.error('--use-all-supports requires PET-DINO or OPUS')
    if args.save_tracking_cache and args.backend not in ('pet_dino','opus'):p.error('--save-tracking-cache requires PET-DINO or OPUS')
    if args.pet_prompt_mode=='text' and args.backend!='pet_dino':p.error('Text mode requires PET-DINO')
    if args.pet_prompt_mode=='text' and args.use_all_supports:p.error('Text mode does not use visual supports; omit --use-all-supports')
    if args.pet_backbone=='swin-l' and not args.checkpoint:p.error('Swin-L requires an explicit trained PET-DINO Swin-L checkpoint')
    if args.confidence is None:args.confidence=.7 if args.backend in ('sine','cosine') else .45
    if args.repo is None:
        args.repo=str(Path(NODE_PARAMETERS['sine_repo']) if args.backend=='sine' else workspace/'ugv_models'/args.backend)
    if not .35<=args.confidence<=1:
        p.error('Baseline tracker requires confidence >= 0.35 so all detections enter class-gated association')
    if args.limit<0: p.error('--limit must be nonnegative')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'overlays').mkdir()
    data=Path(args.dataset)
    if not args.classes:
        from triffid_ugv_perception.ontology import CLASSES
        from triffid_ugv_perception.supports import SupportBank
        bank=SupportBank(args.bank)
        # Availability is discovered explicitly and reported; the production
        # backend itself never silently skips requested classes.
        ready=[]
        unavailable=[]
        for item in CLASSES:
            try: bank.select([item],shots=1 if args.use_all_supports else args.shots)
            except (ValueError,OSError): unavailable.append(item.name)
            else: ready.append(item.name)
        args.classes=','.join(ready)
        write_json(out/'class_availability.json',dict(selected=ready,unavailable=unavailable,shots=args.shots))
        if not ready:p.error(f'No classes have {args.shots} valid distinct support images; add annotations or lower --shots')
    cfg=dict(NODE_PARAMETERS,backend=args.backend,classes=args.classes,shots=args.shots,
        support_bank=args.bank,device=args.device,image_size=args.image_size,confidence_threshold=args.confidence,
        tracker=args.tracker,person_roles=args.person_roles,udc_proof_threshold=args.udc_proof_threshold,dataset=str(data.resolve()))
    if args.backend=='devit':
        cfg.update(devit_repo=args.repo,
            devit_checkpoint=args.checkpoint or str(Path(args.repo)/'weights/vits_0089999.pth'),
            devit_background=args.background or str(Path(args.repo)/'weights/background_prototypes.vits14.pth'),
            devit_support_cache=str(workspace/'ugv_models/devit_support_cache'),
            devit_max_size=args.max_size,devit_proposals=args.proposals,
            devit_nms_iou=args.nms_iou,devit_min_box_area=args.min_box_area)
    elif args.backend=='sine':
        cfg.update(sine_repo=args.repo,sine_checkpoint=args.checkpoint or NODE_PARAMETERS['sine_checkpoint'],
            sine_support_cache=str(workspace/'ugv_models/sine_support_cache'),sine_precision=args.precision,sine_mask_nms_iou=args.mask_nms_iou)
    elif args.backend=='opus':
        cfg.update(opus_repo=args.repo,opus_checkpoint=args.checkpoint or str(workspace/'ugv_models/opus_dinov3_convnext_b.pth'),
            opus_support_cache=str(workspace/'ugv_models/opus_support_cache'),opus_precision=args.precision,
            opus_use_all_supports=args.use_all_supports,opus_prompt_mode=args.opus_prompt_mode,
            opus_tracking_floor=.1)
    elif args.backend=='pet_dino':
        cfg.update(pet_repo=args.repo,pet_checkpoint=args.checkpoint or str(workspace/'ugv_models/pet_dino_swin_t.pth'),
            pet_bert=str(workspace/'ugv_models/pet_bert'),pet_support_cache=str(workspace/'ugv_models/pet_support_cache'),
            pet_precision=args.precision,pet_max_size=args.max_size,pet_backbone=args.pet_backbone,pet_use_all_supports=args.use_all_supports,pet_prompt_mode=args.pet_prompt_mode,
            pet_tracking_floor=.1)
    else:
        cfg.update(cosine_repo=args.repo,cosine_checkpoint=args.checkpoint or str(workspace/'ugv_models/cosine_multiscale.bin'),
            cosine_support_cache=str(workspace/'ugv_models/cosine_support_cache'),cosine_precision=args.precision,
            cosine_mask_nms_iou=args.mask_nms_iou)
    write_json(out/'config.json',cfg)
    import torch
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    detector=create_detector(cfg)
    preparation=time.perf_counter()-started
    tracker=TrackingPipeline(args.tracker,person_roles=args.person_roles,udc_proof_threshold=args.udc_proof_threshold)
    registry_context=dict(dataset_index_sha256=hashlib.sha256((data/'frames.jsonl').read_bytes()).hexdigest(),
        backend=args.backend,checkpoint_sha256=detector.metadata()['checkpoint_sha256'])
    if args.tracker!='botsort' or args.person_roles:
        registry_context.update(tracker=args.tracker,person_roles=args.person_roles)
    if args.tracker=='udc' and args.udc_proof_threshold!=.5:
        registry_context.update(udc_proof_threshold=args.udc_proof_threshold)
    if args.backend=='pet_dino' and args.pet_prompt_mode=='text':
        registry_context.update(pet_prompt_mode='text',text_prompts=[item.name for item in detector.classes])
    if args.backend=='opus':
        registry_context.update(opus_prompt_mode=args.opus_prompt_mode,
            support_cache_sha256=hashlib.sha256(Path(detector.cache).read_bytes()).hexdigest(),
            classes=[item.name for item in detector.classes],image_size=args.image_size)
    registry=ObjectRegistry(args.registry or out/'object_registry.json',confirm_frames=2,context=registry_context)
    outbox=Path(args.outbox) if args.outbox else (
        registry.path.parent/(registry.path.stem+'_events') if args.registry else out/'new_objects')
    records=[json.loads(line) for line in (data/'frames.jsonl').open()]
    frames=list(sample_frames(records))
    if args.limit: frames=frames[:args.limit]
    write_json(out/'selected_frames.json',frames)
    times=[];inference_times=[];total_detections=total_tracks=0
    with (out/'frames.jsonl').open('w') as log:
        for number,record in enumerate(frames):
            start=time.perf_counter()
            image=cv2.imread(str(data/record['file']))
            if image is None: raise ValueError(f'Unreadable frame: {record["file"]}')
            infer_start=time.perf_counter()
            detections=detector.detect(image)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            infer_time=time.perf_counter()-infer_start
            track_start=time.perf_counter()
            tracking_detections=detector.last_tracking_detections if args.backend in ('pet_dino','opus') else detections
            if args.save_tracking_cache:
                with (out/'tracking_cache.jsonl').open('a') as cache:
                    cache.write(json.dumps(dict(frame=number,source=record,detections=detector.last_tracking_detections))+'\n')
            tracked=tracker.update(tracking_detections,image,timestamp_ns=record['bag_timestamp_ns'])
            observations,new=registry.update(tracked,record['bag_timestamp_ns'])
            registry.write_creation_events(outbox)
            track_time=time.perf_counter()-track_start
            segmented=[d for d in detections if d.get('mask') is not None]
            if segmented:
                masks=np.stack([d['mask'] for d in segmented])
                mask_dir=out/'masks';mask_dir.mkdir(exist_ok=True)
                mask_file=f'masks/{number:04d}.npz'
                np.savez_compressed(out/mask_file,packed_masks=np.packbits(masks,axis=-1),shape=np.array(masks.shape))
                for index,d in enumerate(segmented):d.update(mask_file=mask_file,mask_index=index)
            overlay=image.copy()
            for d in detections:
                if d.get('mask') is not None:
                    color=((d['class_id']*67+80)%180+75,(d['class_id']*97+30)%180+75,(d['class_id']*43+90)%180+75)
                    mask=d['mask']
                    overlay[mask]=(overlay[mask]*.8+np.array(color)*.2).astype(np.uint8)
                    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay,contours,-1,color,1)
                a,b,c,e=map(int,d['bbox'])
                cv2.rectangle(overlay,(a,b),(c,e),(128,128,128),1)
            for t in observations:
                a,b,c,e=map(int,t['bbox'])
                cv2.rectangle(overlay,(a,b),(c,e),(0,220,0),2)
                identity=t['object_id'][:8] if t['object_id'] else 'confirming'
                cv2.putText(overlay,f"{t['class_name']} t{t['track_id']} {identity}",(max(a,0),max(b-6,16)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,0),1)
            assert cv2.imwrite(str(out/'overlays'/f'{number:04d}.jpg'),overlay)
            elapsed=time.perf_counter()-start
            times.append(elapsed);inference_times.append(infer_time)
            clean=lambda row:{k:v for k,v in row.items() if k not in ('mask','appearance')}
            result=dict(frame=number,source=record,detections=[clean(d) for d in detections],
                tracks=[clean(t) for t in observations],new_objects=new,
                inference_s=infer_time,tracking_and_registry_s=track_time,total_s=elapsed)
            if args.backend in ('pet_dino','opus'):result['candidate_detections']=detector.last_candidates
            log.write(json.dumps(result)+'\n');log.flush()
            total_detections+=len(detections);total_tracks+=len(observations)
            print(f'{number+1}/{len(frames)}: raw={len(detections)} tracked={len(observations)} new={len(new)} total={elapsed:.3f}s',flush=True)
    summary=dict(frames=len(frames),ways=len(detector.classes),shots=0 if args.backend=='pet_dino' and args.pet_prompt_mode=='text' else ('available' if args.use_all_supports else args.shots),
        detections=total_detections,tracked_observations=total_tracks,unique_objects=len(registry.objects),
        tracker_updates=tracker.calls,tracker=tracker.metadata(),preparation_s=preparation,
        timing=dict(mean_s=float(np.mean(times)),p95_s=float(np.percentile(times,95)),max_s=max(times),
                    mean_inference_s=float(np.mean(inference_times)),frames_over_1s=sum(v>1 for v in times),
                    first_frame_s=times[0]),
        backend=args.backend,
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None),
        registry_path=str(registry.path),registry_context=registry_context,outbox_path=str(outbox),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        detector=detector.metadata(),api_enabled=False,
        limitations=['RGB-only registry reidentification requires appearance and image overlap; no guarantee after arbitrary camera motion',
                     'Detections and identity accuracy are not ground-truth evaluated',
                     'Source frames are sampled at 1 Hz; execution is unpaced and every selected frame is processed'])
    write_json(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='detector'},indent=2),flush=True)


if __name__=='__main__':
    main()
