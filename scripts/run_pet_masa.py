"""PET-DINO detections followed by MASA-R50 tracking; JPG/JSON only, API off."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path('/mnt/d/projects/triffid-hua-perception')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','detect','track'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'samples/PET-DINO/PUI/run')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/PUI/rgb_frames')
    parser.add_argument('--all-frames',action='store_true',help='Use every frame, including subsecond pairs')
    args=parser.parse_args();out=args.output
    if args.stage=='all':
        out.mkdir(parents=True,exist_ok=False)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                                        MPLCONFIGDIR='/tmp/triffid_masa_mpl',TOKENIZERS_PARALLELISM='false')
        for stage,python,paths in [
            ('detect','/home/oracle/miniforge3/envs/fewshot-dev/bin/python',
             ['ugv_models/pet_runtime','ugv_models/cosine_runtime','ugv_models/devit_runtime','ugv_models/pet_dino']),
            ('track',str(ROOT/'ugv_models/opus_env/bin/python'),['ugv_models/opus/third_party/mmdetection'])]:
            env['PYTHONPATH']=':'.join(str(ROOT/p) for p in paths+['src/triffid_ugv_perception'])
            command=[python,__file__,'--stage',stage,'--output',str(out),'--dataset',str(args.dataset)]
            if args.all_frames:command.append('--all-frames')
            subprocess.run(command,env=env,check=True)
        return
    import torch
    dataset=args.dataset
    if args.stage=='detect':
        from run_fewshot_baseline import sample_frames
        from triffid_ugv_perception.models.pet_dino import PetDinoDetector
        from triffid_ugv_perception.config import NODE_PARAMETERS
        report=json.loads((ROOT/'ugv_supports/PUI/import_report.json').read_text())
        cfg=dict(NODE_PARAMETERS,device='cuda',classes=','.join(report['one_shot_classes']),
            shots=1,support_bank=str(ROOT/'ugv_supports/PUI'),image_size=800,confidence_threshold=.35,
            pet_tracking_floor=.1,pet_prompt_mode='visual',pet_use_all_supports=True,
            pet_repo=str(ROOT/'ugv_models/pet_dino'),pet_checkpoint=str(ROOT/'ugv_models/pet_dino_swin_t.pth'),
            pet_bert=str(ROOT/'ugv_models/pet_bert'),pet_support_cache=str(ROOT/'ugv_models/pet_support_cache'),
            pet_precision='float16',pet_max_size=1333,pet_backbone='swin-t')
        start=time.perf_counter();detector=PetDinoDetector(cfg);preparation=time.perf_counter()-start
        records=[json.loads(x) for x in (dataset/'frames.jsonl').read_text().splitlines()]
        frames=records if args.all_frames else list(sample_frames(records))
        if len({(x['bag'],x['topic']) for x in frames})!=1:raise ValueError('Expected one RGB stream')
        times=[]
        with (out/'detections.jsonl').open('w') as log:
            for number,source in enumerate(frames):
                image=cv2.imread(str(dataset/source['file']))
                begin=time.perf_counter();accepted=detector.detect(image);torch.cuda.synchronize()
                duration=time.perf_counter()-begin;times.append(duration)
                clean=lambda d:{k:v for k,v in d.items() if k not in ('appearance','mask')}
                log.write(json.dumps(dict(frame=number,source=source,detector_s=duration,
                    detections=[clean(d) for d in accepted],tracking_candidates=[clean(d) for d in detector.last_tracking_detections]))+'\n')
                log.flush()
        metadata=dict(detector=detector.metadata(),preparation_s=preparation,frames=len(frames),
            confidence=.35,tracking_floor=.1,support_export_sha256=report['export_sha256'],
            dataset=str(dataset),sampling='all_frames' if args.all_frames else 'first_frame_per_recorded_second',
            support_query_overlap='Supports come from this site; support/query independence is not established')
        (out/'detector.json').write_text(json.dumps(metadata,indent=2))
        print(f'PET: {len(frames)} frames; {len(detector.classes)} classes; {sum(detector.metadata()["shots_per_class"].values())} supports; mean {np.mean(times):.3f}s',flush=True)
        return
    from triffid_ugv_perception.tracking.factory import TrackingPipeline
    from triffid_ugv_perception.discoveries import ObjectRegistry
    tracker=TrackingPipeline('masa',person_roles=True)
    context=dict(detector=json.loads((out/'detector.json').read_text())['detector']['checkpoint_sha256'],
                 bank_sha256=hashlib.sha256((ROOT/'ugv_supports/PUI/bank.json').read_bytes()).hexdigest(),
                 tracker=tracker.metadata())
    registry=ObjectRegistry(out/'object_registry.json',context=context)
    frames=[json.loads(x) for x in (out/'detections.jsonl').read_text().splitlines()]
    (out/'overlays').mkdir(exist_ok=False)
    tracker_times=[];estimated=[];events=[];observations_count=raw_count=0;video=None
    with (out/'tracks.jsonl').open('w') as log:
        for row in frames:
            begin=time.perf_counter();image=cv2.imread(str(dataset/row['source']['file']))
            infer=time.perf_counter()
            tracks=tracker.update(row['tracking_candidates'],image,row['source']['bag_timestamp_ns'])
            torch.cuda.synchronize();tracker_s=time.perf_counter()-infer
            obs,new=registry.update(tracks,row['source']['bag_timestamp_ns'])
            ids=[t['object_id'] for t in obs if t['object_id']]
            assert len(ids)==len(set(ids))
            events.extend(new);observations_count+=len(obs);raw_count+=len(row['detections'])
            for d in row['detections']:
                a,b,c,e=map(int,d['bbox']);cv2.rectangle(image,(a,b),(c,e),(130,130,130),1)
            for t in obs:
                a,b,c,e=map(int,t['bbox']);identity=t['object_id'][:6] if t['object_id'] else 'confirming'
                color=(int(60+t['track_id']*71%195),int(60+t['track_id']*113%195),int(60+t['track_id']*43%195))
                cv2.rectangle(image,(a,b),(c,e),color,2)
                cv2.putText(image,f"{t['class_name']} {t['confidence']:.1%} T{t['track_id']} {identity}",(max(0,a),max(18,b-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
            assert cv2.imwrite(str(out/'overlays'/f"{row['frame']:04d}.jpg"),image)
            if video is None:
                video=cv2.VideoWriter(str(out/'pet_masa_1fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1,(image.shape[1],image.shape[0]))
                if not video.isOpened():raise RuntimeError('Could not create overlay video')
            video.write(image)
            total_estimate=row['detector_s']+time.perf_counter()-begin
            tracker_times.append(tracker_s);estimated.append(total_estimate)
            clean=lambda t:{k:v for k,v in t.items() if k not in ('appearance','mask')}
            log.write(json.dumps(dict(frame=row['frame'],source=row['source'],detections=row['detections'],
                tracks=[clean(t) for t in obs],new_objects=new,tracker_s=tracker_s,estimated_pipeline_s=total_estimate))+'\n')
            log.flush()
            print(f"{row['frame']+1}/{len(frames)} raw={len(row['detections'])} tracks={len(obs)} new={len(new)} tracker={tracker_s:.3f}s",flush=True)
    if video is not None:video.release()
    assert len({e['object_id'] for e in events})==len(events)==len(registry.objects)
    (out/'creation_events.json').write_text(json.dumps(events,indent=2))
    metadata=json.loads((out/'detector.json').read_text())
    summary=dict(frames=len(frames),ways=len(metadata['detector']['shots_per_class']),
        supports=sum(metadata['detector']['shots_per_class'].values()),detections=raw_count,
        tracked_observations=observations_count,unique_objects=len(registry.objects),tracker=tracker.metadata(),
        tracker_mean_s=float(np.mean(tracker_times)),tracker_warm_mean_s=float(np.mean(tracker_times[1:])),
        tracker_p95_s=float(np.percentile(tracker_times,95)),estimated_pipeline_mean_s=float(np.mean(estimated)),
        estimated_pipeline_p95_s=float(np.percentile(estimated,95)),estimated_frames_over_1s=sum(t>1 for t in estimated),
        api_enabled=False,accuracy_evaluated=False,
        limitations=['Supports come from this site; this run has no held-out identity/accuracy labels',
                     'PET and MASA use separate runtimes; pipeline latency is a sum of measured components, not a live end-to-end benchmark',
                     'Physical duplicate/false-merge accuracy requires cross-frame identity ground truth'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
