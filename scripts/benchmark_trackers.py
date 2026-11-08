"""Replay identical PET detections/features through tracker and role ablations. No API."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from triffid_ugv_perception.tracking.factory import TrackingPipeline, PERSON_ROLES
from triffid_ugv_perception.tracking.vendor.udc_core import iou_xyxy
from triffid_ugv_perception.discoveries import ObjectRegistry
from triffid_ugv_perception.io import write_json


def describe(values):
    return dict(mean_s=float(np.mean(values)),p95_s=float(np.percentile(values,95)),max_s=float(max(values)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).absolute().parents[1]
    p.add_argument('--cache',type=Path,default=root/'archive/samples_legacy_20261106/udc_tracking_20261105/detector_cache')
    p.add_argument('--dataset',type=Path,default=root/'data/PUI/rgb_frames')
    p.add_argument('--output',type=Path,default=root/'archive/samples_legacy_20261106/udc_tracking_20261105/comparison')
    p.add_argument('--udc-proof-threshold',type=float,default=.35)
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--udc-only',action='store_true')
    selection.add_argument('--botsort-only',action='store_true')
    args=p.parse_args()
    if not .35<=args.udc_proof_threshold<=1:p.error('UDC proof threshold must be between 0.35 and 1')
    cache_file=args.cache/'tracking_cache.jsonl'
    frames=[json.loads(line) for line in cache_file.read_text().splitlines()]
    source_summary=json.loads((args.cache/'summary.json').read_text())
    detector_frames=[json.loads(line) for line in (args.cache/'frames.jsonl').read_text().splitlines()]
    assert len(frames)==len(detector_frames)==source_summary['frames']
    assert [f['source'] for f in frames]==[f['source'] for f in detector_frames]
    streams={(f['source']['bag'],f['source']['topic']) for f in frames}
    if len(streams)!=1:raise ValueError('Use separate tracking runs for different camera streams')
    cache_hash=hashlib.sha256(cache_file.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for name,group_roles,floor in [('botsort',False,.35),('botsort',True,.35),('botsort',True,.1),('udc',False,.35),('udc',True,.35),('udc',True,.1)]:
        if args.udc_only and name!='udc':continue
        if args.botsort_only and name!='botsort':continue
        label=name+('_person_roles' if group_roles else '_class_gated')+('_low_recovery' if floor<.35 else '')
        out=args.output/label;out.mkdir();(out/'overlays').mkdir()
        tracker=TrackingPipeline(name,person_roles=group_roles,udc_proof_threshold=args.udc_proof_threshold)
        registry=ObjectRegistry(out/'registry.json',context=dict(cache_sha256=cache_hash,tracker=label,udc_proof_threshold=args.udc_proof_threshold if name=='udc' else None))
        timing=[];registry_timing=[];total_timing=[];observations_count=0;people_count=0
        duplicate_pairs=0;person_pairs=0;low_recovered=0;events=[];role_switches=0
        last_role={};person_ids=set();current_classes=Counter();revivals=0
        with (out/'frames.jsonl').open('w') as log:
            for f,df in zip(frames,detector_frames):
                started=time.perf_counter()
                image=cv2.imread(str(args.dataset/f['source']['file']))
                if image is None:raise ValueError(f['source']['file'])
                begin=time.perf_counter()
                inputs=[d for d in f['detections'] if d['confidence']>=floor]
                tracked=tracker.update(inputs,image,f['source']['bag_timestamp_ns'])
                timing.append(time.perf_counter()-begin)
                begin=time.perf_counter()
                obs,new=registry.update(tracked,f['source']['bag_timestamp_ns'])
                registry.write_creation_events(out/'new_objects')
                registry_timing.append(time.perf_counter()-begin)
                assert len({t['track_id'] for t in obs})==len(obs)
                assigned=[t['object_id'] for t in obs if t['object_id']]
                assert len(set(assigned))==len(assigned)
                events.extend(new)
                people=[t for t in obs if t['class_name'] in PERSON_ROLES]
                observations_count+=len(obs);people_count+=len(people)
                low_recovered+=sum(t['confidence']<.35 for t in obs)
                revivals+=len(getattr(tracker.engine,'revivals',[]))
                for i,a in enumerate(obs):
                    for b in obs[i+1:]:
                        same_family=(a['class_name'] in PERSON_ROLES and b['class_name'] in PERSON_ROLES) or a['class_id']==b['class_id']
                        if same_family and iou_xyxy(a['bbox'],b['bbox'])>=.7:
                            duplicate_pairs+=1
                            if a['class_name'] in PERSON_ROLES:person_pairs+=1
                for t in obs:
                    current_classes[t['class_name']]+=1
                    if t['class_name'] not in PERSON_ROLES or not t['object_id']:continue
                    oid=t['object_id'];person_ids.add(oid)
                    if oid in last_role and last_role[oid]!=t['class_name']:role_switches+=1
                    last_role[oid]=t['class_name']
                overlay=image.copy()
                for d in f['detections']:
                    if d['confidence']<.35:continue
                    x1,y1,x2,y2=map(int,d['bbox'])
                    cv2.rectangle(overlay,(x1,y1),(x2,y2),(110,110,110),1)
                for t in obs:
                    x1,y1,x2,y2=map(int,t['bbox']);oid=(t['object_id'] or 'confirming')[:8]
                    color=(0,180,255) if t.get('role_ambiguous') else (0,220,0)
                    cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
                    text=f"{t['class_name']} t{t['track_id']} {oid}"+(' role?' if t.get('role_ambiguous') else '')
                    cv2.putText(overlay,text,(max(x1,0),max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                assert cv2.imwrite(str(out/'overlays'/f"{f['frame']:04d}.jpg"),overlay)
                total_timing.append(time.perf_counter()-started)
                clean=lambda t:{k:v for k,v in t.items() if k!='appearance'}
                log.write(json.dumps(dict(frame=f['frame'],source=f['source'],tracks=[clean(t) for t in obs],
                    new_objects=new,tracking_s=timing[-1],registry_s=registry_timing[-1]))+'\n')
        ids=[event['object_id'] for event in events]
        assert len(set(ids))==len(ids)==len(list((out/'new_objects').glob('*.json')))==len(registry.objects)
        summary=dict(name=label,frames=len(frames),input_floor=floor,
            input_detections=sum(sum(d['confidence']>=floor for d in f['detections']) for f in frames),
            input_cache_sha256=cache_hash,tracker=tracker.metadata(),api_enabled=False,
            tracker_updates=tracker.calls,tracker_timing=describe(timing),registry_timing=describe(registry_timing),
            replay_with_read_and_overlay_timing=describe(total_timing),
            estimated_complete_mean_s=float(np.mean(total_timing))+source_summary['timing']['mean_inference_s'],
            observations=observations_count,person_observations=people_count,registry_objects=len(registry.objects),
            person_registry_ids=len(person_ids),overlapping_hypothesis_pairs=duplicate_pairs,
            overlapping_person_pairs=person_pairs,low_confidence_observations=low_recovered,
            appearance_revivals=revivals,role_changes_per_uuid=role_switches,class_observations=dict(current_classes),
            audit_passed=True,
            limitations=['Overlap pairs are a duplicate proxy, not ground-truth duplicate counts',
                'Fewer IDs can reflect missed objects or false merges; no identity-accuracy conclusion',
                'Estimated complete mean adds measured detector inference to CPU replay; not a live end-to-end measurement',
                'UDC proof gating differs from BoT-SORT; output counts are not comparable accuracy scores'])
        write_json(out/'summary.json',summary);results.append(summary)
        print(json.dumps({k:summary[k] for k in ['name','frames','observations','person_registry_ids','overlapping_person_pairs','tracker_timing']}),flush=True)
    write_json(args.output/'results.json',dict(source_detector=source_summary['detector'],input_cache_sha256=cache_hash,runs=results,accuracy_status='Query identity/object ground truth unavailable'))
    options=''.join(f'<option value="{f["frame"]}">{f["frame"]:04d}</option>' for f in frames)
    tiles=''.join(f'<figure><figcaption>{r["name"]}</figcaption><a target="_blank"><img data-run="{r["name"]}"></a></figure>' for r in results)
    html='''<!doctype html><meta charset="utf-8"><title>Tracker comparison</title><style>body{font-family:system-ui;margin:24px;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}select{font-size:18px}</style><h1>Same PET-DINO detections: tracker / person-role comparison</h1><p>Grey: detector hypotheses at 0.35+. Green: observed tracks. Orange: ambiguous role. UUIDs are local predictions. Views without low_recovery use detections at 0.35+. The low_recovery view additionally uses 0.1+ recovery candidates. This is not a labelled accuracy evaluation.</p><label>Frame <select id="frame">OPTIONS</select></label><main>TILES</main><script>function update(){const f=document.getElementById('frame').value.padStart(4,'0');document.querySelectorAll('img').forEach(i=>{i.src=i.dataset.run+'/overlays/'+f+'.jpg';i.parentElement.href=i.src;});}document.getElementById('frame').onchange=update;update();</script>'''
    (args.output/'comparison.html').write_text(html.replace('OPTIONS',options).replace('TILES',tiles))


if __name__=='__main__':main()
