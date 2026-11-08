"""Controlled native-MASA CPU replay using frozen, complete GPU-extracted embeddings."""
import argparse
from collections import defaultdict,deque
from dataclasses import asdict
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from mmengine.structures import InstanceData

from triffid_ugv_perception.discovery import DiscoveryService,DiscoverySettings
from triffid_ugv_perception.tracking.class_correction import ClassChangeAssociation,AssociationSettings

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
WORK=ROOT/'ugv_models/comparison_work/class_correction/YOLOE'


def native_class():
    path=ROOT/'ugv_models/masa/masa/models/tracker/masa_tao_tracker.py'
    spec=importlib.util.spec_from_file_location('masa_class_correction_replay',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.MasaTaoTracker


def key(box,score,label):
    return int(label),float(np.float32(score)),tuple(np.asarray(box,dtype=np.float32).tolist())


def replay(rows,native,corrected,path,settings):
    tracker=native(init_score_thr=.35,obj_score_thr=.1,match_score_thr=.5,memo_tracklet_frames=60,
                   memo_momentum=.8,with_cats=True,max_distance=-1,fps=1)
    policy=ClassChangeAssociation(settings) if corrected else None
    if policy:tracker.class_change_policy=policy
    service=DiscoveryService(path,DiscoverySettings(namespace='PUI/YOLOE/class_check',
        embedding_space='MASA-R50/class_correction_v1' if corrected else 'MASA-R50/baseline',class_correction=corrected))
    first=rows[0]['source']['bag_timestamp_ns'];packets=[];durations=[];parity=[]
    try:
        for row in rows:
            begin=time.perf_counter();timestamp=row['source']['bag_timestamp_ns']
            tick=(timestamp-first+10**9-1)//10**9
            tracker.tracks={k:v for k,v in tracker.tracks.items() if tick-v['last_frame']<60}
            candidates=row['candidates'];raw_features=np.asarray(row['raw_embeddings'] or [],dtype=np.float32)
            tracks=[]
            if candidates:
                boxes=torch.tensor([d['bbox'] for d in candidates],dtype=torch.float32)
                scores=torch.tensor([d['confidence'] for d in candidates],dtype=torch.float32)
                labels=torch.tensor([10000 if d['class_id']==-1 else d['class_id'] for d in candidates])
                features=torch.from_numpy(raw_features)
                sample=SimpleNamespace(metainfo=dict(frame_id=int(tick),scale_factor=(1.,1.)),
                    pred_instances=InstanceData(bboxes=boxes,labels=labels,scores=scores))
                model=SimpleNamespace(track_head=SimpleNamespace(predict=lambda *a,**kw:features))
                if policy:policy.set_sources(candidates)
                with torch.inference_mode():output=tracker.track(model,None,[],sample,rescale=False)
                mapping=defaultdict(deque)
                for i,d in enumerate(candidates):mapping[key(d['bbox'],d['confidence'],10000 if d['class_id']==-1 else d['class_id'])].append(i)
                for box,score,label,tid in zip(output.bboxes.tolist(),output.scores.tolist(),output.labels.tolist(),output.instances_id.tolist()):
                    index=mapping[key(box,score,label)].popleft();source=candidates[index]
                    vector=raw_features[index];vector=vector/max(float(np.linalg.norm(vector)),1e-12)
                    item=dict(source,bbox=box,confidence=score,track_id=tid,appearance=vector.tolist(),recovered=score<.35)
                    if policy:
                        item['class_evidence']=policy.evidence.get(key(box,score,label),[])
                        evidence=item['class_evidence'] or item.get('role_evidence')
                        if evidence:
                            best=max(evidence,key=lambda r:r['confidence']);item.update(class_id=best['class_id'],class_name=best['class_name'])
                        item['identity_class_id']=-2
                    tracks.append(item)
            if not corrected:
                original={(tuple(np.asarray(t['bbox'],np.float32)),t['track_id']) for t in row['baseline_tracks']}
                current={(tuple(np.asarray(t['bbox'],np.float32)),t['track_id']) for t in tracks}
                parity.append(original==current)
            result,_=service.process(tracks,[None]*len(tracks),timestamp)
            clean=lambda t:{k:v for k,v in t.items() if k not in ('appearance','mask')}
            packet=dict(frame=row['frame'],source=row['source'],detections=row['detections'],
                tracks=[clean(t) for t in result.observations],new_object_ids=result.created_ids,
                updated_object_ids=result.updated_ids or [],geojson_export_ids=result.created_ids,
                api_ready_event_ids=[],api_queue_pending=0,api_sent=0,api_enabled=False,
                world_coordinates_available=False,association=policy.last_diagnostics if policy else {})
            packet['class_correction']=corrected
            packet['association_registry_s']=time.perf_counter()-begin
            packets.append(packet);durations.append(packet['association_registry_s'])
        service.registry.flush()
        records=service.store.records()
        summary=dict(frames=len(rows),objects=len(records),new_events=sum(len(p['new_object_ids']) for p in packets),
            class_updates=sum(bool(t.get('class_update') and t['class_update']['kind']=='class') for p in packets for t in p['tracks']),
            update_events=sum(len(p['updated_object_ids']) for p in packets),
            cpu_association_registry_mean_ms=float(np.mean(durations))*1000,
            cpu_association_registry_p95_ms=float(np.percentile(durations,95))*1000,
            baseline_id_parity_frames=sum(parity) if parity else None,api_enabled=False)
    finally:service.close()
    return packets,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=15)
    parser.add_argument('--name',default='bag_check')
    parser.add_argument('--similarity',type=float,default=.85)
    args=parser.parse_args()
    torch.set_num_threads(4)
    with gzip.open(WORK/'association_cache.jsonl.gz','rt') as handle:rows=[json.loads(l) for l in handle]
    if args.limit:rows=rows[:args.limit]
    run=WORK/args.name
    if run.exists():raise ValueError('Replay name exists; use a fresh --name')
    run.mkdir();native=native_class();settings=AssociationSettings(cross_similarity=args.similarity)
    output={}
    for name,corrected in [('baseline',False),('corrected',True)]:
        packets,summary=replay(rows,native,corrected,run/(name+'.sqlite3'),settings)
        (run/(name+'.jsonl')).write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in packets))
        output[name]=summary
    output['association_settings']=asdict(settings)
    (run/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)
    for p in packets:
        if p['frame']>17:break
        print('FRAME',p['frame']+1,[(t['track_id'],t['class_name'],round(t['confidence'],2),str(t.get('object_id'))[:6],t.get('class_update')) for t in p['tracks'] if t['class_name'] in ('bag','chainsaw')],flush=True)


if __name__=='__main__':main()
