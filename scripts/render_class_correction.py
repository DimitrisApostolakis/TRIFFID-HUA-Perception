"""Render a validated cached MASA replay; no model inference, API or HTML."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
from pui_comparison_views import VideoSet,geojson_collection,competing_labels,summarize_packets,VIEW_SCHEMA,install_replacement

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
WORK=ROOT/'ugv_models/comparison_work/class_correction/YOLOE'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',required=True)
    parser.add_argument('--run',default='class_correction_v1')
    args=parser.parse_args()
    replay=WORK/args.replay
    packets=[json.loads(l) for l in (replay/'corrected.jsonl').open()]
    baseline=[json.loads(l) for l in (replay/'baseline.jsonl').open()]
    cached=[json.loads(l) for l in (WORK/'detections.jsonl').open()]
    comparison=json.loads((replay/'summary.json').read_text())
    assert comparison['baseline']['baseline_id_parity_frames']==len(packets)
    original=ROOT/'samples/YOLOE/original/ugv_pui_20261106'
    audit=json.loads((original/'audit.json').read_text())
    provenance=json.loads((WORK/'association_cache.json').read_text())
    assert provenance['bank_sha256']==audit['bank_sha256']
    assert provenance['dataset_index_sha256']==audit['dataset_index_sha256']
    destination=ROOT/'samples/YOLOE/original'/args.run
    output=replay/'rendered'
    output.mkdir(parents=True,exist_ok=False)
    videos=VideoSet(output,'YOLOE',len(packets));timings=[]
    try:
        for packet,old,row in zip(packets,baseline,cached):
            assert packet['source']==old['source']==row['source']
            packet['class_correction']=True
            packet['competing_labels']=competing_labels(packet['detections'],packet['tracks'])
            image=cv2.imread(str(Path(audit['input_dataset'])/packet['source']['file']))
            if image is None:raise ValueError('Missing source frame')
            mask_path=WORK/'mask_overlays'/f"{packet['frame']:04d}.jpg"
            masks=cv2.imread(str(mask_path)) if mask_path.exists() else None
            videos.write(image,packet,masks)
            timings.append(dict(frame=packet['frame'],timestamp_ns=packet['source']['bag_timestamp_ns'],
                detector_s=row['detector_s'],cached_cpu_association_registry_s=packet['association_registry_s'],
                baseline_cpu_association_registry_s=old['association_registry_s'],
                detections=len(packet['detections']),tracks=len(packet['tracks']),
                new_objects=len(packet['new_object_ids']),updates=len(packet['updated_object_ids']),
                api_ready_creations=0,geojson_exports=len(packet['new_object_ids'])))
    finally:videos.close()
    collection=geojson_collection(packets,'YOLOE')
    audit.update(status='complete',artifact_schema=VIEW_SCHEMA,class_correction=True,frames=len(packets),
        new_objects=len(collection['features']),geojson_features=len(collection['features']),
        detections=sum(t['detections'] for t in timings),track_observations=sum(t['tracks'] for t in timings),
        local_update_events=sum(t['updates'] for t in timings),api_ready_creations=0,api_sent=0,
        export_video_policy='NEW creates a local UUID once; UPDATED revises that same local feature. API OFF.',
        class_competition=summarize_packets(packets),class_correction_comparison=comparison,
        timing_note='Detector scores/times and MASA embeddings were computed on GPU once and frozen. Matching and registry were replayed on CPU for baseline and corrected policies; these CPU times exclude image features and rendering and are not live end-to-end latency.',
        replay_source=str(replay),association_cache=str(WORK/'association_cache.jsonl.gz'))
    audit['class_evidence_policy']=dict(window_seconds=5,reliable_threshold=.35,
        minimum_support=2,recent_reliable_observations=3,normalized_vote_lead=.15,
        decay_per_second=.8,confidence_update_step=.10)
    audit['tracker']['class_correction']=True
    audit['tracker']['engine']['class_compatibility']='soft_with_guarded_class_changes'
    audit['association_runtime']=dict(python=audit['tracker_runtime']['python'],
        torch=audit['tracker_runtime']['torch'],device='cpu',input='frozen GPU MASA embeddings')
    audit['policy_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'src/triffid_ugv_perception/triffid_ugv_perception/tracking/class_correction.py',
                  ROOT/'ugv_models/masa/masa/models/tracker/masa_tao_tracker.py']}
    # Old whole-stage timings do not describe this cached replay.
    audit.pop('timing_summary',None);audit.pop('tracker_initialization_s',None)
    for name,value in [('audit.json',audit),('detections.geojson',collection)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (output/'observations.jsonl').write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in packets))
    with (output/'timings.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
    install_replacement(output,destination,len(packets))
    print(destination,flush=True)


if __name__=='__main__':main()
