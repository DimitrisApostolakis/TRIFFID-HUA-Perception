"""Extract complete YOLOE candidates + frozen MASA embeddings for controlled CPU replay."""
import argparse
import gzip
import json
from pathlib import Path
import time

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
WORK=ROOT/'ugv_models/comparison_work/class_correction/YOLOE'


def main():
    import torch
    import cv2
    from triffid_ugv_perception.tracking.factory import TrackingPipeline
    from pui_comparison_detectors import digest
    WORK.mkdir(parents=True,exist_ok=True)
    tracker=TrackingPipeline('masa',person_roles=True)
    tracker.engine.capture_candidates=True
    rows=[json.loads(l) for l in (WORK/'detections.jsonl').open()]
    dataset=ROOT/'data/PUI/rgb_frames'
    with gzip.open(WORK/'association_cache.jsonl.gz','wt') as output:
        for row in rows:
            image=cv2.imread(str(dataset/row['source']['file']))
            torch.cuda.synchronize();start=time.perf_counter()
            tracks=tracker.update(row['candidates'],image,row['source']['bag_timestamp_ns'])
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            candidates=[{k:v for k,v in d.items() if k not in ('appearance','mask')} for d in tracker.engine.last_candidate_observations]
            clean=lambda d:{k:v for k,v in d.items() if k not in ('appearance','mask')}
            output.write(json.dumps(dict(frame=row['frame'],source=row['source'],detections=row['detections'],
                candidates=candidates,raw_embeddings=tracker.engine.last_raw_embeddings,
                baseline_tracks=[clean(t) for t in tracks],detector_s=row['detector_s'],baseline_tracker_s=elapsed))+'\n')
            print(f"MASA feature cache {row['frame']+1}/{len(rows)}",flush=True)
    metadata=dict(tracker=tracker.metadata(),detector=json.loads((WORK/'detector.json').read_text()),
        bank_sha256=digest(ROOT/'ugv_supports/PUI/bank.json'),dataset_index_sha256=digest(dataset/'frames.jsonl'),
        frames=len(rows),api_enabled=False,cache_schema='raw_MASA_embeddings_v1')
    (WORK/'association_cache.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
