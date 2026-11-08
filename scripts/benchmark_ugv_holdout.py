"""Score a detector on the frozen saved ROSBAG-PUI annotations, without tracking."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np
import torch
from pui_comparison_detectors import build_detector, digest
from triffid_uav_perception.evaluation import mask_counts, rates, box_counts

ROOT = Path(__file__).resolve().parents[1]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--split', type=Path, default=ROOT/'data/ROSBAG-PUI/evaluation_v2/split.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    split = json.loads(args.split.read_text())
    if digest(split['bank']) != split['bank_sha256']:
        raise ValueError('Frozen support bank changed')
    args.output.mkdir(parents=True,exist_ok=False)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    start = time.perf_counter()
    detector = build_detector(args.model, split['bank'], split['classes'])
    torch.cuda.synchronize(); preparation_s = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    cat = json.loads((ROOT/'data/ROSBAG-PUI/snapshot_20261107/metadata/categories.json').read_text())
    names = {c['id']:c['name'] for c in cat}
    selected = {c['id'] for c in cat if c['name'] in split['classes']}
    totals = defaultdict(lambda:dict(tp=0,fp=0,fn=0))
    box_totals = defaultdict(lambda:dict(tp=0,fp=0,fn=0))
    records=[]; coco_predictions=[]; times=[]; semantic_classes=set(); box_classes=set()
    has_masks = None
    for q in split['queries']:
        if digest(q['image']) != q['image_sha256']:
            raise ValueError('Query image changed')
        bgr = cv2.imread(q['image']); h,w=bgr.shape[:2]
        truth = {int(cid):cv2.imread(path,0)>0 for cid,path in q['masks'].items()}
        valid=np.zeros((h,w),bool)
        for mask in truth.values():valid |= mask
        torch.cuda.synchronize(); start=time.perf_counter()
        detections=detector.detect(bgr)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start; times.append(elapsed)
        has_masks = any(d.get('mask') is not None for d in detections) if has_masks is None else has_masks or any(d.get('mask') is not None for d in detections)
        rows=[]; overlay=bgr.copy()
        for cid in sorted(selected):
            ds=[d for d in detections if int(d['class_id'])==cid]
            gtboxes=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in q['boxes'] if a['category_id']==cid]
            counts=box_counts([d['bbox'] for d in ds],gtboxes)
            if gtboxes:box_classes.add(cid)
            for key,value in counts.items():box_totals[cid][key]+=value
            row=dict(class_id=cid,class_name=names[cid],boxes=counts)
            if cid in truth:semantic_classes.add(cid)
            pred=np.zeros((h,w),bool)
            for d in ds:
                if d.get('mask') is not None:
                    if d['mask'].shape != (h,w):raise ValueError('Prediction mask dimensions differ')
                    pred |= np.asarray(d['mask'],bool)
            if any(d.get('mask') is not None for d in detections):
                counts=mask_counts(pred,truth.get(cid,np.zeros((h,w),bool)),valid)
                row['semantic']=counts
                for key,value in counts.items():totals[cid][key]+=value
            rows.append(row)
        clean=[]
        for d in detections:
            x,y,x1,y1=map(float,d['bbox'])
            coco_predictions.append(dict(image_id=q['task_id'],category_id=int(d['class_id']),bbox=[x,y,x1-x,y1-y],score=float(d['confidence'])))
            clean.append({k:v for k,v in d.items() if k not in ('mask','appearance')})
            cv2.rectangle(overlay,(int(x),int(y)),(int(x1),int(y1)),(0,255,255),2)
            cv2.putText(overlay,d['class_name'],(max(0,int(x)),max(15,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
        cv2.imwrite(str(args.output/f'task_{q["task_id"]:06d}.jpg'),overlay)
        records.append(dict(task_id=q['task_id'],annotation_id=q['annotation_id'],inference_s=elapsed,
            valid_pixel_fraction=float(valid.mean()),detections=clean,by_class=rows))
        print(f'{args.model} task={q["task_id"]} detections={len(detections)} {elapsed:.3f}s',flush=True)
    boxes={k:sum(v[k] for v in box_totals.values()) for k in ('tp','fp','fn')}
    by_class=[dict(class_id=cid,class_name=names[cid],boxes=box_totals[cid],box_rates=rates(box_totals[cid]),
        labeled_mask_counts=totals[cid] if has_masks else None,
        labeled_mask_rates=rates(totals[cid]) if has_masks else None) for cid in sorted(selected)]
    # Class-aware COCO AP only over categories with saved query boxes; other
    # unsupported classes are reported separately and never silently scored zero.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(str(ROOT/'data/ROSBAG-PUI/snapshot_20261107/annotations/object_detection.json'))
    ap=None
    if coco_predictions:
        dt=gt.loadRes(coco_predictions)
        evaluation=COCOeval(gt,dt,'bbox');evaluation.params.imgIds=[q['task_id'] for q in split['queries']]
        evaluation.params.catIds=sorted(box_classes)
        evaluation.evaluate();evaluation.accumulate();evaluation.summarize()
        ap=dict(AP_50_95=float(evaluation.stats[0]),AP50=float(evaluation.stats[1]),AP75=float(evaluation.stats[2]))
    else:
        ap=dict(AP_50_95=0.,AP50=0.,AP75=0.)
    result=dict(schema='triffid.ugv.holdout.metrics.v1',model=args.model,status='complete',
        split_sha256=digest(args.split),split=split,preparation_s=preparation_s,
        mean_inference_s=float(np.mean(times)),second_frame_inference_s=times[-1],
        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        box_counts=boxes,box_rates=rates(boxes),coco_bbox=ap,
        labeled_region_miou=float(np.mean([rates(totals[c])['iou'] for c in semantic_classes])) if has_masks and semantic_classes else None,
        annotated_but_unsupported=sorted({names[a['category_id']] for q in split['queries'] for a in q['boxes'] if a['category_id'] not in selected}),
        mask_capability='native instance/semantic masks' if has_masks else 'no masks returned; no boxes rasterized',
        by_class=by_class,frames=records,backend=detector.metadata(),
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
        api_enabled=False,tracking_evaluated=False,
        protocol=dict(boxes='COCO bbox AP plus class-aware IoU .5 maximum-cardinality P/R/F1, at native fixed acceptance thresholds',
            masks='Class union IoU within union of valid saved brush masks; absent-class predictions count as false positives there',
            limitations=split['limitations']+['AP is threshold-truncated by each adapter; thresholds are recorded, not tuned on these queries.']))
    dump(args.output/'metrics.json',result)
    dump(args.output/'coco_predictions.json',coco_predictions)
    print(json.dumps({k:result[k] for k in ('model','labeled_region_miou','box_rates','coco_bbox','mean_inference_s')},indent=2),flush=True)

if __name__=='__main__':main()
