"""Evaluate saved predictions against independently annotated COCO query boxes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from triffid_ugv_perception.io import write_json


def evaluate(run,ground_truth,threshold=.35):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    root=Path(run);truth=json.loads(Path(ground_truth).read_text())
    if truth.get('annotation_status')=='unlabelled':raise ValueError('Review template is unlabelled; accuracy cannot be measured')
    rows=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
    names={Path(row['source']['file']).name:row for row in rows}
    def query_name(image):
        name=Path(image['file_name']).name
        if name in names:return name
        # Label Studio can prefix the uploaded basename with an upload hash.
        if '-' in name and name.split('-',1)[1] in names:return name.split('-',1)[1]
        return name
    images=[image for image in truth['images'] if query_name(image) in names]
    if not images:raise ValueError('Ground truth contains no matching query frames')
    selected_names=set(json.loads((root/'class_availability.json').read_text())['selected'])
    category_ids=[category['id'] for category in truth['categories'] if category['name'] in selected_names]
    if not category_ids:raise ValueError('No selected class names match ground truth')
    by_name={category['name']:category['id'] for category in truth['categories']}
    predictions=[]
    for image in images:
        row=names[query_name(image)]
        for prediction in row.get('candidate_detections',row['detections']):
            if prediction['class_name'] not in by_name:continue
            x1,y1,x2,y2=prediction['bbox']
            predictions.append(dict(image_id=image['id'],category_id=by_name[prediction['class_name']],
                bbox=[x1,y1,x2-x1,y2-y1],score=prediction['confidence']))
    truth.setdefault('info',{})
    gt=COCO();gt.dataset=truth;gt.createIndex()
    if predictions:det=gt.loadRes(predictions)
    else:
        det=COCO();det.dataset=dict(images=truth['images'],categories=truth['categories'],annotations=[]);det.createIndex()
    evaluator=COCOeval(gt,det,'bbox');evaluator.params.imgIds=[image['id'] for image in images];evaluator.params.catIds=category_ids
    evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    counts=defaultdict(lambda:dict(tp=0,fp=0,fn=0))
    annotations=defaultdict(list)
    for annotation in truth['annotations']:
        if annotation['category_id'] in category_ids and not annotation.get('iscrowd',0):
            annotations[(annotation['image_id'],annotation['category_id'])].append(annotation['bbox'])
    for image in images:
        for category_id in category_ids:
            boxes=annotations[(image['id'],category_id)];matched=set()
            candidates=sorted([p for p in predictions if p['image_id']==image['id'] and p['category_id']==category_id and p['score']>=threshold],key=lambda p:p['score'],reverse=True)
            for prediction in candidates:
                x,y,w,h=prediction['bbox'];best=-1;overlap=0.
                for index,(a,b,c,d) in enumerate(boxes):
                    if index in matched:continue
                    intersection=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
                    value=intersection/max(w*h+c*d-intersection,1e-8)
                    if value>overlap:best,overlap=index,value
                if overlap>=.5:matched.add(best);counts[category_id]['tp']+=1
                else:counts[category_id]['fp']+=1
            counts[category_id]['fn']+=len(boxes)-len(matched)
    per_class={}
    for category in truth['categories']:
        if category['id'] not in category_ids:continue
        values=counts[category['id']];tp,fp,fn=values['tp'],values['fp'],values['fn']
        per_class[category['name']]=dict(values,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
    result=dict(evaluated_images=len(images),bbox_AP=float(evaluator.stats[0]),bbox_AP50=float(evaluator.stats[1]),
        confidence_threshold=threshold,precision_recall_iou=.5,per_class=per_class,
        ground_truth=str(Path(ground_truth).resolve()),limitations='Requires exhaustive independent query annotations; crowd/ignore semantics follow COCO for AP only')
    write_json(root/'accuracy.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run');parser.add_argument('ground_truth')
    args=parser.parse_args();print(json.dumps(evaluate(args.run,args.ground_truth),indent=2))
