"""Fixed-support, two-frame diagnostic for FoRIS/FSS-SAM3/SANSA/FROST.

Class episodes, original-resolution masks, IoU and box matching are shared;
native model preprocessing/decision rules are preserved. No labels tune models.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import random
from evaluation_provenance import capture

from triffid_uav_perception.evaluation import mask_counts, rates, box_counts
from triffid_uav_perception.export import write_json
from triffid_uav_perception.media import read_rgb
from triffid_uav_perception.ontology import CLASSES
from triffid_uav_perception.supports import SupportBank
from triffid_uav_perception.visualize import class_color

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT/'data/UAV-PUI/holdout_1_3/split.json'


def build_backend(name):
    if name in ('rsrm','fssdino'):
        from triffid_uav_perception.models.dino_variants import DinoVariantBackend
        return DinoVariantBackend(name)
    if name == 'insid3':
        from triffid_uav_perception.models.dinov3_fewshot import Dinov3FewShotBackend
        return Dinov3FewShotBackend(name, '/mnt/d/Models/ThirdParty/INSID3',
            '/mnt/d/Models/ThirdParty/dinov3',
            '/mnt/d/Models/Checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    if name == 'sansa':
        from triffid_uav_perception.models.sansa import SansaBackend
        return SansaBackend('/mnt/d/Models/ThirdParty/SANSA',
                            '/mnt/d/Models/Checkpoints/SANSA/adapter_generalist.pth',
                            '/mnt/d/Models/Checkpoints/SANSA/sam2_hiera_large.pt', image_size=640)
    if name.startswith('fss-sam3'):
        from triffid_uav_perception.models.fss_sam3 import FssSam3Backend
        return FssSam3Backend('/mnt/d/Models/ThirdParty/FSS-SAM3',
                             '/mnt/d/Models/Checkpoints/FSS-SAM3/sam3.pt',
                             '/mnt/d/Models/Checkpoints/FSS-SAM3/bpe_simple_vocab_16e6.txt.gz',
                             use_text=name.endswith('-text'))
    from triffid_uav_perception.models.dinov3_fewshot import Dinov3FewShotBackend
    repo = ROOT/'uav_models/FoRIS' if name == 'foris' else Path('/mnt/d/Models/ThirdParty/FROST')
    return Dinov3FewShotBackend(name, repo, '/mnt/d/Models/ThirdParty/dinov3',
                              '/mnt/d/Models/Checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')


def components(mask, minimum=16):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [[int(x), int(y), int(x+w), int(y+h)] for x,y,w,h,area in stats[1:] if area >= minimum]


def draw(rgb, masks, classes):
    canvas = rgb.copy()
    names = {c.id: c.name for c in classes}
    for cid, mask in sorted(masks.items()):
        if cid not in names:
            continue
        color = np.asarray(class_color(names[cid]), dtype=np.float32)
        canvas[mask] = (.55*canvas[mask] + .45*color).astype(np.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['foris', 'fss-sam3', 'fss-sam3-text', 'sansa', 'frost', 'insid3', 'rsrm', 'fssdino'], required=True)
    parser.add_argument('--output', type=Path, help='Fresh output directory; existing artifacts are preserved')
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text())
    model_name = dict(foris='FoRIS', frost='FROST', sansa='SANSA', insid3='INSID3', rsrm='RSRM-ViT-L', fssdino='FSSDINO-ViT-L',
                      **{'gf-sam':'GF-SAM'},
                      **{'fss-sam3': 'FSS-SAM3', 'fss-sam3-text': 'FSS-SAM3-text'})[args.backend]
    out = args.output or ROOT/'samples'/model_name/'UAV/holdout_1_3_1shot'
    if out.exists():
        raise ValueError(f'Preserve completed or partial artifacts: {out}')
    out.mkdir(parents=True)
    bank_path = Path(split['bank'])
    assert hashlib.sha256(bank_path.read_bytes()).hexdigest() == split['bank_sha256'], 'Support split changed'
    classes = tuple(c for c in CLASSES if c.id in split['evaluated_class_ids'])
    selection = SupportBank(bank_path).select(classes, shots=1)
    for item in classes:
        for support in selection.for_class(item):
            assert hashlib.sha256(support.image.read_bytes()).hexdigest() not in split['held_out_image_sha256']
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)
    backend = build_backend(args.backend)
    print(f'{model_name}: preparing {len(classes)} classes, one support each, holdout tasks 1/3', flush=True)
    started = time.perf_counter()
    backend.prepare(classes, selection)
    torch.cuda.synchronize()
    preparation_s = time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats()
    totals = defaultdict(lambda: {kind: dict(tp=0,fp=0,fn=0) for kind in ('labeled', 'full', 'boxes')})
    rows, frames, writer = [], [], None
    try:
        for query in split['queries']:
            rgb = read_rgb(query['image'])
            h,w = rgb.shape[:2]
            truth = {int(cid): cv2.imread(path, cv2.IMREAD_GRAYSCALE)>0 for cid,path in query['masks'].items()}
            valid = np.zeros((h,w),bool)
            for mask in truth.values():
                valid |= mask
            frame_rows, predictions = [], {}
            iterator = iter(backend.predict(rgb, classes))
            for item in classes:
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = next(iterator)
                torch.cuda.synchronize()
                inference_s = time.perf_counter()-start
                assert prediction.class_id == item.id
                pred = np.asarray(prediction.mask, dtype=bool)
                gt = truth.get(item.id, np.zeros((h,w),bool))
                labeled = mask_counts(pred, gt, valid)
                full = mask_counts(pred, gt)
                row = dict(task_id=query['task_id'], class_id=item.id, class_name=item.name,
                           inference_s=inference_s, labeled=labeled, full=full,
                           labeled_rates=rates(labeled), full_rates=rates(full),
                           gt_pixels=int(gt.sum()), prediction_pixels=int(pred.sum()))
                for kind, values in [('labeled', labeled), ('full', full)]:
                    for key,value in values.items():
                        totals[item.id][kind][key] += value
                if item.geometry == 'Point':
                    gt_boxes = query['boxes'].get(str(item.id), [])
                    counts = box_counts(components(pred), gt_boxes)
                    row['boxes'] = counts
                    for key,value in counts.items():
                        totals[item.id]['boxes'][key] += value
                folder = out/'masks'/f'task_{query["task_id"]:06d}'
                folder.mkdir(parents=True,exist_ok=True)
                assert cv2.imwrite(str(folder/f'class_{item.id:02d}.png'), pred.astype(np.uint8)*255)
                predictions[item.id] = pred
                rows.append(row)
                frame_rows.append(row)
                print(f'{model_name} task={query["task_id"]} {item.name}: {inference_s:.3f}s '
                      f'IoU(labeled)={row["labeled_rates"]["iou"]}', flush=True)
            try:
                next(iterator)
                raise RuntimeError('Unexpected extra prediction')
            except StopIteration:
                pass
            frame_time = sum(row['inference_s'] for row in frame_rows)
            frame_iou = np.mean([r['labeled_rates']['iou'] for r in frame_rows if r['gt_pixels']])
            left = cv2.resize(draw(rgb, truth, classes),(1920,1080))
            right = cv2.resize(draw(rgb,predictions,classes),(1920,1080))
            canvas = np.concatenate([left,right],axis=1)
            canvas = cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR)
            cv2.rectangle(canvas,(0,0),(3840,65),(15,15,15),-1)
            title=(f'{model_name} | task {query["task_id"]} | GT left / prediction right | '
                   f'{len(classes)}-way 1-shot | {frame_time:.3f}s | labeled mIoU={frame_iou:.1%}')
            cv2.putText(canvas,title,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
            legend=' | '.join(c.name for c in classes)
            cv2.putText(canvas,legend,(12,53),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1,cv2.LINE_AA)
            (out/'overlays').mkdir(exist_ok=True)
            assert cv2.imwrite(str(out/'overlays'/f'task_{query["task_id"]:06d}.jpg'),canvas)
            if writer is None:
                writer=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1,(3840,1080))
                if not writer.isOpened():
                    raise OSError('Could not create comparison video')
            writer.write(canvas)
            frames.append(dict(task_id=query['task_id'],inference_s=frame_time,
                               labeled_pixel_fraction=float(valid.mean()),labeled_miou=float(frame_iou)))
    finally:
        if writer is not None:
            writer.release()
    by_class=[]
    for item in classes:
        value=totals[item.id]
        by_class.append(dict(class_id=item.id,class_name=item.name,geometry=item.geometry,
                             labeled=value['labeled'],labeled_rates=rates(value['labeled']),
                             full=value['full'],full_rates=rates(value['full']),
                             boxes=value['boxes'] if item.geometry=='Point' else None))
    point_counts={key:sum(row['boxes'][key] for row in by_class if row['boxes'] is not None)
                  for key in ('tp','fp','fn')}
    point_rates=rates(point_counts)
    summary=dict(model=model_name,ways=len(classes),shots=1,held_out_tasks=[1,3],
                 labeled_region_miou=float(np.mean([c['labeled_rates']['iou'] for c in by_class])),
                 full_frame_miou_assuming_unlabeled_background=float(np.mean([c['full_rates']['iou'] for c in by_class])),
                 point_box_f1_at_50=point_rates['dice'],point_box_counts=point_counts,
                 point_box_precision=point_rates['precision'],point_box_recall=point_rates['recall'],
                 preparation_s=preparation_s,mean_frame_inference_s=float(np.mean([f['inference_s'] for f in frames])),
                 second_frame_inference_s=frames[-1]['inference_s'],
                 peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                 frames=frames,by_class=by_class,backend=backend.metadata(),supports=selection.metadata(),
                 split=split,api_enabled=False,
                 protocol=dict(primary='IoU restricted to the union of saved brush annotations; overlapping class masks remain independent',
                               secondary='Full-frame IoU assumes unannotated pixels are background; not an exhaustive-label benchmark',
                               boxes='Class-aware maximum-cardinality matching at IoU .5; predicted instances are connected components >=16 pixels',
                               limitations='Two annotated frames from this site, not an independent-scene test; no threshold tuning on holdout'))
    summary['provenance']=capture(split,summary['backend'],__file__)
    summary['runtime']=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),seed=0)
    write_json(out/'metrics.json',summary)
    write_json(out/'timings.json',dict(preparation_s=preparation_s,frames=frames,class_episodes=rows))
    with (out/'metrics.csv').open('w',newline='') as stream:
        fields=['class_name','labeled_iou','full_iou','foreground_recall']
        table=csv.DictWriter(stream,fieldnames=fields);table.writeheader()
        for c in by_class:
            table.writerow(dict(class_name=c['class_name'],labeled_iou=c['labeled_rates']['iou'],
                                full_iou=c['full_rates']['iou'],foreground_recall=c['labeled_rates']['recall']))
    print(json.dumps({k:summary[k] for k in ['model','ways','labeled_region_miou',
          'full_frame_miou_assuming_unlabeled_background','point_box_f1_at_50','mean_frame_inference_s']},indent=2),flush=True)


if __name__=='__main__':
    main()
