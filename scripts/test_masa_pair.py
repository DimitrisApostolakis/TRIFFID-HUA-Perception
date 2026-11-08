"""MASA-R50 association test using two annotated query images; no HTML/API."""
import importlib
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
import torch

ROOT = Path('/mnt/d/projects/triffid-hua-perception')
REPO = ROOT / 'ugv_models/masa'
OUT = ROOT / 'archive/samples_legacy_20261106/masa_pair_20261106/class_safe'


def main():
    from mmengine.config import Config
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample, TrackDataSample
    register_all_modules()
    tracker_source = REPO / 'masa/models/tracker/masa_tao_tracker.py'
    source = tracker_source.read_text()
    marker = '# Respect the advertised category gate for heterogeneous objects.'
    if marker not in source:
        needle = '            # track according to match_scores\n'
        if source.count(needle) != 1:
            raise RuntimeError('Unexpected MASA source: review category-gate compatibility')
        gate = ('            ' + marker + '\n'
                '            if self.with_cats:\n'
                '                match_scores = match_scores.masked_fill(\n'
                '                    labels[:, None] != memo_labels[None, :], -1.0\n'
                '                )\n\n')
        tracker_source.write_text(source.replace(needle, gate + needle))
    # Load only official inference modules, without unrelated demo/dataset dependencies.
    for name, relative in [('masa', 'masa'), ('masa.models', 'masa/models'),
                           ('masa.models.necks', 'masa/models/necks'),
                           ('masa.models.mot', 'masa/models/mot'),
                           ('masa.models.tracker', 'masa/models/tracker')]:
        module = types.ModuleType(name)
        module.__path__ = [str(REPO / relative)]
        sys.modules[name] = module
    for name in ['masa.models.losses', 'masa.models.necks.deform_fusion',
                 'masa.models.tracker.masa_tao_tracker', 'masa.models.mot.masa']:
        importlib.import_module(name)
    cfg = Config.fromfile(str(REPO / 'configs/masa-one/masa_r50_plug_and_play.py'))
    cfg.model.tracker.update(fps=1, memo_tracklet_frames=60, max_distance=-1, with_cats=True)
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(ROOT / 'ugv_models/masa_r50.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    del checkpoint
    model.cuda().eval().requires_grad_(False)
    tasks = json.loads(Path('/mnt/c/Users/apost/Downloads/project-4-at-2026-11-06-00-53-a91d4d1a.json').read_text())
    paths = sorted(Path('/mnt/c/Users/apost/Desktop/rosbag').glob('*.png'))
    first = int(paths[0].stem.split('_')[0])
    from triffid_ugv_perception.ontology import BY_NAME
    OUT.mkdir(parents=True, exist_ok=False)
    summaries, previous = [], {}
    for number, path in enumerate(paths):
        task = next(t for t in tasks if path.name in t['data']['image'])
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        rows = []
        results = [r for a in task['annotations'] if not a.get('was_cancelled') for r in a['result']]
        for r in results:
            v = r['value']
            if r['type'] != 'rectanglelabels' or 'x' not in v:
                continue
            label = v['rectanglelabels'][0].lower()
            box = [v['x']*w/100, v['y']*h/100, (v['x']+v['width'])*w/100, (v['y']+v['height'])*h/100]
            rows.append(dict(region_id=r['id'], class_id=BY_NAME[label].id, class_name=label, bbox=box))
        scale = 1024/max(h, w)
        nw, nh = round(w*scale), round(h*scale)
        resized = cv2.resize(image, (nw, nh))[:, :, ::-1].copy()
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().cuda()
        mean = tensor.new_tensor([123.675, 116.28, 103.53])[:, None, None]
        std = tensor.new_tensor([58.395, 57.12, 57.375])[:, None, None]
        tensor = (tensor-mean)/std
        tensor = torch.nn.functional.pad(tensor, (0, (-nw)%32, 0, (-nh)%32))
        elapsed_s = (int(path.stem.split('_')[0])-first)/1e9
        sample = DetDataSample(metainfo=dict(frame_id=round(elapsed_s), ori_shape=(h,w),
            img_shape=(nh,nw), scale_factor=(nw/w,nh/h)))
        sample.det_bboxes = torch.tensor([[*r['bbox'], 1.] for r in rows], device='cuda')
        sample.det_labels = torch.tensor([r['class_id'] for r in rows], device='cuda')
        sequence = TrackDataSample(video_data_samples=[sample])
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pred = model.predict(tensor[None,None], [sequence])[0][0].pred_track_instances
        torch.cuda.synchronize()
        duration = time.perf_counter()-start
        records = []
        from triffid_ugv_perception.ontology import BY_ID
        for box, label, identity in zip(pred.bboxes.cpu().tolist(), pred.labels.cpu().tolist(), pred.instances_id.cpu().tolist()):
            label = BY_ID[label].name
            reused = identity in previous
            color = (int(70+identity*71%185), int(70+identity*113%185), int(70+identity*43%185))
            a,b,c,d = map(int, box)
            cv2.rectangle(image, (a,b), (c,d), color, 2)
            text = f'{label} ID{identity} ' + ('SEEN' if reused else 'NEW')
            cv2.putText(image, text, (max(0,a),max(16,b-5)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2)
            records.append(dict(class_name=label, track_id=identity, bbox=box, reused=reused,
                                previous_class=previous.get(identity)))
        cv2.imwrite(str(OUT/f'{number+1:02d}_masa.jpg'), image)
        summaries.append(dict(image=path.name, task_id=task['id'], elapsed_s=elapsed_s,
                              input_boxes=len(rows), tracks=len(records), reused_ids=sum(r['reused'] for r in records),
                              tracking_s=duration, tracks_detail=records))
        previous.update({r['track_id']:r['class_name'] for r in records})
    report = dict(method='official MASA-R50; annotated boxes as inputs, not PET predictions',
                  strict_checkpoint_load=True, memory_seconds=60, spatial_gate=False, class_compatibility=True,
                  tracker_change='enforce existing with_cats option in matching scores; learned weights unchanged',
                  frames=summaries, unique_ids=len(previous), api_enabled=False,
                  identity_accuracy='Not scored: annotation regions do not provide cross-frame identity labels',
                  support_bank_changed=False)
    (OUT/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}))
    print([{k:v for k,v in f.items() if k!='tracks_detail'} for f in summaries])


if __name__ == '__main__':
    main()
