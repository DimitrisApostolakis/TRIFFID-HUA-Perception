"""Build a reviewable report from completed metrics; never rank failed jobs."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def percentage(value):
    return '—' if value is None else f'{value*100:.1f}%'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uav',type=Path,required=True)
    p.add_argument('--ugv',type=Path,required=True)
    p.add_argument('--ugv-retry',type=Path,nargs='*',default=[])
    p.add_argument('--uav-retry',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,default=ROOT/'samples/evaluation/REPORT.md')
    args=p.parse_args()
    text=['# PUI frozen evaluation','',
        'Target: current RTX 3060, at least one processed frame/s for the complete live pipeline after warmup.',
        'Two annotated frames per platform are a site diagnostic. These scores do not validate production accuracy or re-ID.',
        '', '## UAV: tasks 1 and 3','',
        'Same six one-shot class episodes. Primary score uses saved brush pixels. RSRM/FSSDINO use the user-authorized DINOv3-L replacement.', '',
        '| Model | State | Annotated mIoU | Full-frame mIoU* | Component-box F1 | Cold frame | Second frame |',
        '|---|---|---:|---:|---:|---:|---:|']
    for platform,folder in [('uav',args.uav),('ugv',args.ugv)]:
        state=json.loads((folder/'status.json').read_text())
        overrides={}
        retries=args.ugv_retry if platform=='ugv' else args.uav_retry
        if retries:
            for retry in retries:
                rs=json.loads((retry/'status.json').read_text())
                for name,status in rs['models'].items():
                    if status['status']=='complete':overrides[name]=(status,retry/name/'metrics.json')
            for name,(status,_) in overrides.items():
                if name not in state['models']:state['models'][name]=status
        if platform=='ugv':
            text += ['', '## UGV: ROSBAG-PUI tasks 49 and 50','',
                'Threshold-truncated COCO bbox AP; native fixed support/score settings. Helmet has no support and is reported as unsupported. Maskless detectors receive no synthetic masks.', '',
                '| Model | State | Bbox AP50–95 | Bbox F1 @ .5 | Annotated mask mIoU | Cold query | Second query |',
                '|---|---|---:|---:|---:|---:|---:|']
        for model,status in state['models'].items():
            metric=folder/model/'metrics.json'
            if model in overrides:status,metric=overrides[model]
            if status['status']!='complete' or not metric.is_file():
                text.append(f'| {model} | {status["status"]} | — | — | — | — | — |')
                continue
            m=json.loads(metric.read_text())
            cold=m['frames'][0]['inference_s']
            if platform=='uav':
                values=[percentage(m[k]) for k in ('labeled_region_miou','full_frame_miou_assuming_unlabeled_background','point_box_f1_at_50')]
            else:
                values=[percentage(m['coco_bbox']['AP_50_95']),percentage(m['box_rates']['dice']),percentage(m['labeled_region_miou'])]
            text.append(f'| {m["model"]} | complete | '+ ' | '.join(values)+f' | {cold:.3f} s | {m["second_frame_inference_s"]:.3f} s |')
    text += ['', '*Full-frame UAV IoU assumes unannotated pixels are background.',
        'Inference excludes initialization/support preparation, image I/O, scoring, rendering, tracking, projection and delivery. A second-frame inference time below one second is only an initial screening result, not a passed live production gate.',
        '', 'Re-ID: offline UGV comparisons use MASA-R50; the live ROS node still uses legacy ByteTracker without learned appearance. See `docs/REID_SYSTEM.md` and `docs/EVALUATION_AND_PRODUCTION.md`.',
        '', f'UAV run: `{args.uav}`. UGV run: `{args.ugv}`.',
        'Each model directory contains its metrics and visual artifacts; suite-level files contain logs and run status.']
    if args.ugv_retry:text += ['Retry runs: '+', '.join(f'`{p}`' for p in args.ugv_retry)+'. Only successful retry metrics replace failed rows.']
    if args.uav_retry:text += ['Additional UAV / retry runs: '+', '.join(f'`{p}`' for p in args.uav_retry)+'. Successful additional models are appended; successful retries replace failed rows.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(text)+'\n')
    print(args.output)

if __name__=='__main__':main()
