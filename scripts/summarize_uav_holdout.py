"""Collect the fixed UAV holdout runs and validate their shared protocol/artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path('/mnt/d/projects/triffid-hua-perception')
MODELS = ['FoRIS', 'FSS-SAM3-text', 'SANSA', 'FROST', 'FSS-SAM3']


def main():
    output = ROOT/'samples/UAV-PUI/holdout_1_3_1shot'
    output.mkdir(parents=True, exist_ok=True)
    reports = {name: json.loads((ROOT/'samples'/name/'UAV/holdout_1_3_1shot/metrics.json').read_text())
               for name in MODELS}
    reference = reports['SANSA']
    rows = []
    for name, report in reports.items():
        assert report['supports'] == reference['supports'], f'Different supports: {name}'
        assert report['split'] == reference['split'], f'Different split: {name}'
        group_scores = {}
        for geometry in ['Point', 'Polygon']:
            values = [c['labeled_rates']['iou'] for c in report['by_class'] if c['geometry'] == geometry]
            group_scores[geometry] = sum(values)/len(values)
        folder = ROOT/'samples'/name/'UAV/holdout_1_3_1shot'
        video = cv2.VideoCapture(str(folder/'comparison.mp4'))
        assert video.isOpened() and int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        video.release()
        for task in (1, 3):
            assert (folder/'overlays'/f'task_{task:06d}.jpg').exists()
            assert len(list((folder/'masks'/f'task_{task:06d}').glob('class_*.png'))) == 6
        rows.append(dict(method=name,
                         labeled_miou_percent=100*report['labeled_region_miou'],
                         full_frame_miou_percent=100*report['full_frame_miou_assuming_unlabeled_background'],
                         point_mask_miou_percent=100*group_scores['Point'],
                         polygon_mask_miou_percent=100*group_scores['Polygon'],
                         point_box_f1_percent=100*report['point_box_f1_at_50'],
                         warm_frame_s=report['second_frame_inference_s'],
                         mean_frame_s=report['mean_frame_inference_s'],
                         preparation_s=report['preparation_s'],
                         peak_gpu_mib=report['peak_gpu_allocated_mib'], artifacts=str(folder)))
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'comparison.json').write_text(json.dumps(dict(
        protocol='6-way 1-shot; task/image 1 and 3 content held out; identical supports', rows=rows), indent=2)+'\n')
    text = ['# UAV-PUI held-out comparison', '',
            'Queries: Label Studio tasks **1 and 3**, view_000deg/view_180deg; saved annotations 17 and 18.', '',
            'Tasks 1/3 and exact image-content copies are excluded from the active support bank. '
            'The frozen manifest and split are in `data/UAV-PUI/holdout_1_3/`. All methods use **one '
            'identical support per each of six classes**: green tree, window, building, dirt road, green grass '
            'and furniture. Dry grass has no remaining support after this holdout and is excluded from the score.', '',
            'Primary mIoU pools pixel counts per class inside the union of saved brush annotations, then '
            'averages the six class IoUs. Overlapping class masks remain independent. Annotation coverage '
            'is 90.19% and 95.01%. Full-frame IoU is separate and assumes unlabeled pixels are background.', '',
            'Point-box F1 applies to **window and furniture only**, using class-aware maximum-cardinality '
            'matching at IoU >=0.5. Predicted boxes come from eight-connected mask components of at least '
            '16 pixels. This is not COCO AP or a metric for all UAV point classes.', '',
            '| Method | Annotated mIoU | Full-frame mIoU* | Point-box F1 | Warm six-class frame |',
            '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        name = {'FSS-SAM3-text': 'FSS-SAM3 (text + visual)',
                'FSS-SAM3': 'FSS-SAM3 (visual only ablation)'}.get(row['method'], row['method'])
        text.append(f"| {name} | {row['labeled_miou_percent']:.1f}% | "
                    f"{row['full_frame_miou_percent']:.1f}% | {row['point_box_f1_percent']:.1f}% | "
                    f"{row['warm_frame_s']:.2f} s |")
    text += ['', '*Full-frame score assumes unannotated pixels are background. Warm latency is the '
             'second query frame, excluding disk reads, evaluation, overlays and video writing. Cold/mean '
             'class timings, preparation and GPU peak allocation are retained in each run\'s JSON.', '',
             'All methods run independent class episodes rather than a native joint six-way forward. '
             'The summed timings cover all six classes on the same RTX 3060. SANSA uses 640 inputs, '
             'FSS-SAM3 a 1008 collage (40% query, 60% support), and FoRIS/FROST native 1024 inputs. '
             'Masks are restored to original 3840x2160 resolution.', '',
             'FSS-SAM3 text+visual is the authors\' default; the visual-only ablation removes class text. '
             'FoRIS uses released default bilinear refinement, without optional CRF. FoRIS and FROST '
             'use the same official local DINOv3-L/16 checkpoint. No training, threshold tuning on '
             'held-out masks, tracking or API calls occur.', '',
             '## Per-class annotated IoU', '',
             '| Class | FoRIS | FSS-SAM3 text+visual | SANSA | FROST |',
             '| --- | ---: | ---: | ---: | ---: |']
    for index, item in enumerate(reference['by_class']):
        values = [reports[name]['by_class'][index]['labeled_rates']['iou']*100 for name in MODELS[:4]]
        text.append('| '+item['class_name']+' | '+' | '.join(f'{value:.1f}%' for value in values)+' |')
    text += ['', 'FROST leads pooled mask mIoU; FSS-SAM3 text+visual leads component box F1, driven by '
             'windows. FSS-SAM3 misses furniture and misses green tree in task 3 with text prompting. '
             'SANSA is fastest. None of these unoptimized six-class pipelines meets 1 frame/s on this '
             'GPU. Two frames from this site do not establish unseen-site or unseen-object accuracy.', '',
             '## Artifacts', '']
    for name in MODELS:
        prefix = '../../'+name+'/UAV/holdout_1_3_1shot/'
        text.append(f'- {name}: [video]({prefix}comparison.mp4), '
                    f'[task 1]({prefix}overlays/task_000001.jpg), [task 3]({prefix}overlays/task_000003.jpg), '
                    f'[metrics]({prefix}metrics.json), [timings]({prefix}timings.json).')
    text += ['', 'Reproduce with `bash scripts/run_uav_holdout.sh <foris|fss-sam3-text|sansa|frost|fss-sam3>` '
             'using fresh output directories. Source dataset SHA256: `'+reference['split']['source_archive_sha256']+'`.']
    (output/'README.md').write_text('\n'.join(text)+'\n')
    active = ROOT/'uav_supports/PUI'
    manifest = json.loads((active/'bank.json').read_text())
    report = json.loads((active/'import_report.json').read_text())
    heldout = set(reference['split']['held_out_image_sha256'])
    count = 0
    for name, entries in manifest['classes'].items():
        for entry in entries[:report['classes'][name]['imported']]:
            assert hashlib.sha256((active/entry['image']).read_bytes()).hexdigest() not in heldout
            count += 1
    assert count == 47
    print('Verified all 47 active supports exclude held-out content; all model videos, masks and overlays exist.')
    print('\n'.join(text[11:18]))


if __name__ == '__main__':
    main()
