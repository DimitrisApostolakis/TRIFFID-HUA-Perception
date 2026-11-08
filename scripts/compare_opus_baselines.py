"""Compare matched saved runs and show raw predictions beside tracked outputs."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='Display name=run directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    results, panels, source_frames, common_episode = [], [], None, None
    for spec in args.run:
        name, directory = spec.split('=', 1)
        directory = Path(directory).resolve()
        summary = json.loads((directory / 'summary.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        rows = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
        current = [row['source'] for row in rows]
        if source_frames is None:
            source_frames = current
        if current != source_frames:
            raise ValueError('All comparisons must use the same ordered query frames')
        detector = summary['detector']
        episode = dict(classes=config['classes'], supports=detector['support_bank']['classes'],
                       confidence=config['confidence_threshold'], tracker=summary['tracker'])
        if common_episode is None:
            common_episode = episode
        if episode != common_episode:
            raise ValueError('Comparison classes, supports, confidence and tracker settings must match')
        if summary['api_enabled'] or detector['query_encodings'] != len(rows):
            raise ValueError('Expected no API calls and exactly one backbone encoding per frame')
        precision = detector.get('precision', config.get('pet_precision', 'float32'))
        raw_dir = root / (f'run_{len(results):02d}_' + summary['backend'] + '_' + detector.get('prompt_mode', 'visual') + '_' + str(detector['image_size']) + '_' + precision + '_raw')
        raw_dir.mkdir(exist_ok=True)
        counts = Counter()
        for row in rows:
            image = cv2.imread(str(Path(config['dataset']) / row['source']['file'])) if 'dataset' in config else None
            if image is None:
                # Older runner configs store only detector parameters.
                workspace = Path(__file__).resolve().parents[1]
                image = cv2.imread(str(workspace / 'data/PUI/rgb_frames' / row['source']['file']))
            if image is None:
                raise ValueError('Query image missing')
            for detection in row['detections']:
                counts[detection['class_name']] += 1
                a, b, c, d = map(int, detection['bbox'])
                cid = detection['class_id']
                color = (int(70 + cid * 61 % 185), int(70 + cid * 97 % 185), int(70 + cid * 29 % 185))
                cv2.rectangle(image, (a, b), (c, d), color, 2)
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                cv2.putText(image, label, (max(0, a), max(18, b - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
            assert cv2.imwrite(str(raw_dir / f"{row['frame']:04d}.jpg"), image)
        warm = rows[1:]
        results.append(dict(name=name, directory=str(directory), frames=len(rows), ways=summary['ways'],
            shots=summary['shots'], supports=detector.get('shots_per_class'),
            image_size=detector['image_size'], precision=precision,
            warm_mean_total_s=float(np.mean([row['total_s'] for row in warm])),
            warm_p95_total_s=float(np.percentile([row['total_s'] for row in warm], 95)),
            warm_mean_inference_s=float(np.mean([row['inference_s'] for row in warm])),
            warm_frames_over_1s=sum(row['total_s'] > 1 for row in warm),
            first_frame_s=summary['timing']['first_frame_s'], preparation_s=summary['preparation_s'],
            runtime=summary.get('runtime'),
            peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes'], detections=sum(counts.values()),
            class_detections=dict(counts), tracked_observations=summary['tracked_observations'],
            unique_objects=summary['unique_objects'], tracker=summary['tracker']))
        panels.append((name + ': raw detections', os.path.relpath(raw_dir, root)))
        panels.append((name + ': UDC / registry', os.path.relpath(directory / 'overlays', root.resolve())))
    report = dict(runs=results, same_query_frames=True, same_episode_and_tracker=True, api_enabled=False,
                  accuracy_evaluated=False, limitations=[
                      'No query ground truth: counts are not precision/recall or physical identity accuracy.',
                      'OPUS uses a square canvas (native 640, optional higher resolutions); PET native 800 uses a short-edge resize. These are deployable configurations, not an isolated architecture ablation.',
                      'Runtime is measured on this shared RTX 3060; CPU includes decoding, tracker, registry and JPEG export.',
                      'Per-frame timer excludes one-time initialization and the final frame JSONL logging/printing after overlay export.',
                      'RGB persistence cannot guarantee no repeated physical object after arbitrary viewpoint changes.'])
    (root / 'comparison_results.json').write_text(json.dumps(report, indent=2))
    table_rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in [
        row['name'], f"{row['warm_mean_total_s']:.3f}", f"{row['warm_p95_total_s']:.3f}",
        row['warm_frames_over_1s'], row['detections'], row['unique_objects']]) + '</tr>' for row in results)
    options = ''.join(f'<option value="{index}">{index:04d}</option>' for index in range(len(source_frames)))
    tiles = ''.join(f'<figure><figcaption>{html.escape(name)}</figcaption><a target="_blank"><img data-directory="{html.escape(path, quote=True)}"></a></figure>' for name, path in panels)
    page = '''<!doctype html><meta charset="utf-8"><title>OPUS baseline comparison</title>
<style>body{font-family:system-ui;margin:24px;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}td,th{padding:6px 12px;text-align:left}select{font-size:18px}</style>
<h1>OPUS / PET-DINO: same one-Hz frames</h1><p>Raw panels show class + confidence before tracking. Tracked panels show observed IDs; grey boxes are detector hypotheses. API disabled. Counts do not measure accuracy; query ground truth is unavailable. Each model uses its native preprocessing.</p>
<table><tr><th>Run</th><th>Warm mean (s)</th><th>Warm p95 (s)</th><th>Warm frames &gt;1s</th><th>Detections</th><th>Registry IDs</th></tr>TABLE</table>
<p><a href="comparison_results.json">Full timings and class counts</a></p><label>Frame <select id="frame">OPTIONS</select></label><main>TILES</main>
<script>function update(){const f=document.getElementById('frame').value.padStart(4,'0');document.querySelectorAll('img').forEach(i=>{i.src=i.dataset.directory+'/'+f+'.jpg';i.parentElement.href=i.src;});}document.getElementById('frame').onchange=update;update();</script>'''
    (root / 'comparison.html').write_text(page.replace('TABLE', table_rows).replace('OPTIONS', options).replace('TILES', tiles))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
