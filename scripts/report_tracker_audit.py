"""Summarize correctness replays on frozen PET detections; no accuracy claims."""
import argparse
import html
import json
from pathlib import Path
import shutil


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pet-raw', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    runs, sequences, hashes = [], [], []
    directories = {}
    for phase in ['before', 'after', 'botsort_controls_v3']:
        result = json.loads((root / phase / 'results.json').read_text())
        for run in result['runs']:
            directory = root / phase / run['name']
            frames = read_rows(directory / 'frames.jsonl')
            assert run['audit_passed'] and run['api_enabled'] is False
            assert len(frames) == run['frames'] == run['tracker_updates'] == 65
            sequences.append([row['source'] for row in frames])
            hashes.append(run['input_cache_sha256'])
            runs.append(dict(phase=phase, name=run['name'], observations=run['observations'],
                registry_objects=run['registry_objects'], person_registry_ids=run['person_registry_ids'],
                overlapping_person_pairs=run['overlapping_person_pairs'], tracker=run['tracker'],
                mean_tracker_ms=1000*run['tracker_timing']['mean_s'],
                p95_tracker_ms=1000*run['tracker_timing']['p95_s']))
            directories[phase + '/' + run['name']] = directory
    assert len(set(hashes)) == 1 and all(rows == sequences[0] for rows in sequences)
    raw = root / 'pet_raw'
    raw.mkdir(exist_ok=True)
    for index in range(65):
        shutil.copyfile(args.pet_raw / f'{index:04d}.jpg', raw / f'{index:04d}.jpg')
    selected = [
        ('PET-DINO raw: class and confidence, score >=0.35', 'pet_raw'),
        ('UDC before: person roles + weak recovery', 'before/udc_person_roles_low_recovery/overlays'),
        ('UDC corrected: person roles + weak recovery', 'after/udc_person_roles_low_recovery/overlays'),
        ('BoT-SORT corrected: person roles, high candidates', 'botsort_controls_v3/botsort_person_roles/overlays'),
        ('BoT-SORT corrected: person roles + weak recovery', 'botsort_controls_v3/botsort_person_roles_low_recovery/overlays'),
    ]
    for _, directory in selected:
        assert all((root / directory / f'{index:04d}.jpg').is_file() for index in range(65))
    differences = {}
    for label in ['udc_class_gated', 'udc_person_roles', 'udc_person_roles_low_recovery']:
        old = read_rows(directories['before/' + label] / 'frames.jsonl')
        new = read_rows(directories['after/' + label] / 'frames.jsonl')
        def signature(row):
            return [(t['track_id'], t['class_id'], t['bbox'], t['confidence'], t.get('role_evidence')) for t in row['tracks']]
        differences[label] = [a['frame'] for a, b in zip(old, new) if signature(a) != signature(b)]
    report = dict(frames=65, detector='frozen PET-DINO 800 / all 141 supports / 28 ways',
                  input_cache_sha256=hashes[0], same_source_frames=True, api_enabled=False,
                  accuracy_evaluated=False, unit_tests_passed=66, runs=runs,
                  udc_changed_frames_excluding_random_uuids_and_timing=differences,
                  limitations=[
                      'No identity ground truth: counts and overlapping boxes are diagnostic proxies.',
                      'UUIDs are independent per replay; compare continuity within a view, not UUID strings across views.',
                      'Tracker timings include camera compensation and association, but detector features are precomputed.',
                      'No MASA, Deep OC-SORT, GLoMOT or other literature candidate has been benchmarked here.'])
    (root / 'audit_comparison.json').write_text(json.dumps(report, indent=2))
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in [
        row['phase'] + '/' + row['name'], f"{row['mean_tracker_ms']:.2f}",
        f"{row['p95_tracker_ms']:.2f}", row['observations'], row['registry_objects'], row['overlapping_person_pairs']]) + '</tr>' for row in runs)
    tiles = ''.join(f'<figure><figcaption>{html.escape(name)}</figcaption><a target="_blank"><img data-directory="{directory}"></a></figure>' for name, directory in selected)
    options = ''.join(f'<option value="{i}">{i:04d}</option>' for i in range(65))
    page = '''<!doctype html><meta charset="utf-8"><title>PET-DINO tracker audit</title>
<style>body{font:16px system-ui;margin:24px;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:white;padding:8px}figure:first-child{grid-column:1/-1;max-width:1280px}img{width:100%}td,th{padding:6px;text-align:left}select{font-size:18px}</style>
<h1>PET-DINO: tracker correctness audit at 1 Hz</h1><p>Every view uses identical frozen PET detections and source images. Raw boxes are separate from tracker output. Grey boxes in tracked views are raw hypotheses; green labels are observed tracks. Each replay has an independent registry, so UUID strings differ across views. Counts are not identity accuracy; query identity ground truth is unavailable. API disabled.</p>
<p><a href="audit_comparison.json">Measurements and changed frames</a></p>
<table><tr><th>Replay</th><th>Tracker mean ms</th><th>p95 ms</th><th>Observations</th><th>UUIDs</th><th>Person overlap pairs</th></tr>TABLE</table>
<p>Feature extraction is precomputed; tracker time includes camera compensation where enabled.</p><label>Frame <select id="frame">OPTIONS</select></label><main>TILES</main>
<script>function update(){const f=document.getElementById('frame').value.padStart(4,'0');document.querySelectorAll('img').forEach(i=>{i.src=i.dataset.directory+'/'+f+'.jpg';i.parentElement.href=i.src;});}document.getElementById('frame').onchange=update;update();</script>'''
    (root / 'comparison.html').write_text(page.replace('TABLE', table).replace('OPTIONS', options).replace('TILES', tiles))
    print(json.dumps(dict(frames=65, replay_variants=len(runs), accuracy_evaluated=False,
                          changed_frames=differences), indent=2))


if __name__ == '__main__':
    main()
