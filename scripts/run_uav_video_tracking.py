"""Whole-clip, chronological one-FPS point detections followed by MASA. No metrics."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import cv2

from triffid_uav_perception.export import document, write_json
from triffid_uav_perception.media import VideoReader, read_rgb
from triffid_uav_perception.ontology import CLASSES
from triffid_uav_perception.pipeline import infer_source
from triffid_uav_perception.srt_metadata import SrtIndex, find_sidecar_srt
from triffid_uav_perception.supports import SupportBank

ROOT = Path('/mnt/d/projects/triffid-hua-perception')
VIDEO = Path('/mnt/d/Datasets/PUI/PUI - Raw_data/PUI DRONE/DJI_202610191547_004/DJI_20261019155500_0001_S.MP4')
DATA = ROOT/'data/UAV-PUI/DJI_20261019155500_0001_S/chronological_1fps'
SPLIT = ROOT/'data/UAV-PUI/holdout_1_3/split.json'
RUN = 'full_video_1fps_holdout_1_3'
NAMES = {'sansa':'SANSA', 'fss-sam3-text':'FSS-SAM3', 'foris':'FoRIS', 'frost':'FROST'}
ENVIRONMENTS = {'sansa':'sansa', 'fss-sam3-text':'fss-sam3', 'foris':'frost', 'frost':'frost'}


def prepare():
    manifest = DATA/'frames.json'
    if manifest.exists():
        return manifest
    if DATA.exists():
        raise ValueError('Preserve the partial frame export before retrying')
    DATA.mkdir(parents=True)
    srt = find_sidecar_srt(VIDEO)
    index = SrtIndex.from_file(srt)
    sources = []
    with VideoReader(VIDEO) as reader:
        info = reader.info
        for second in range(math.ceil(info.frame_count/info.fps)):
            number = round(second*info.fps)
            if number >= info.frame_count:
                break
            rgb = reader.read_rgb(number)
            timestamp = number/info.fps
            telemetry = index.at(number, timestamp)
            if telemetry is None:
                raise ValueError(f'Missing SRT at {timestamp}')
            path = DATA/'frames'/f'frame_{number:06d}.jpg'
            path.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY,94]):
                raise OSError('Could not write shared frame')
            sources.append(dict(source_id=f'frame_{number:06d}', source_type='video_frame',
                                frame_index=number, timestamp_s=timestamp, width=info.width, height=info.height,
                                image_file=str(path), telemetry=asdict(telemetry),
                                telemetry_time_delta_s=timestamp-telemetry.timestamp_s,
                                sampling_description='full timeline @ 1 FPS', observations=[]))
    split = json.loads(SPLIT.read_text())
    bankfile = Path(split['bank'])
    assert hashlib.sha256(bankfile.read_bytes()).hexdigest() == split['bank_sha256']
    active_report = json.loads((ROOT/'uav_supports/PUI/import_report.json').read_text())
    classes = [c for c in CLASSES if c.geometry=='Point' and active_report['classes'][c.name]['imported']]
    SupportBank(bankfile).select(classes,shots=1)
    write_json(manifest, dict(video=info.to_dict(), srt_file=str(srt), sampling_fps=1,
                              sources=sources, bank=str(bankfile), bank_sha256=split['bank_sha256'],
                              class_ids=[c.id for c in classes], classes=[c.name for c in classes],
                              held_out_support_tasks=[1,3], accuracy_evaluated=False))
    print(f'Prepared {len(sources)} shared chronological frames; point classes: {[c.name for c in classes]}',flush=True)
    return manifest


def detect(name):
    import torch
    from benchmark_uav_holdout import build_backend
    manifest = json.loads((DATA/'frames.json').read_text())
    out = ROOT/'samples'/NAMES[name]/'UAV'/RUN/'detections'
    final = out/'point_detections.json'
    if final.exists():
        print(f'{name}: using completed detection artifact {final}',flush=True)
        return
    if out.exists():
        raise ValueError(f'Partial output exists: {out}')
    out.mkdir(parents=True)
    bank = Path(manifest['bank'])
    assert hashlib.sha256(bank.read_bytes()).hexdigest() == manifest['bank_sha256']
    classes = tuple(c for c in CLASSES if c.id in manifest['class_ids'])
    selection = SupportBank(bank).select(classes,shots=1)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    backend = build_backend(name)
    print(f'{name}: preparing {len(classes)} point classes for {len(manifest["sources"])} frames',flush=True)
    start = time.perf_counter()
    backend.prepare(classes,selection)
    torch.cuda.synchronize()
    preparation = time.perf_counter()-start
    sources, timings = [], []
    checkpoint = out/'in_progress.jsonl'
    with checkpoint.open('x') as log:
        for i,original in enumerate(manifest['sources']):
            source = dict(original, observations=[])
            rgb = read_rgb(source['image_file'])
            torch.cuda.synchronize()
            start = time.perf_counter()
            infer_source(backend,rgb,classes,source,min_area_px=16)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            sources.append(source)
            timings.append(dict(source_id=source['source_id'],timestamp_s=source['timestamp_s'],
                                detector_and_components_s=elapsed,observations=len(source['observations'])))
            log.write(json.dumps(dict(source=source,timing=timings[-1]))+'\n')
            log.flush()
            print(f'{name} {i+1}/{len(manifest["sources"])} t={source["timestamp_s"]:.2f}s '
                  f'detections={len(source["observations"])} inference={elapsed:.3f}s',flush=True)
    payload = document('point_detections',sources,backend.metadata(),classes)
    payload.update(video=manifest['video'],srt_file=manifest['srt_file'],support_bank=selection.metadata(),
                   sampling_fps=1,held_out_support_tasks=[1,3],accuracy_evaluated=False,
                   postprocessing=dict(min_area_px=16,connectivity=8))
    write_json(final,payload)
    write_json(out/'timings.json',dict(preparation_s=preparation,frames=timings,
                mean_detector_and_components_s=sum(t['detector_and_components_s'] for t in timings)/len(timings),
                accuracy_evaluated=False,api_enabled=False))
    checkpoint.unlink()


def all_models():
    import os
    prepare()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               MPLCONFIGDIR='/tmp/triffid_uav_video_mpl',PYTHONPATH=str(ROOT/'src/triffid_uav_perception'))
    for name in ['fss-sam3-text','sansa','foris','frost']:
        python = '/home/oracle/miniforge3/envs/'+ENVIRONMENTS[name]+'/bin/python'
        subprocess.run([python,__file__,'--detect',name],env=env,check=True)
        base = ROOT/'samples'/NAMES[name]/'UAV'/RUN
        if (base/'tracking/timings.json').exists():
            continue
        subprocess.run(['bash',str(ROOT/'scripts/run_uav_tracking.sh'),
                        str(base/'detections/point_detections.json'),'--output',str(base/'tracking'),
                        '--accept-unscored-masks'],env=env,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detect',choices=list(NAMES))
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.detect:
        detect(args.detect)
    elif args.prepare:
        prepare()
    else:
        all_models()
