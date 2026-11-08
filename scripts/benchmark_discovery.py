#!/usr/bin/env python3
"""Measure discovery overhead with synthetic observations; no models or API calls.

Run with PYTHONPATH=src/triffid_ugv_perception python scripts/benchmark_discovery.py
Temporary state is created on the selected filesystem and removed on exit.
"""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import time

import numpy as np

from triffid_ugv_perception.discovery import DiscoverySettings, DiscoveryService


def benchmark(directory, frames=1200, objects=20):
    logical_time = [0.]
    service = DiscoveryService(directory/'mission.db', DiscoverySettings(),
                               clock=lambda:logical_time[0])
    tracks = [dict(track_id=i,class_name=f'class-{i}',confidence=.9,
                   bbox=[i*20,0,i*20+10,10],appearance=[1.,0.]) for i in range(objects)]
    features = [dict(type='Feature',
                     geometry=dict(type='Point',coordinates=[23.7+i*.0001,37.9]),
                     properties=dict(location_valid=True,heading_valid=True,local_frame=False))
                for i in range(objects)]
    timings = {'ordinary':[], 'periodic_flush':[], 'initial_confirmation':[]}
    try:
        for frame in range(frames):
            logical_time[0] = float(frame)
            flush_due = logical_time[0]-service.registry.last_flush >= service.settings.flush_interval_s
            started = time.perf_counter_ns()
            result, _ = service.process(tracks,features,(frame+1)*10**9)
            elapsed_ms = (time.perf_counter_ns()-started)/1e6
            category = ('initial_confirmation' if result.created_ids else
                        'periodic_flush' if flush_due else 'ordinary')
            timings[category].append(elapsed_ms)
        print(f'Synthetic frames={frames}; objects/frame={objects}; API=off; no model inference')
        for category, values in timings.items():
            if values:
                print(category, dict(n=len(values), median_ms=round(float(np.median(values)),3),
                                     p95_ms=round(float(np.percentile(values,95)),3)))
        print('state', service.status())
        print('files_bytes', sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()))
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=1200)
    parser.add_argument('--objects',type=int,default=20)
    parser.add_argument('--work-dir',type=Path,default=Path(gettempdir()),
                        help='Existing directory on the filesystem to measure')
    args = parser.parse_args()
    if args.frames < 2 or not 1 <= args.objects <= 128:
        parser.error('Use at least two frames and 1–128 objects')
    with TemporaryDirectory(prefix='discovery-benchmark-',dir=args.work_dir) as temporary:
        benchmark(Path(temporary),args.frames,args.objects)


if __name__ == '__main__':
    main()
