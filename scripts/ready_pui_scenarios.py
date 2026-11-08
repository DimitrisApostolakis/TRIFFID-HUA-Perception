"""Reindex staged camera bags, verify pairs, and extract every RGB frame in ROS."""
import heapq
import argparse
import json
from pathlib import Path
import subprocess
import sys

import rosbag2_py as bag
import yaml

ROOT = Path('/ws/data/PUI/scenarios')
TOPICS = {
    '/b2/camera_front_435i/realsense_front_435i/color/image_raw',
    '/b2/camera_front_435i/realsense_front_435i/color/camera_info',
    '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw',
    '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/camera_info',
    '/b2/camera_front_435i/realsense_front_435i/extrinsics/depth_to_color',
    '/b2/ugv_status', '/fix', '/tf', '/tf_static',
    '/b2/nicla/magnetometer/heading', '/b2/nicla/magnetometer/heading_broadcasted',
}


def merge(paths, output):
    readers, queue, known, counts = [], [], set(), {}
    writer = bag.SequentialWriter()
    writer.open(bag.StorageOptions(uri=str(output), storage_id='mcap'), bag.ConverterOptions('', ''))
    for i, path in enumerate(paths):
        reader = bag.SequentialReader()
        reader.open(bag.StorageOptions(uri=str(path)), bag.ConverterOptions('', ''))
        for topic in reader.get_all_topics_and_types():
            if topic.name in TOPICS and topic.name not in known:
                writer.create_topic(topic)
                known.add(topic.name)
        reader.set_filter(bag.StorageFilter(topics=sorted(TOPICS)))
        readers.append(reader)
        if reader.has_next():
            topic, data, timestamp = reader.read_next()
            heapq.heappush(queue, (timestamp, i, topic, data))
    last = None
    while queue:
        timestamp, i, topic, data = heapq.heappop(queue)
        if last is not None and timestamp < last:
            raise RuntimeError('Nonmonotonic paired recording')
        writer.write(topic, data, timestamp)
        last = timestamp
        counts[topic] = counts.get(topic, 0) + 1
        if readers[i].has_next():
            topic, data, timestamp = readers[i].read_next()
            heapq.heappush(queue, (timestamp, i, topic, data))
    del writer
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['scenario_1','scenario_2'], action='append')
    args = parser.parse_args()
    summaries = json.loads((ROOT/'summary.json').read_text()) if (ROOT/'summary.json').exists() else {}
    for scenario in args.scenario or ('scenario_1', 'scenario_2'):
        root = ROOT / scenario
        paths = sorted((root / 'bags').iterdir())
        for path in paths:
            if not (path / 'metadata.yaml').exists():
                subprocess.run(['ros2', 'bag', 'reindex', str(path)], check=True)
        infos = []
        cameras = []
        for path in paths:
            info = yaml.safe_load((path / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
            start = info['starting_time']['nanoseconds_since_epoch']
            end = start + info['duration']['nanoseconds']
            rgb_count = sum(t['message_count'] for t in info['topics_with_message_count']
                if t['topic_metadata']['type'] == 'sensor_msgs/msg/Image'
                and '/color/' in t['topic_metadata']['name'])
            infos.append(dict(bag=path.name, start_ns=start, end_ns=end, rgb_count=rgb_count,
                              duration_s=(end-start)/1e9))
            if rgb_count:
                cameras.append(path)
        overlap_start = max(x['start_ns'] for x in infos)
        overlap_end = min(x['end_ns'] for x in infos)
        summary = dict(bags=infos, overlap_seconds=max(0, overlap_end-overlap_start)/1e9,
                       timestamps_modified=False, paired=False)
        if overlap_end >= overlap_start:
            counts = merge(paths, root / 'paired_bag')
            summary.update(paired=True, paired_topic_counts=counts)
        else:
            summary['problem'] = 'Camera and NUC recordings do not overlap; do not replay as a synchronized pair.'
            summary['gap_seconds'] = (overlap_start-overlap_end)/1e9
        (root / 'readiness.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(scenario, json.dumps(summary), flush=True)
        if not cameras:
            raise RuntimeError(f'No RGB camera bag in {scenario}')
        subprocess.run([sys.executable, '/ws/scripts/extract_rosbag_rgb.py',
                        *map(str,cameras), '--output', str(root/'rgb_frames')], check=True)
        # Reference the exported PNGs, retaining one frame per recorded second
        # for the existing 1 FPS evaluation workflow without duplicate images.
        seen, selected = set(), []
        for line in (root/'rgb_frames/frames.jsonl').read_text().splitlines():
            record = json.loads(line)
            key = (record['bag'], record['topic'], record['bag_timestamp_ns']//10**9)
            if key not in seen:
                seen.add(key)
                selected.append(record)
        (root/'rgb_frames/frames_1fps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
        summary['rgb_frames'] = sum(x['rgb_count'] for x in infos)
        summary['rgb_frames_1fps'] = len(selected)
        (root/'readiness.json').write_text(json.dumps(summary, indent=2)+'\n')
        summaries[scenario] = summary
    (ROOT/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')


if __name__ == '__main__':
    main()
