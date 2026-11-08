"""Losslessly export every raw color image, preserving bag/header timestamps."""
import argparse
import json
from pathlib import Path
import cv2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import rosbag2_py

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('bags', nargs='+')
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
bridge = CvBridge()
counts = {}
with (out / 'frames.jsonl').open('w') as index:
    for bag in args.bags:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions('', ''))
        topics = [t.name for t in reader.get_all_topics_and_types() if t.type == 'sensor_msgs/msg/Image' and '/color/' in t.name]
        if not topics:
            continue
        reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
        while reader.has_next():
            topic, raw, timestamp = reader.read_next()
            msg = deserialize_message(raw, Image)
            key = Path(bag).name + '/' + topic.strip('/').replace('/', '__')
            number = counts.get(key, 0)
            name = f'{key}/{timestamp}_{number:06d}.png'
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise RuntimeError(f'Refusing to overwrite {path}')
            image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            assert cv2.imwrite(str(path), image)
            index.write(json.dumps(dict(file=name, bag=bag, topic=topic, bag_timestamp_ns=timestamp,
                header_timestamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                frame_id=msg.header.frame_id, source_encoding=msg.encoding, width=msg.width, height=msg.height))+'\n')
            counts[key] = number + 1
(out / 'summary.json').write_text(json.dumps(counts, indent=2)+'\n')
print(json.dumps(counts, indent=2))
