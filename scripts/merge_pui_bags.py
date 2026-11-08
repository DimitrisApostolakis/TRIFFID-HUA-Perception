"""Merge paired bags by original recorded timestamp, retaining required topics."""
import heapq
from pathlib import Path
import rosbag2_py as bag

root=Path('/ws/data/PUI/original_rosbag')
readers=[]
writer=bag.SequentialWriter()
writer.open(bag.StorageOptions(uri=str(root/'paired_bag'),storage_id='mcap'),bag.ConverterOptions('',''))
known=set()
selected={
 '/b2/camera_front_435i/realsense_front_435i/color/image_raw',
 '/b2/camera_front_435i/realsense_front_435i/color/camera_info',
 '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw',
 '/b2/ugv_status','/fix','/tf','/tf_static',
 '/b2/nicla/magnetometer/heading'}
for path in sorted((root/'bags').iterdir()):
 r=bag.SequentialReader();r.open(bag.StorageOptions(uri=str(path)),bag.ConverterOptions('',''))
 for t in r.get_all_topics_and_types():
  if t.name in selected and t.name not in known:
   writer.create_topic(t);known.add(t.name)
 r.set_filter(bag.StorageFilter(topics=sorted(selected)))
 readers.append(r)
queue=[]
for i,r in enumerate(readers):
 if r.has_next():
  topic,data,stamp=r.read_next();heapq.heappush(queue,(stamp,i,topic,data))
counts={}
while queue:
 stamp,i,topic,data=heapq.heappop(queue)
 writer.write(topic,data,stamp);counts[topic]=counts.get(topic,0)+1
 if readers[i].has_next():
  topic,data,stamp=readers[i].read_next();heapq.heappush(queue,(stamp,i,topic,data))
del writer
import json
(root/'merge_counts.json').write_text(json.dumps(counts,indent=2))
print(counts)
