"""Localize the two scenario 2 test images; never borrow noncontemporaneous GPS/heading."""
from bisect import bisect_left
import json
from pathlib import Path
import sqlite3

import numpy as np
from cv_bridge import CvBridge
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage

from triffid_ugv_perception.calibration import calibrated_intrinsics, resolve_static_transform
from triffid_ugv_perception.pipeline import localize_detections

ROOT=Path('/ws/archive/samples_legacy_20261106/scenario2_pair_20261106')
COLOR='/b2/camera_front_435i/realsense_front_435i/color/'
DEPTH='/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/'


def timestamp(msg):
    return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec


def main():
    camera=Path('/ws/data/PUI/scenarios/scenario_2/bags/rosbag2_2026_10_20-16_57_36/rosbag2_2026_10_20-16_57_36_0.db3')
    connection=sqlite3.connect(f'file:{camera}?mode=ro',uri=True)
    def load(topic,cls):
        return [deserialize_message(raw,cls) for raw, in connection.execute(
            'select data from messages where topic_id in (select id from topics where name=?) order by timestamp',(topic,))]
    depths=load(DEPTH+'image_raw',Image)
    infos=load(COLOR+'camera_info',CameraInfo)
    connection.close()
    transforms={}
    source=Path('/bags/scenario_2/rosbag2_2026_10_20-16_37_26')
    for path in sorted(source.glob('*.mcap')):
        with path.open('rb') as handle:
            for _,_,message in make_reader(handle).iter_messages(topics=['/tf_static','/b2/tf_static']):
                for t in deserialize_message(message.data,TFMessage).transforms:
                    v,q=t.transform.translation,t.transform.rotation
                    transforms[(t.header.frame_id,t.child_frame_id)]=dict(
                        parent=t.header.frame_id,child=t.child_frame_id,
                        translation_m=[v.x,v.y,v.z],quaternion_xyzw=[q.x,q.y,q.z,q.w])
        try:
            resolve_static_transform(transforms.values(),'b2/base_link','b2/rs_front_color_optical_frame')
            break
        except ValueError:
            continue
    rows=[json.loads(x) for x in (ROOT/'pet_masa/tracks.jsonl').read_text().splitlines()]
    bridge=CvBridge();results=[]
    for row in rows:
        source_row=row['source'];t=source_row['header_timestamp_ns']
        depth=min(depths,key=lambda x:abs(timestamp(x)-t))
        info=min(infos,key=lambda x:abs(timestamp(x)-t))
        delta=abs(timestamp(depth)-t)/1e6
        if delta>50:raise ValueError('No synchronized depth within 50 ms')
        if depth.encoding not in ('16UC1','mono16') or depth.header.frame_id!=source_row['frame_id'] or info.header.frame_id!=source_row['frame_id']:
            raise ValueError('Unaligned/unsupported depth')
        image=bridge.imgmsg_to_cv2(depth,desired_encoding='passthrough')
        if image.shape!=(source_row['height'],source_row['width']):
            raise ValueError('Depth and RGB must share the same pixel grid')
        intrinsics=calibrated_intrinsics(info,image.shape)
        mount=resolve_static_transform(transforms.values(),'b2/base_link',source_row['frame_id'])
        tracks=[]
        for track in row['tracks']:
            localized=localize_detections([track],image,image.shape,intrinsics,
                                          merge_dist_m=0.,camera_to_target=mount)
            item=dict(track_id=track['track_id'],object_id=track['object_id'],
                      class_name=track['class_name'],confidence=track['confidence'],bbox=track['bbox'],
                      position_robot_m=None,world_coordinates=None)
            if localized:
                det=localized[0]
                item.update(position_robot_m=det['position'],depth_samples=det['n_depth_pts'])
            tracks.append(item)
        results.append(dict(frame=row['frame'],depth_delta_ms=delta,
                            intrinsics=intrinsics,camera_to_robot_matrix=mount.tolist(),tracks=tracks))
    output=dict(frame='b2/base_link',axes='x forward, y left, z up; metres',
        heading_topic='/b2/nicla/magnetometer/heading',heading_estimation=False,
        world_projection=False,reason='No timestamp-overlapping GPS/heading for these images',
        mount_note='Recorded static mount from the scenario 2 NUC bag, assuming the rigid mount was unchanged between recordings',
        saturated_depth_rejected=True,frames=results)
    (ROOT/'pet_masa/coordinates_robot.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(frames=len(results),localizations=[sum(t['position_robot_m'] is not None for t in r['tracks']) for r in results],world_projection=False)),flush=True)


if __name__=='__main__':main()
