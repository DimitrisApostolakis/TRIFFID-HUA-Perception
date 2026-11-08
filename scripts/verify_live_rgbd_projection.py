"""Exercise ROS synchronization/TF with real recorded images and a dummy detector.

This verifies live projection plumbing, not PET/MASA accuracy or throughput.
No external API or saved dummy detection overlays are used.
"""
import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message
import rosbag2_py as bag
from sensor_msgs.msg import Image, CameraInfo

from triffid_ugv_perception.ugv_node import UGVPerceptionNode
from triffid_ugv_perception.calibration import resolve_static_transform

ROOT = Path('/ws/data/PUI/scenarios')
COLOR = '/b2/camera_front_435i/realsense_front_435i/color/'
DEPTH = '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/'


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_dummy_detections:=true',
        '-p', 'save_debug_frames:=false', '-p', 'save_thermal_frames:=false',
        '-p', 'publish_debug_image:=false', '-p', 'tracker_n_init:=1',
        '-p', f'rgb_image_topic:={COLOR}image_raw', '-p', f'depth_image_topic:={DEPTH}image_raw'])
    node = UGVPerceptionNode()
    outputs = []
    class Capture:
        def publish(self, msg): outputs.append(msg)
    node.pub_det3d = Capture()
    deltas = []
    def observe(rgb, depth):
        time_ns=lambda x:x.header.stamp.sec*10**9+x.header.stamp.nanosec
        deltas.append(abs(time_ns(rgb)-time_ns(depth))/1e6)
    node.rgbd_sync.registerCallback(observe)
    audit = json.loads((ROOT/'coordinate_validation.json').read_text())['recordings']['scenario_1']
    for record in audit['robot']['static_transforms']:
        transform=TransformStamped()
        transform.header.frame_id=record['parent'];transform.child_frame_id=record['child']
        transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z=record['translation_m']
        transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z,transform.transform.rotation.w=record['quaternion_xyzw']
        node.tf_buffer.set_transform_static(transform,'recorded_rosbag')
    expected=resolve_static_transform(audit['robot']['static_transforms'],'b2/base_link','b2/rs_front_color_optical_frame')
    actual=node.tf_buffer.lookup_transform('b2/base_link','b2/rs_front_color_optical_frame',rclpy.time.Time())
    np.testing.assert_allclose([actual.transform.translation.x,actual.transform.translation.y,actual.transform.translation.z],expected[:3,3],atol=1e-9)
    reader=bag.SequentialReader()
    reader.open(bag.StorageOptions(uri=str(ROOT/'scenario_1/paired_bag')),bag.ConverterOptions('',''))
    reader.set_filter(bag.StorageFilter(topics=[COLOR+'camera_info',COLOR+'image_raw',DEPTH+'image_raw']))
    rgb_messages=depth_messages=0
    try:
        while reader.has_next() and len(outputs)<12:
            topic,raw,_=reader.read_next()
            if topic==COLOR+'camera_info':
                node.camera_info_callback(deserialize_message(raw,CameraInfo))
            elif topic==COLOR+'image_raw':
                rgb_messages+=1;node.sub_rgb.signalMessage(deserialize_message(raw,Image))
            else:
                depth_messages+=1;node.sub_depth.signalMessage(deserialize_message(raw,Image))
        assert len(outputs)>=12 and all(x.header.frame_id=='b2/base_link' for x in outputs)
        assert deltas and max(deltas)<=50.
        assert node.camera_info.width==1280 and node.camera_info.height==720
        positions=[[d.bbox.center.position.x,d.bbox.center.position.y,d.bbox.center.position.z]
                   for row in outputs for d in row.detections]
        assert positions and np.isfinite(positions).all()
        summary=dict(test='ROS synchronized RGB-D projection using recorded scenario 1 images',
            detector='dummy; not an object-detection accuracy or PET/MASA latency test',
            input_rgb_messages=rgb_messages,input_depth_messages=depth_messages,
            output_messages=len(outputs),max_depth_delta_ms=max(deltas),
            actual_intrinsics=list(node.camera_info.k),camera_mount_translation_m=expected[:3,3].tolist(),
            sample_position_body_m=positions[0],api_enabled=False)
        (ROOT/'live_projection_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
