"""Real ROS + GPU integration replay. API off; temporal accuracy is not measured."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import TransformStamped
from triffid_ugv_perception.ugv_node import UGVPerceptionNode
from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
from triffid_ugv_perception.evidence import decode_packet

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=20)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    settings=dict(backend='yoloe',tracker='masa',async_inference=False,
        support_bank='/ws/ugv_supports/PUI',image_size=640,
        use_rich_observations=True,discovery_embedding_space='auto',
        discovery_namespace='PUI/live-validation/front',discovery_db_path=str(out/'discoveries.sqlite3'),
        publish_to_api=False,use_position_api=False,save_to_disk=False,save_debug_frames=False,
        save_thermal_frames=False,publish_debug_image=False,
        rgb_image_topic='/b2/camera_front_435i/realsense_front_435i/color/image_raw',
        depth_image_topic='/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw')
    rosargs=['--ros-args']
    for key,value in settings.items():rosargs+=['-p',f'{key}:={str(value).lower() if isinstance(value,bool) else value}']
    rclpy.init(args=rosargs)
    start=time.perf_counter();node=UGVPerceptionNode();bridge=GeoJSONBridge()
    preparation=time.perf_counter()-start
    executor=SingleThreadedExecutor();executor.add_node(node);executor.add_node(bridge)
    packets=[];collections=[];outputs=[]
    class Capture:
        def __init__(self,original,rows,convert=lambda x:x):self.original=original;self.rows=rows;self.convert=convert
        def publish(self,msg):self.original.publish(msg);self.rows.append(self.convert(msg))
        def get_subscription_count(self):return self.original.get_subscription_count()
    node.pub_evidence=Capture(node.pub_evidence,packets,lambda m:decode_packet(m.data))
    node.pub_det3d=Capture(node.pub_det3d,outputs)
    bridge.pub_geojson=Capture(bridge.pub_geojson,collections,lambda m:json.loads(m.data))
    coordinate=json.loads(Path('/ws/data/PUI/scenarios/coordinate_validation.json').read_text())['recordings']['scenario_1']
    for record in coordinate['robot']['static_transforms']:
        t=TransformStamped();t.header.frame_id=record['parent'];t.child_frame_id=record['child']
        t.transform.translation.x,t.transform.translation.y,t.transform.translation.z=map(float,record['translation_m'])
        t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w=map(float,record['quaternion_xyzw'])
        node.tf_buffer.set_transform_static(t,'recorded_rosbag')
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri='/ws/data/PUI/scenarios/scenario_1/paired_bag'),rosbag2_py.ConverterOptions('',''))
    color='/b2/camera_front_435i/realsense_front_435i/color/'
    depth='/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw'
    reader.set_filter(rosbag2_py.StorageFilter(topics=[color+'image_raw',color+'camera_info',depth]))
    # Wait for the actual DDS publisher/subscriber handshake before injecting frames.
    deadline=time.monotonic()+10
    while not node.pub_evidence.get_subscription_count() and time.monotonic()<deadline:executor.spin_once(timeout_sec=.05)
    if not node.pub_evidence.get_subscription_count():raise RuntimeError('Rich observation ROS subscriber did not connect')
    rows=[];first=None;bucket=-1;rgb_selected=None;last_pair=None;latest_depth=None
    try:
        while reader.has_next() and len(rows)<args.frames:
            topic,raw,_=reader.read_next()
            if topic==color+'camera_info':node.camera_info_callback(deserialize_message(raw,CameraInfo));continue
            msg=deserialize_message(raw,Image);stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            if topic==color+'image_raw':
                if first is None:first=stamp
                current=(stamp-first)//10**9
                if current<=bucket:continue
                bucket=current;rgb_selected=msg
            else:
                latest_depth=msg
            if rgb_selected is None or latest_depth is None:continue
            msg=latest_depth
            stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            rgb_stamp=rgb_selected.header.stamp.sec*10**9+rgb_selected.header.stamp.nanosec
            if abs(stamp-rgb_stamp)>50_000_000:continue
            before=len(collections);started=time.perf_counter()
            node.synchronized_callback(rgb_selected,msg)
            deadline=time.monotonic()+5
            while len(collections)==before and time.monotonic()<deadline:executor.spin_once(timeout_sec=.01)
            if len(collections)==before:raise RuntimeError('No registry output through the actual ROS evidence topic')
            elapsed=time.perf_counter()-started
            assert packets[-1]['embedding_space']==node.embedding_space
            assert packets[-1]['timestamp_ns']==rgb_stamp
            for track in packets[-1]['tracks']:
                assert track['track_id'].startswith(node.session_id+':')
                assert len(track['appearance'])==256
                assert np.isfinite(track['position']).all()
            rows.append(dict(timestamp_ns=rgb_stamp,total_processing_and_ros_s=elapsed,localized_tracks=len(packets[-1]['tracks']),
                registry_objects=bridge.discovery.status()['objects']))
            print(json.dumps(rows[-1]),flush=True)
            last_pair=(rgb_selected,msg);rgb_selected=None;latest_depth=None
        assert len(rows)==args.frames
        # Empty-frame regression: deliberately bypass just the detector; MASA,
        # projection, DDS transport and registry still execute their live code.
        calls=node.tracker.calls
        original=node.detector
        class EmptyDetector:
            last_tracking_detections=[]
            def detect(self,image):return []
        node.detector=EmptyDetector()
        rgb,dep=last_pair
        next_stamp=rows[-1]['timestamp_ns']+61*10**9
        for msg in (rgb,dep):msg.header.stamp.sec=next_stamp//10**9;msg.header.stamp.nanosec=next_stamp%10**9
        before=len(collections);node.synchronized_callback(rgb,dep)
        deadline=time.monotonic()+5
        while len(collections)==before and time.monotonic()<deadline:executor.spin_once(timeout_sec=.01)
        assert node.tracker.calls==calls+1 and not packets[-1]['tracks']
        assert not node.tracker.engine.model.tracker.tracks
        assert len(collections)>before
        node.detector=original
        bridge.discovery.registry.flush()
        records=bridge.discovery.store.records()
        assert records and all(r['appearance'] and r['bbox'] for r in records)
        times=np.array([r['total_processing_and_ros_s'] for r in rows[1:]])
        result=dict(test='YOLOE-v8-L + MASA-R50 + calibrated RGB-D + actual ROS evidence transport + persistent re-ID registry',
            frames=len(rows),preparation_s=preparation,rows=rows,
            warm_mean_s=float(times.mean()),warm_p95_s=float(np.percentile(times,95)),
            warm_processing_capacity_fps=float(1/times.mean()),warm_frames_over_one_second=int((times>1).sum()),
            detector=node.detector.metadata(),tracker=node.tracker.metadata(),embedding_space=node.embedding_space,
            registry_objects=len(records),all_registry_records_have_appearance_and_bbox=True,
            empty_frame_ages_masa_and_reaches_registry=True,api_enabled=False,
            limitations=['Unpaced replay with recorded calibration/static TF; not a live-camera/network latency test.',
                'No annotated temporal IDs, so no re-ID accuracy/HOTA/IDF1 claim.',
                'No injected valid GPS heading; API remains disabled and registry features are unlocated.'])
        (out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ['frames','warm_mean_s','warm_p95_s','warm_processing_capacity_fps','registry_objects']},indent=2),flush=True)
    finally:
        executor.remove_node(node);executor.remove_node(bridge);executor.shutdown()
        node.close();bridge.close();node.destroy_node();bridge.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
