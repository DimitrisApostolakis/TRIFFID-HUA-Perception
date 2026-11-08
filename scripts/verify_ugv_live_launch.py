"""Feed real recorded RGB-D over DDS to the separate live launch processes."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
import rosbag2_py
from triffid_ugv_perception.config import NODE_PARAMETERS
from triffid_ugv_perception.evidence import decode_packet

def stamp(msg):return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri='/ws/data/PUI/scenarios/scenario_1/paired_bag'),rosbag2_py.ConverterOptions('',''))
    color='/b2/camera_front_435i/realsense_front_435i/color/'
    depth='/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw'
    reader.set_filter(rosbag2_py.StorageFilter(topics=[color+'image_raw',color+'camera_info',depth]))
    rq=deque(maxlen=5);dq=deque(maxlen=5);pairs=[];info=None;last=-1
    while reader.has_next() and len(pairs)<90:
        topic,raw,_=reader.read_next()
        if topic==color+'camera_info':info=deserialize_message(raw,CameraInfo);continue
        msg=deserialize_message(raw,Image)
        (rq if topic==color+'image_raw' else dq).append(msg)
        if not rq or not dq:continue
        candidates=[(abs(stamp(a)-stamp(b)),i,j) for i,a in enumerate(rq) for j,b in enumerate(dq) if stamp(a)>last]
        if not candidates:continue
        delta,i,j=min(candidates)
        if delta>50_000_000:continue
        a,b=rq[i],dq[j];pairs.append((a,b));last=stamp(a)
        rq.remove(a);dq.remove(b)
    assert len(pairs)==90 and info is not None
    rclpy.init();node=rclpy.create_node('ugv_live_replay_input')
    rgb=node.create_publisher(Image,NODE_PARAMETERS['rgb_image_topic'],5)
    dep=node.create_publisher(Image,NODE_PARAMETERS['depth_image_topic'],5)
    ci=node.create_publisher(CameraInfo,color+'camera_info',1)
    tf=node.create_publisher(TFMessage,'/tf_static',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
    statuses=[];packets=[];collections=[]
    node.create_subscription(String,'/ugv/detections/front/pipeline_status',lambda m:statuses.append((time.monotonic(),json.loads(m.data))),50)
    node.create_subscription(String,'/ugv/detections/front/tracked_observations',lambda m:packets.append(decode_packet(m.data)),50)
    node.create_subscription(String,'/ugv/detections/front/geojson',lambda m:collections.append(json.loads(m.data)),50)
    transforms=[]
    audit=json.loads(Path('/ws/data/PUI/scenarios/coordinate_validation.json').read_text())['recordings']['scenario_1']
    for record in audit['robot']['static_transforms']:
        t=TransformStamped();t.header.frame_id=record['parent'];t.child_frame_id=record['child']
        t.transform.translation.x,t.transform.translation.y,t.transform.translation.z=map(float,record['translation_m'])
        t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w=map(float,record['quaternion_xyzw'])
        transforms.append(t)
    tfmsg=TFMessage();tfmsg.transforms=transforms
    deadline=time.monotonic()+240
    try:
        while not (rgb.get_subscription_count() and dep.get_subscription_count()) and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        assert rgb.get_subscription_count() and dep.get_subscription_count(),'Live node did not become ready'
        for _ in range(15):
            ci.publish(info);tf.publish(tfmsg);rclpy.spin_once(node,timeout_sec=.1)
        start=time.monotonic()
        for index,(a,b) in enumerate(pairs):
            rgb.publish(a);dep.publish(b)
            if index%8==0:ci.publish(info)
            until=start+(index+1)/8.
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=min(.01,max(0.,until-time.monotonic())))
        until=time.monotonic()+3
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
        assert len(statuses)>=8 and packets and collections
        latest=statuses[-1][1];assert latest['queue']['failed']==0
        assert latest['queue']['received']>=70 and latest['queue']['replaced']>0
        assert max(s['queue']['pending'] for _,s in statuses)<=1
        warm=np.asarray([s['processing_s'] for _,s in statuses[1:]])
        assert np.percentile(warm,95)<1.,'Live processing exceeded the one-second budget'
        observed=[t for t,p in statuses[1:]]
        fps=(len(observed)-1)/(observed[-1]-observed[0])
        assert fps>=1.,'Observed warm publication rate is below 1 frame/s'
        result=dict(test='Separate ROS launch processes, real DDS RGB-D input at 8 Hz, asynchronous newest-frame inference, MASA appearance transport and registry',
            input_pairs=90,outputs=len(statuses),observed_warm_output_fps=fps,
            inference_period_s=latest.get('inference_period_s'),
            warm_processing_mean_s=float(warm.mean()),warm_processing_p95_s=float(np.percentile(warm,95)),
            queue=latest['queue'],inferred_frames=latest['inferred_frames'],
            rich_messages=len(packets),geojson_messages=len(collections),
            appearance_dimensions=sorted({len(t['appearance']) for p in packets for t in p['tracks']}),
            session_ids=sorted({p['session_id'] for p in packets}),api_enabled=False,
            limitations=['Recorded-image DDS replay, not a connected physical camera or calibrated geographic accuracy test.',
                'No temporal identity ground truth; no re-ID accuracy claim.'])
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)
    finally:node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
