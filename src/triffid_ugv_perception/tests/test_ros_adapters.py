"""Transport smoke tests with ROS message/node doubles; no live ROS/network involved."""
import importlib
import sys
import types
from types import SimpleNamespace as NS
import numpy as np
import pytest

class Header:
    def __init__(self): self.stamp = NS(sec=1,nanosec=2); self.frame_id = ''
class Image:
    def __init__(self): self.header = Header()
class CameraInfo:
    def __init__(self):
        self.header=Header();self.header.frame_id='camera_optical'
        self.width=8;self.height=6;self.k=[4.,0.,4.,0.,4.,3.,0.,0.,1.];self.d=[0.]*5
class Detection3DArray:
    def __init__(self): self.header = Header(); self.detections = []
class Detection3D:
    def __init__(self):
        self.header = Header(); self.results = []; self.id = ''
        self.bbox = NS(center=NS(position=NS(x=0.,y=0.,z=0.)),size=NS(x=0.,y=0.,z=0.))
class Hypothesis:
    def __init__(self): self.hypothesis = NS(class_id='',score=0.)
class Logger:
    def __getattr__(self, name): return lambda *a, **kw: None
class Publisher:
    def __init__(self): self.messages = []
    def publish(self, value): self.messages.append(value)
    def get_subscription_count(self): return 1
class Node:
    overrides = {}
    def __init__(self, name): self.params = {}; self.name = name; self.subscriptions=[]
    def declare_parameter(self, name, value): self.params[name] = self.overrides.get(name,value)
    def get_parameter(self, name): return NS(value=self.params[name])
    def get_logger(self): return Logger()
    def create_publisher(self, *args): return Publisher()
    def create_subscription(self, *args): self.subscriptions.append(args); return args
    def create_timer(self, *args): return args
class CvBridge:
    def imgmsg_to_cv2(self, msg, **kw): return msg.array
    def cv2_to_imgmsg(self, image, **kw):
        msg=Image();msg.array=image;return msg

@pytest.fixture
def ros(monkeypatch, tmp_path):
    modules={}
    def add(name, **attrs):
        module=types.ModuleType(name)
        for key,value in attrs.items(): setattr(module,key,value)
        modules[name]=module;monkeypatch.setitem(sys.modules,name,module)
    add('rclpy');add('rclpy.node',Node=Node)
    add('rclpy.time',Time=NS(from_msg=lambda stamp:(stamp.sec,stamp.nanosec)))
    add('rclpy.qos',QoSProfile=lambda **kw: kw,ReliabilityPolicy=NS(RELIABLE=1,BEST_EFFORT=2),DurabilityPolicy=NS(VOLATILE=1),HistoryPolicy=NS(KEEP_LAST=1))
    add('sensor_msgs');add('sensor_msgs.msg',Image=Image,CameraInfo=CameraInfo,NavSatFix=type('NavSatFix',(),{}))
    class Buffer:
        def lookup_transform(self,*args):
            return NS(transform=NS(translation=NS(x=.4,y=.02,z=.28),rotation=NS(x=-.5,y=.5,z=-.5,w=.5)))
    class Synchronizer:
        def __init__(self,*args,**kwargs): pass
        def registerCallback(self,callback): self.callback=callback
    add('tf2_ros',Buffer=Buffer,TransformListener=lambda *args:None,TransformException=type('TransformException',(Exception,),{}))
    add('message_filters',Subscriber=lambda *args,**kwargs:NS(),ApproximateTimeSynchronizer=Synchronizer)
    add('vision_msgs');add('vision_msgs.msg',Detection3DArray=Detection3DArray,Detection3D=Detection3D,ObjectHypothesisWithPose=Hypothesis)
    add('std_msgs');add('std_msgs.msg',String=type('String',(),{}),Float32=type('Float32',(),{}))
    add('cv_bridge',CvBridge=CvBridge)
    # Remove adapter modules after each test so the doubles never leak into other imports.
    names=['ugv_node','messages','debug','position','geojson_bridge']
    for name in names: monkeypatch.delitem(sys.modules,'triffid_ugv_perception.'+name,raising=False)
    monkeypatch.setattr(Node,'overrides',dict(discovery_db_path=str(tmp_path/"discoveries.sqlite3"),use_dummy_detections=True,tracker_n_init=1,save_debug_frames=False,save_thermal_frames=False,publish_debug_image=False,use_position_api=False,save_to_disk=False,publish_to_api=False))
    yield
    for name in names: sys.modules.pop('triffid_ugv_perception.'+name,None)

def test_node_rgbd_through_geojson(ros):
    from triffid_ugv_perception.ugv_node import UGVPerceptionNode
    from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
    node=UGVPerceptionNode()
    rgb=Image();rgb.array=np.zeros((6,8,3),np.uint8);rgb.encoding='bgr8'
    node.rgb_callback(rgb)
    assert node.pub_det3d.messages == []  # still waiting for depth
    depth=Image();depth.array=np.full((6,8),2000,np.uint16);depth.encoding='16UC1'
    rgb.header.frame_id=depth.header.frame_id='camera_optical'
    node.camera_info_callback(CameraInfo())
    node.synchronized_callback(rgb,depth)
    result=node.pub_det3d.messages[-1]
    assert result.header.frame_id == 'b2/base_link'
    assert len(result.detections) == 1
    assert result.detections[0].bbox.center.position.x == pytest.approx(2.4)
    assert result.detections[0].results[0].hypothesis.class_id == 'water'
    bridge=GeoJSONBridge()
    try:
        bridge.ugv_callback(result)
        import json
        feature=json.loads(bridge.pub_geojson.messages[-1].data)['features'][0]
        assert feature['properties']['local_frame'] is True
        assert feature['geometry'] is None
        assert feature['properties']['discovery_state'] == 'confirming'
    finally: bridge.close()


def test_masa_rich_path_reidentifies_after_session_change_and_advances_empty_frames(ros,monkeypatch):
    import json
    from triffid_ugv_perception import ugv_node
    from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
    import triffid_ugv_perception.tracking.factory as factory
    Node.overrides.update(use_dummy_detections=False,tracker='masa',use_rich_observations=True,
        discovery_embedding_space='auto',async_inference=False,publish_debug_image=True)
    class Detector:
        empty=False
        def metadata(self):return dict(name='yoloe_v8l')
        def detect(self,image):
            self.last_tracking_detections=[] if self.empty else [dict(bbox=[1,1,5,5],class_id=23,class_name='citizen',confidence=.9)]
            return self.last_tracking_detections
    class Tracker:
        def __init__(self,*args,**kwargs):self.engine=NS(checkpoint_hash='a'*64,tracker_source_hash='b'*64);self.calls=0
        def update(self,rows,image,timestamp_ns):
            self.calls+=1
            return [dict(row,track_id=7,appearance=[1.,0.],identity_class_id=-1) for row in rows]
    monkeypatch.setattr(ugv_node,'create_detector',lambda cfg:Detector())
    monkeypatch.setattr(factory,'TrackingPipeline',Tracker)
    node=ugv_node.UGVPerceptionNode();bridge=GeoJSONBridge()
    rgb=Image();rgb.array=np.zeros((6,8,3),np.uint8);rgb.encoding='bgr8'
    depth=Image();depth.array=np.full((6,8),2000,np.uint16);depth.encoding='16UC1'
    rgb.header.frame_id=depth.header.frame_id='camera_optical';node.camera_info_callback(CameraInfo())
    try:
        for stamp in (1,2):
            rgb.header.stamp.sec=depth.header.stamp.sec=stamp
            node.synchronized_callback(rgb,depth);bridge.rich_callback(node.pub_evidence.messages[-1])
        assert bridge.discovery.status()['objects']==1
        oid=bridge.discovery.store.records()[0]['object_id']
        node.session_id='newsession'
        rgb.header.stamp.sec=depth.header.stamp.sec=3
        node.synchronized_callback(rgb,depth);bridge.rich_callback(node.pub_evidence.messages[-1])
        assert bridge.discovery.status()['objects']==1
        observation=json.loads(node.pub_evidence.messages[-1].data)['tracks'][0]
        assert observation['appearance']==[1.,0.] and observation['track_id']=='newsession:7'
        assert json.loads(bridge.pub_geojson.messages[-1].data)['features'][0]['id']==oid
        node.detector.empty=True;rgb.header.stamp.sec=depth.header.stamp.sec=4
        node.synchronized_callback(rgb,depth);bridge.rich_callback(node.pub_evidence.messages[-1])
        assert node.tracker.calls==4
        assert json.loads(node.pub_evidence.messages[-1].data)['tracks']==[]
        assert node.pub_det3d.messages[-1].detections==[]
        assert bridge.discovery.worker is None
    finally:node.close();bridge.close()


def test_ros_projection_rejects_missing_calibration_stale_depth_and_missing_tf(ros):
    from triffid_ugv_perception.ugv_node import UGVPerceptionNode
    from tf2_ros import TransformException
    node=UGVPerceptionNode()
    rgb=Image();rgb.array=np.zeros((6,8,3),np.uint8);rgb.encoding='bgr8'
    depth=Image();depth.array=np.full((6,8),2000,np.uint16);depth.encoding='16UC1'
    rgb.header.frame_id=depth.header.frame_id='camera_optical'
    node.synchronized_callback(rgb,depth)
    assert node.pub_det3d.messages == []
    node.camera_info_callback(CameraInfo())
    depth.header.stamp.nanosec=100_000_002
    node.synchronized_callback(rgb,depth)
    assert node.pub_det3d.messages == []
    depth.header.stamp.nanosec=2
    def unavailable(*args): raise TransformException('No camera mount')
    node.tf_buffer.lookup_transform=unavailable
    node.synchronized_callback(rgb,depth)
    assert node.pub_det3d.messages == []

def test_position_priority_staleness_and_compass(ros,monkeypatch):
    from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
    bridge=GeoJSONBridge();p=bridge.position
    try:
        assert any(s[1]=='/b2/nicla/magnetometer/heading' for s in bridge.subscriptions)
        p.gps_callback(NS(latitude=37.96,longitude=23.70,altitude=100.))
        assert p.snapshot()[-1] == 'fix'
        with p._position_lock:
            p._api_position_valid=True;p._api_lat=38.;p._api_lon=24.;p._select_position_locked()
        assert p.snapshot()[-1] == 'api'
        p.ugv_status_callback(NS(latitude=39.,longitude=25.,gnss_valid=True))
        assert p.snapshot()[-1] == 'ugv_status'
        import triffid_ugv_perception.position as module
        monkeypatch.setattr(module.time,'monotonic',lambda:p._status_received_monotonic+4.)
        p._check_ugv_status_freshness();assert p.snapshot()[-1] == 'api'
        p.heading_callback(NS(data=90.))
        east,north,up=p._body_to_enu(1.,0.,0.,p.robot_heading)
        assert east == pytest.approx(1.) and north == pytest.approx(0.,abs=1e-12)
        p.ugv_status_callback(NS(latitude=0.,longitude=0.,gnss_valid=False))
        assert p.snapshot()[-1] == 'api'
    finally: bridge.close()


def test_discovery_waits_for_fresh_gps_and_selected_heading_topic(ros,monkeypatch):
    import json
    import triffid_ugv_perception.position as position
    from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
    now=[100.];monkeypatch.setattr(position.time,'monotonic',lambda:now[0])
    bridge=GeoJSONBridge();p=bridge.position
    try:
        msg=Detection3DArray();det=Detection3D();det.id='7'
        hypothesis=Hypothesis();hypothesis.hypothesis.class_id='civilian';hypothesis.hypothesis.score=.9
        det.results=[hypothesis];det.bbox.center.position.x=4.;msg.detections=[det]
        bridge.ugv_callback(msg)
        msg.header.stamp.sec=2;bridge.ugv_callback(msg)
        assert bridge.discovery.status()['waiting']==1
        assert bridge.discovery.status()['pending']==0
        p._receive_fix(NS(latitude=37.96,longitude=23.70,altitude=100.,status=NS(status=0),header=msg.header))
        assert not p.upload_location_ready(2*10**9+2)
        p._receive_heading(NS(data=90.))
        assert p.upload_location_ready(2*10**9+2)
        msg.header.stamp.sec=3;bridge.ugv_callback(msg)
        status=bridge.discovery.status()
        assert status['objects']==1 and status['pending']==1
        payload=json.loads(bridge.discovery.store.db.execute('SELECT payload FROM outbox').fetchone()[0])
        assert payload['properties']['location_valid'] is True
        assert payload['geometry']['type']=='Point'
        now[0]=104.
        assert not p.upload_location_ready(3*10**9+2)
        msg.header.stamp.sec=4;bridge.ugv_callback(msg)
        collection=json.loads(bridge.pub_geojson.messages[-1].data)
        assert collection['features'][0]['geometry'] is None
        bridge._periodic_flush()
        assert json.loads(bridge.pub_status.messages[-1].data)['pending']==1
        assert bridge.discovery.worker is None
    finally:bridge.close()
