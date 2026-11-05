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
def ros(monkeypatch):
    modules={}
    def add(name, **attrs):
        module=types.ModuleType(name)
        for key,value in attrs.items(): setattr(module,key,value)
        modules[name]=module;monkeypatch.setitem(sys.modules,name,module)
    add('rclpy');add('rclpy.node',Node=Node)
    add('rclpy.qos',QoSProfile=lambda **kw: kw,ReliabilityPolicy=NS(RELIABLE=1),DurabilityPolicy=NS(VOLATILE=1),HistoryPolicy=NS(KEEP_LAST=1))
    add('sensor_msgs');add('sensor_msgs.msg',Image=Image,NavSatFix=type('NavSatFix',(),{}))
    add('vision_msgs');add('vision_msgs.msg',Detection3DArray=Detection3DArray,Detection3D=Detection3D,ObjectHypothesisWithPose=Hypothesis)
    add('std_msgs');add('std_msgs.msg',String=type('String',(),{}),Float32=type('Float32',(),{}))
    add('cv_bridge',CvBridge=CvBridge)
    # Remove adapter modules after each test so the doubles never leak into other imports.
    names=['ugv_node','messages','debug','position','geojson_bridge']
    for name in names: monkeypatch.delitem(sys.modules,'triffid_ugv_perception.'+name,raising=False)
    monkeypatch.setattr(Node,'overrides',dict(use_dummy_detections=True,tracker_n_init=1,save_debug_frames=False,save_thermal_frames=False,publish_debug_image=False,use_position_api=False,save_to_disk=False,publish_to_api=False))
    yield
    for name in names: sys.modules.pop('triffid_ugv_perception.'+name,None)

def test_node_rgbd_through_geojson(ros):
    from triffid_ugv_perception.ugv_node import UGVPerceptionNode
    from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
    node=UGVPerceptionNode()
    rgb=Image();rgb.array=np.zeros((6,8,3),np.uint8);rgb.encoding='bgr8'
    node.rgb_callback(rgb)
    assert node.pub_det3d.messages == []  # still waiting for depth
    depth=Image();depth.array=np.full((6,8),2000,np.uint16)
    node.depth_callback(depth);node.rgb_callback(rgb)
    result=node.pub_det3d.messages[-1]
    assert result.header.frame_id == 'b2/base_link'
    assert len(result.detections) == 1
    assert result.detections[0].bbox.center.position.x == 2.
    assert result.detections[0].results[0].hypothesis.class_id == 'water'
    bridge=GeoJSONBridge()
    try:
        bridge.ugv_callback(result)
        import json
        feature=json.loads(bridge.pub_geojson.messages[-1].data)['features'][0]
        assert feature['properties']['local_frame'] is True
        assert feature['geometry']['type'] == 'Point'
    finally: bridge.position.close()

def test_position_priority_staleness_and_compass(ros,monkeypatch):
    from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
    bridge=GeoJSONBridge();p=bridge.position
    try:
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
    finally: p.close()
