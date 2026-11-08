"""ROS adapter for persistent discovery identities and durable PUT/PATCH delivery."""
import json
import math
from dataclasses import replace
from pathlib import Path

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import String

from .config import BRIDGE_PARAMETERS, declare_parameters
from .discovery import DiscoveryService, DiscoverySettings
from .geojson import GeoJSONFormatter
from .ontology import BY_NAME
from .position import PositionProvider
from .storage import _atomic_write_geojson
from .evidence import decode_packet


class GeoJSONBridge(Node):
    def __init__(self):
        super().__init__('geojson_bridge')
        declare_parameters(self,BRIDGE_PARAMETERS)
        value = lambda key:self.get_parameter(key).value
        self.publish_to_api = bool(value('publish_to_api'))
        self.save_to_disk = bool(value('save_to_disk'))
        self.output_dir = Path(value('output_dir'))
        self.target_geojson_stamp = int(value('target_geojson_stamp'))
        self.snapshot_max_objects = int(value('snapshot_max_objects'))
        self.snapshot_max_bytes = int(value('snapshot_max_bytes'))
        if self.snapshot_max_objects < 1 or self.snapshot_max_bytes < 128:
            raise ValueError('Snapshot limits must be positive (bytes >= 128)')
        settings = DiscoverySettings(namespace=value('discovery_namespace'),
            updates_enabled=bool(value('discovery_updates_enabled')),
            update_interval_s=float(value('discovery_update_interval_s')),
            update_distance_m=float(value('discovery_update_distance_m')),
            update_altitude_m=float(value('discovery_update_altitude_m')),
            update_confidence_delta=float(value('discovery_update_confidence_delta')),
            embedding_space=value('discovery_embedding_space'),confirm_frames=int(value('discovery_confirm_frames')),
            max_objects=int(value('discovery_max_objects')),max_pending=int(value('discovery_max_pending')),
            max_pending_bytes=int(value('discovery_max_pending_bytes')),
            max_database_bytes=int(value('discovery_max_database_bytes')),
            flush_interval_s=float(value('discovery_flush_interval_s')))
        self.rich_input=bool(value('use_rich_observations'))
        self.rich_max_bytes=int(value('rich_max_payload_bytes'))
        self._settings=settings
        self._discovery_args=dict(api_enabled=self.publish_to_api,api_url=value('api_url'),
            uniqueness_contract=bool(value('api_uuid_uniqueness_verified')))
        self._discovery_path=value('discovery_db_path')
        self.discovery=None
        if not (self.rich_input and settings.embedding_space=='auto'):
            self._open_discovery(settings.embedding_space)
        self.position = PositionProvider(self)
        self.formatter = GeoJSONFormatter(self.position)
        self.sub_ugv = (self.create_subscription(String,'/ugv/detections/front/tracked_observations',self.rich_callback,1)
            if self.rich_input else self.create_subscription(Detection3DArray,
                '/ugv/detections/front/detections_3d',self.ugv_callback,10))
        self.pub_geojson = self.create_publisher(String,'/ugv/detections/front/geojson',10)
        self.pub_status = self.create_publisher(String,'/ugv/detections/front/discovery_status',10)
        self.create_timer(float(value('publish_period_s')),self._periodic_flush)
        mode = 'PUT/PATCH' if settings.updates_enabled else 'create-only'
        self.get_logger().info(f'Persistent {mode} discovery ready; API '+('enabled' if self.publish_to_api else 'disabled'))

    def _open_discovery(self,space):
        self.discovery=DiscoveryService(self._discovery_path,replace(self._settings,embedding_space=space),**self._discovery_args)

    def rich_callback(self,msg):
        try:
            packet=decode_packet(msg.data,self.rich_max_bytes)
            expected=self.get_parameter('rich_target_frame').value
            if packet['frame_id']!=expected:raise ValueError('Unexpected rich observation coordinate frame')
            space=packet['embedding_space']
            if self.discovery is None:self._open_discovery(space)
            if self.discovery.settings.embedding_space!=space:raise ValueError('Appearance space changed; use a separate registry namespace/database')
            tracks=packet['tracks'];detections=[]
            for t in tracks:
                x,y,z=t['position']
                detections.append(dict(coordinates=self.position.body_to_gps(x,y,z),position=(x,y,z),size=t['extent'],
                    class_name=t['class_name'],confidence=t['confidence'],track_id=t['track_id']))
            self._process_tracks(tracks,detections,packet['timestamp_ns'])
        except (ValueError,KeyError,TypeError) as error:
            self.get_logger().error(f'Rich observation rejected: {str(error)[:256]}')

    def ugv_callback(self, msg):
        timestamp = int(msg.header.stamp.sec)*10**9+int(msg.header.stamp.nanosec)
        detections, tracks = [], []
        for det in msg.detections:
            if not det.results or not str(det.id):
                continue
            x,y,z = det.bbox.center.position.x,det.bbox.center.position.y,det.bbox.center.position.z
            if not all(math.isfinite(v) for v in (x,y,z,det.results[0].hypothesis.score,
                    det.bbox.size.x,det.bbox.size.y,det.bbox.size.z)):
                continue
            name = det.results[0].hypothesis.class_id
            confidence = det.results[0].hypothesis.score
            item = BY_NAME.get(str(name).lower())
            tracks.append(dict(track_id=str(det.id),class_name=name,class_id=item.id if item else None,
                               confidence=float(confidence)))
            detections.append(dict(coordinates=self.position.body_to_gps(x,y,z),position=(x,y,z),
                size=(det.bbox.size.x,det.bbox.size.y,det.bbox.size.z),class_name=name,
                confidence=float(confidence),track_id=str(det.id)))
        self._process_tracks(tracks,detections,timestamp)

    def _process_tracks(self,tracks,detections,timestamp):
        ready=self.position.upload_location_ready(timestamp)
        features = self.formatter.detections_to_geojson(detections,source='ugv',detection_type='seg')['features']
        for feature in features:
            feature['properties'].update(location_valid=ready,heading_valid=self.position.heading_valid,
                                         location_quality='estimated' if ready else 'waiting_for_fresh_pose')
        # Empty frames must reach the registry to reset unconfirmed tracks.
        result, collection = self.discovery.process(tracks,features,timestamp)
        output = String()
        output.data = json.dumps(collection,allow_nan=False)
        self.pub_geojson.publish(output)
        if self.save_to_disk and self.target_geojson_stamp and timestamp==self.target_geojson_stamp:
            path = self.output_dir/f'ugv_detections_{timestamp//10**9}_{timestamp%10**9:09d}.geojson'
            self._save_snapshot(path,collection)

    def _periodic_flush(self):
        if self.discovery is None:return
        try:
            self.discovery.registry.flush()
            self.discovery.store.promote_ready()
            self.discovery.store.housekeeping()
        except Exception as error:
            self.get_logger().warn(f'Discovery refresh paused: {str(error)[:256]}',throttle_duration_sec=10.)
        status = self.discovery.status()
        message = String()
        message.data = json.dumps(status)
        self.pub_status.publish(message)
        if status['paused_reason']:
            self.get_logger().warn(status['paused_reason'],throttle_duration_sec=10.)
        if self.save_to_disk:
            collection = self.discovery.store.snapshot(self.snapshot_max_objects,self.snapshot_max_bytes)
            self._save_snapshot(self.output_dir/'ugv_detections.geojson',collection)

    def _save_snapshot(self, path, collection):
        try:
            _atomic_write_geojson(path,collection,max_bytes=self.snapshot_max_bytes)
        except (OSError,ValueError) as error:
            self.get_logger().warn(f'Optional snapshot skipped: {str(error)[:256]}',throttle_duration_sec=10.)

    def close(self):
        self.position.close()
        if self.discovery is not None:self.discovery.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GeoJSONBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
