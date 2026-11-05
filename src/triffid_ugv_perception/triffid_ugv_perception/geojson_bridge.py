"""ROS bridge: subscriptions, publication, periodic snapshots and bounded upload queue."""
import json
import queue
import threading
from pathlib import Path
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import String
from .config import BRIDGE_PARAMETERS, declare_parameters
from .position import PositionProvider
from .geojson import GeoJSONFormatter
from .delivery import FeatureDelivery
from .storage import _atomic_write_geojson
from .spatial import _deduplicate_features, _merge_features

class GeoJSONBridge(Node):
    def __init__(self):
        super().__init__('geojson_bridge')

        declare_parameters(self, BRIDGE_PARAMETERS)

        self.api_url = self.get_parameter('api_url').value
        self.publish_to_api = self.get_parameter('publish_to_api').value
        self.save_to_disk = self.get_parameter('save_to_disk').value
        self.publish_period_s = float(self.get_parameter('publish_period_s').value)
        self.output_dir = self.get_parameter('output_dir').value
        self.target_geojson_stamp = int(
            self.get_parameter('target_geojson_stamp').value
        )

        self.position = PositionProvider(self)
        self.formatter = GeoJSONFormatter(self.position)

        # Subscribers 
        self.sub_ugv = self.create_subscription(
            Detection3DArray,
            '/ugv/detections/front/detections_3d',
            self.ugv_callback,
            10,
        )
        # Publishers
        self.pub_geojson = self.create_publisher(
            String,
            '/ugv/detections/front/geojson',
            10,
        )

        outputs = []
        if self.save_to_disk:
            outputs.append(f'snapshot to {self.output_dir}/ugv_detections.geojson')
        if self.publish_to_api:
            outputs.append(f'PUT to {self.api_url}')
        self.get_logger().info('GeoJSON Bridge started.')
        self.get_logger().info(
            f'  Every {self.publish_period_s}s: '
            + (' AND '.join(outputs) if outputs
               else 'nothing (both save_to_disk and publish_to_api are off)')
        )
        # Spatial deduplication radius (metres)
        self._dedup_radius_m = float(self.get_parameter('dedup_radius_m').value)

        # API uploads run on one worker thread; depth-1 queue drops a
        # stale pending snapshot instead of piling up threads.
        self._accumulated = []
        self.delivery = FeatureDelivery(self.api_url, self._dedup_radius_m, self.get_logger())
        self._api_queue = queue.Queue(maxsize=1)
        self._api_worker = None
        if self.publish_to_api:
            self._api_worker = threading.Thread(
                target=self._api_worker_loop, daemon=True,
            )
            self._api_worker.start()
        self.create_timer(self.publish_period_s, self._periodic_flush)

    def ugv_callback(self, msg: Detection3DArray):
        """Process UGV 3D detections (b2/base_link) → GeoJSON."""
        detections = []
        for det in msg.detections:
            # 3D position in b2/base_link (X=fwd, Y=left, Z=up)
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            z = det.bbox.center.position.z

            # 3D bbox extent (metres)
            sx = det.bbox.size.x
            sy = det.bbox.size.y
            sz = det.bbox.size.z

            # Convert body-frame centre to GPS [lon, lat, alt]
            lon, lat, alt = self.position.body_to_gps(x, y, z)

            # Extract class and confidence
            class_name = 'unknown'
            confidence = 0.0
            if det.results:
                class_name = det.results[0].hypothesis.class_id
                confidence = det.results[0].hypothesis.score

            detections.append({
                'coordinates': (lon, lat, alt),
                'size': (sx, sy, sz),
                'position': (x, y, z),
                'class_name': class_name,
                'confidence': confidence,
                'track_id': det.id,
            })

        if not detections:
            return

        if not self.position.gps_valid:
            self.get_logger().warn(
                'No UGV global position — publishing with local_frame=true '
                '(body-frame coords).',
                throttle_duration_sec=5.0,
            )

        geojson = self.formatter.detections_to_geojson(
            detections, source='ugv', detection_type='seg',
        )
        geojson['features'] = _deduplicate_features(
            geojson['features'], self._dedup_radius_m
        )

        stamp_key = (
            f'{msg.header.stamp.sec}_{msg.header.stamp.nanosec:09d}'
        )
        stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )
        if self.target_geojson_stamp and stamp_ns == self.target_geojson_stamp:
            target_geojson = {
                "type": "FeatureCollection",
                "frame_stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec),
                },
                "features": [],
            }
            for feature in geojson['features']:
                copied = json.loads(json.dumps(feature))
                copied.setdefault('properties', {})['frame_stamp'] = stamp_key
                target_geojson['features'].append(copied)

            gps_valid, robot_lat, robot_lon, robot_alt, position_source = self.position.snapshot()

            if gps_valid:
                target_geojson['features'].append({
                    "type": "Feature",
                    "id": "ugv",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [robot_lon, robot_lat],
                    },
                    "properties": {
                        "class": "ugv",
                        "source": "ugv_pose",
                        "position_source": position_source,
                        "local_frame": False,
                        "altitude_m": round(float(robot_alt), 2),
                        "frame_stamp": stamp_key,
                        "marker-color": "#000000",
                        "marker-size": "large",
                        "marker-symbol": "car",
                    },
                })

            target_path = (
                Path(self.output_dir)
                / f'ugv_detections_{stamp_key}.geojson'
            )
            try:
                _atomic_write_geojson(target_path, target_geojson)
                self.get_logger().info(
                    f'Target-frame GeoJSON: '
                    f'{len(target_geojson["features"])} features → '
                    f'{target_path}'
                )
            except OSError as e:
                self.get_logger().warn(
                    f'Target-frame GeoJSON write failed: {e}'
                )

        self._accumulated = _merge_features(
            self._accumulated, geojson['features'], self._dedup_radius_m,
        )
        self._publish(geojson)

    def _publish(self, geojson: dict):
        """Publish GeoJSON to the ROS2 topic (per frame)."""
        msg = String()
        msg.data = json.dumps(geojson, indent=2)
        self.pub_geojson.publish(msg)

        self.get_logger().info(
            f'Published GeoJSON with {len(geojson["features"])} features',
            throttle_duration_sec=1.0,
        )

    def _periodic_flush(self):
        """Flush the accumulated feature set: disk snapshot and/or API
        upload, each independently controlled by its parameter."""
        if not self._accumulated:
            return
        collection = {
            "type": "FeatureCollection",
            "features": list(self._accumulated),
        }
        if self.save_to_disk:
            path = Path(self.output_dir) / 'ugv_detections.geojson'
            try:
                _atomic_write_geojson(path, collection)
                self.get_logger().info(
                    f'Snapshot: {len(collection["features"])} features → {path}',
                    throttle_duration_sec=30.0,
                )
            except OSError as e:
                self.get_logger().warn(
                    f'GeoJSON snapshot write failed: {e}',
                    throttle_duration_sec=30.0,
                )
        if self.publish_to_api:
            # Drop the stale queued snapshot (if any) — the newest one
            # contains everything the older one did.
            try:
                self._api_queue.put_nowait(collection)
            except queue.Full:
                try:
                    self._api_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._api_queue.put_nowait(collection)
                except queue.Full:
                    pass

    def _api_worker_loop(self):
        """Single long-lived uploader; a None item is the shutdown sentinel."""
        while True:
            collection = self._api_queue.get()
            if collection is None:
                return
            self.delivery._send_to_api(collection)

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
            node.position.close()
            if node._api_worker is not None:
                # Best-effort worker stop: clear any queued snapshot and
                # deliver the shutdown sentinel (worker is a daemon, so a
                # full queue at exit is not fatal).
                try:
                    node._api_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    node._api_queue.put_nowait(None)
                except queue.Full:
                    pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
