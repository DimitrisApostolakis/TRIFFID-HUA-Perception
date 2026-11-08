"""UGV ROS transport and orchestration; model/geometry/tracking/output are independent."""
import os
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from tf2_ros import Buffer, TransformListener, TransformException
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
from uuid import uuid4
from std_msgs.msg import String
from .config import NODE_PARAMETERS, declare_parameters
from .calibration import calibrated_intrinsics, rigid_matrix
from .models import create_detector
from .pipeline import localize_detections
from .tracker import ByteTracker
from .debug import DebugPublisher
from .messages import detection_array
from .evidence import encode_packet
from .latest_frame import LatestFrameWorker

class UGVPerceptionNode(Node):
    def __init__(self):
        super().__init__('ugv_perception_node')

        declare_parameters(self, NODE_PARAMETERS)

        self.target_frame = self.get_parameter('target_frame').value
        self.use_dummy = self.get_parameter('use_dummy_detections').value
        self.tracker_iou = self.get_parameter('tracker_iou_threshold').value
        self.tracker_iou_low = self.get_parameter('tracker_iou_threshold_low').value
        self.tracker_conf_high = self.get_parameter('tracker_conf_high').value
        self.tracker_max_age = self.get_parameter('tracker_max_age').value
        self.tracker_n_init = self.get_parameter('tracker_n_init').value
        self.tracker_pos_gate = self.get_parameter('tracker_pos_gate').value
        self.nms_merge_dist_m = self.get_parameter('nms_merge_dist_m').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.save_debug_frames = self.get_parameter('save_debug_frames').value
        self.debug_frames_dir = self.get_parameter('debug_frames_dir').value
        self.debug_frame_period_s = self.get_parameter('debug_frame_period_s').value
        self.save_thermal_frames = self.get_parameter('save_thermal_frames').value
        self.thermal_frames_dir = self.get_parameter('thermal_frames_dir').value
        self.thermal_frame_period_s = self.get_parameter('thermal_frame_period_s').value
        self.rgb_topic = self.get_parameter('rgb_image_topic').value
        self.depth_topic = self.get_parameter('depth_image_topic').value
        self.info_topic = (self.get_parameter('camera_info_topic').value or
                           self.rgb_topic.rsplit('/', 1)[0] + '/camera_info')
        self.sync_tolerance = float(self.get_parameter('rgbd_sync_tolerance_s').value)
        if not 0 < self.sync_tolerance <= 0.2:
            raise ValueError('rgbd_sync_tolerance_s must be in (0, 0.2]')
        self.thermal_topic = self.get_parameter('thermal_image_topic').value

        self.detector = create_detector({name: self.get_parameter(name).value for name in NODE_PARAMETERS})
        model_info = self.detector.metadata()
        self.get_logger().info(
            f'Detector: {model_info["name"]}; instance method: '
            f'{model_info.get("instance_method", "backend instances")}'
        )
        if model_info.get('support_bank'):
            bank = model_info['support_bank']
            self.get_logger().info(
                f'UGV bank: {self.get_parameter("support_bank").value}; '
                f'{bank["shots"]} shot(s), manifest SHA256={bank["manifest_sha256"]}'
            )

        # ── State ───────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.debug = DebugPublisher(self, self.bridge)
        if self.save_debug_frames:
            try:
                os.makedirs(self.debug_frames_dir, exist_ok=True)
            except OSError as e:
                self.get_logger().error(
                    f'Cannot create debug frame directory {self.debug_frames_dir}: {e}'
                )
                self.save_debug_frames = False
        if self.save_thermal_frames:
            try:
                os.makedirs(self.thermal_frames_dir, exist_ok=True)
            except OSError as e:
                self.get_logger().error(
                    f'Cannot create thermal frame directory {self.thermal_frames_dir}: {e}'
                )
                self.save_thermal_frames = False
        self.camera_frame = 'waiting for CameraInfo'
        self.camera_info = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.depth_image = None          # latest depth frame (uint16, mm)
        self.depth_stamp = None
        self.masa_enabled=self.get_parameter('tracker').value=='masa' and not self.use_dummy
        if self.get_parameter('tracker').value not in ('masa','bytetrack'):
            raise ValueError('Live tracker must be masa or bytetrack')
        self.session_id=uuid4().hex
        self.embedding_space='none'
        self.last_processed_stamp=None
        self.inferred_frames=0
        if self.masa_enabled:
            from .tracking.factory import TrackingPipeline
            self.tracker=TrackingPipeline('masa',person_roles=bool(self.get_parameter('person_roles').value),
                masa_repo=self.get_parameter('masa_repo').value,masa_checkpoint=self.get_parameter('masa_checkpoint').value,
                memory_seconds=int(self.get_parameter('masa_memory_seconds').value))
            self.embedding_space='MASA-R50:'+self.tracker.engine.checkpoint_hash
            for parameter,attribute in [('masa_expected_checkpoint_sha256','checkpoint_hash'),
                    ('masa_expected_tracker_source_sha256','tracker_source_hash'),('masa_expected_source_revision','revision')]:
                expected=self.get_parameter(parameter).value
                if expected and expected!=getattr(self.tracker.engine,attribute):raise ValueError(f'Live asset fingerprint changed: {parameter}')
        else:
            self.tracker = ByteTracker(
                iou_threshold=float(self.tracker_iou),
                iou_threshold_low=float(self.tracker_iou_low),
                conf_threshold_high=float(self.tracker_conf_high),
                max_age=int(self.tracker_max_age),
                n_init=int(self.tracker_n_init),
                pos_gate=float(self.tracker_pos_gate),
            )

        # ── QoS ─────────────────────────────────────────────────────
        # A best-effort subscriber accepts both sensor-data publishers and
        # the reliable publishers used by the recorded bags.
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # ── Subscribers ─────────────────────────────────────────────
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic,
                                                 self.camera_info_callback, sensor_qos)
        self.sub_rgb = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=sensor_qos)
        self.sub_depth = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        self.rgbd_sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth], queue_size=30, slop=self.sync_tolerance)
        self.rgbd_sync.registerCallback(self.synchronized_callback)
        self.sub_thermal = self.create_subscription(
            Image,
            self.thermal_topic,
            self.debug.thermal_callback,
            sensor_qos,
        )
        # ── Publishers ───────────────────────────────────────────────
        self.pub_det3d = self.create_publisher(
            Detection3DArray,
            '/ugv/detections/front/detections_3d',
            10,
        )
        self.pub_evidence=self.create_publisher(String,'/ugv/detections/front/tracked_observations',1)
        self.pub_pipeline_status=self.create_publisher(String,'/ugv/detections/front/pipeline_status',1)
        self.pub_seg = self.create_publisher(
            Image,
            '/ugv/detections/front/segmentation',
            10,
        )
        self.pub_debug = self.create_publisher(
            Image,
            '/ugv/detections/front/debug_image',
            10,
        )

        self.get_logger().info('UGV Perception node started (pixel-aligned RGB-D pipeline).')
        self.get_logger().info(f'  RGB topic:       {self.rgb_topic}')
        self.get_logger().info(f'  Depth topic:     {self.depth_topic}')
        self.get_logger().info(f'  Thermal topic:   {self.thermal_topic}  (listen/debug only)')
        if self.save_thermal_frames:
            self.get_logger().info(
                f'  Thermal frames:  {self.thermal_frames_dir} '
                f'(debug snapshots, min period {float(self.thermal_frame_period_s):.2f}s)'
            )
        self.get_logger().info(f'  Camera frame:    {self.camera_frame}')
        self.get_logger().info(f'  Output topic:    /ugv/detections/front/detections_3d  (frame: {self.target_frame})')
        self.get_logger().info(f'  Seg topic:       /ugv/detections/front/segmentation  (mono8 label map)')
        self.get_logger().info(f'  Debug topic:     /ugv/detections/front/debug_image  (RGB+ID overlay)')
        if self.save_debug_frames:
            self.get_logger().info(
                f'  Debug frames:    {self.debug_frames_dir} '
                f'(detection frames, min period {float(self.debug_frame_period_s):.2f}s)'
            )
        self.get_logger().info('  Tracker: '+('MASA-R50 appearance association, timestamp-aged gallery' if self.masa_enabled
            else f'ByteTracker IoU={self.tracker_iou:.2f}, max_age={int(self.tracker_max_age)}'))
        if self.use_dummy:
            self.get_logger().warn('*** DUMMY DETECTION MODE — bypassing model inference ***')
        self.worker=None
        if bool(self.get_parameter('async_inference').value) and not self.use_dummy:
            self.worker=LatestFrameWorker(self._process_queued_pair,
                float(self.get_parameter('inference_period_s').value),float(self.get_parameter('max_pending_age_s').value),
                on_error=lambda error:self.get_logger().error(f'Live frame failed: {str(error)[:256]}'))
            self.worker.start()
        self.get_logger().info(f'Live pipeline ready: {model_info["name"]} + {"MASA-R50" if self.masa_enabled else "ByteTracker"}; embedding_space={self.embedding_space}')

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_frame = msg.header.frame_id

    def synchronized_callback(self, rgb, depth):
        rgb_ns = rgb.header.stamp.sec * 10**9 + rgb.header.stamp.nanosec
        depth_ns = depth.header.stamp.sec * 10**9 + depth.header.stamp.nanosec
        if abs(rgb_ns-depth_ns) > self.sync_tolerance * 1e9:
            return
        if depth.encoding not in ('16UC1', 'mono16'):
            self.get_logger().warn('Skipping unsupported depth encoding; millimetre depth required.', throttle_duration_sec=5.)
            return
        if not rgb.header.frame_id or depth.header.frame_id != rgb.header.frame_id:
            self.get_logger().warn('Skipping RGB/depth with different or missing optical frames.', throttle_duration_sec=5.)
            return
        if self.worker is not None:
            self.worker.submit((rgb,depth,self.camera_info))
            return
        self._process_pair(rgb,depth,self.camera_info)

    def _process_queued_pair(self,pair,received):
        started=time.perf_counter()
        self._process_pair(*pair)
        status=String();status.data=json.dumps(dict(schema='triffid.ugv.pipeline-status.v1',
            timestamp_ns=pair[0].header.stamp.sec*10**9+pair[0].header.stamp.nanosec,
            processing_s=time.perf_counter()-started,received_to_completion_s=time.monotonic()-received,
            inference_period_s=self.worker.period,
            inferred_frames=self.inferred_frames,queue=self.worker.stats(),embedding_space=self.embedding_space,session_id=self.session_id),allow_nan=False)
        self.pub_pipeline_status.publish(status)

    def _process_pair(self,rgb,depth,info):
        self.depth_image = None
        self.depth_callback(depth)
        if self.depth_image is not None:
            self.rgb_callback(rgb,camera_info=info)

    def depth_callback(self, msg: Image):
        """Store latest depth image (16UC1, millimetres)."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def rgb_callback(self, msg: Image, camera_info=None):
        """Process a calibrated RGB frame after bounded-time depth matching."""

        # Guard: need a depth image
        if self.depth_image is None:
            self.get_logger().warn(
                'Waiting for Depth image…', throttle_duration_sec=5.0,
            )
            return

        info = self.camera_info if camera_info is None else camera_info
        if info is None or not msg.header.frame_id or info.header.frame_id != msg.header.frame_id:
            self.get_logger().warn('Waiting for CameraInfo in the RGB optical frame.', throttle_duration_sec=5.)
            return

        # Convert RGB image — handle both standard bgr8/rgb8 and YUYV (RealSense 435i)
        try:
            if msg.encoding in ('yuv422', 'yuv422_yuy2', 'YUV422'):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 2)
                )
                cv_image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        # Aligned depth must share the calibrated RGB pixel grid.
        depth_h, depth_w = self.depth_image.shape[:2]
        rgb_h, rgb_w = cv_image.shape[:2]
        if (depth_h, depth_w) != (rgb_h, rgb_w):
            self.get_logger().warn(
                'Skipping depth/RGB resolution mismatch; aligned depth is required.',
                throttle_duration_sec=10.0,
            )
            return
        try:
            intrinsics = calibrated_intrinsics(info, (rgb_h, rgb_w))
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp))
            t, q = transform.transform.translation, transform.transform.rotation
            camera_to_target = rigid_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
        except (ValueError, TransformException) as error:
            self.get_logger().warn(f'Skipping uncalibrated frame: {error}', throttle_duration_sec=5.)
            return

        # ── Step 1: Model detection on RGB ────────────────────────────
        timestamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        if self.last_processed_stamp is not None and timestamp<=self.last_processed_stamp:return
        self.last_processed_stamp=timestamp
        raw_detections = self.detector.detect(cv_image)
        self.inferred_frames+=1
        if self.masa_enabled:
            candidates=getattr(self.detector,'last_tracking_detections',raw_detections)
            tracks_2d=self.tracker.update(candidates,cv_image,timestamp_ns=timestamp)
            for t in tracks_2d:t['track_id']=self.session_id+':'+str(t['track_id'])
            tracked=localize_detections(tracks_2d,self.depth_image,cv_image.shape[:2],intrinsics,
                self.nms_merge_dist_m,camera_to_target=camera_to_target,deduplicate=False)
            evidence=String();evidence.data=encode_packet(tracked,timestamp,self.target_frame,self.embedding_space,self.session_id)
            self.pub_evidence.publish(evidence)
        else:
            detections_3d=localize_detections(raw_detections,self.depth_image,cv_image.shape[:2],intrinsics,
                self.nms_merge_dist_m,camera_to_target=camera_to_target)
            tracked=self.tracker.update(detections_3d)

        # ── Step 7: Publish Detection3DArray ─────────────────────────
        self.pub_det3d.publish(detection_array(tracked, msg.header.stamp, self.target_frame))

        # ── Step 8: Publish segmentation label map (mono8) ───────────
        self.debug._publish_segmentation(cv_image, raw_detections, msg.header)
        self.debug._publish_debug_overlay(cv_image, tracked, msg.header)

        if tracked:
            self.get_logger().info(
                f'Published {len(tracked)} detections '
                f'(IDs: {[t["track_id"] for t in tracked]})',
                throttle_duration_sec=1.0,
            )

    def _publish_empty(self, stamp):
        """Publish an empty Detection3DArray (keeps downstream aware)."""
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        self.pub_det3d.publish(msg)

    def close(self):
        if self.worker is not None:self.worker.close()

def main(args=None):
    rclpy.init(args=args)
    node = UGVPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
