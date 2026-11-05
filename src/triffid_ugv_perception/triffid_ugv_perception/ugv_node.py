"""UGV ROS transport and orchestration; model/geometry/tracking/output are independent."""
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from .config import (NODE_PARAMETERS, declare_parameters, CAMERA_FRAME,
                     CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)
from .models import create_detector
from .pipeline import localize_detections
from .tracker import ByteTracker
from .debug import DebugPublisher
from .messages import detection_array

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
        self.camera_frame = CAMERA_FRAME
        self.depth_image = None          # latest depth frame (uint16, mm)
        self.depth_stamp = None
        self.tracker = ByteTracker(
            iou_threshold=float(self.tracker_iou),
            iou_threshold_low=float(self.tracker_iou_low),
            conf_threshold_high=float(self.tracker_conf_high),
            max_age=int(self.tracker_max_age),
            n_init=int(self.tracker_n_init),
            pos_gate=float(self.tracker_pos_gate),
        )

        # ── QoS ─────────────────────────────────────────────────────
        # The rosbag records sensor topics as RELIABLE + VOLATILE.
        # We match that exactly so ros2 bag play data flows correctly.
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # ── Subscribers ─────────────────────────────────────────────
        self.sub_rgb = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            sensor_qos,
        )
        self.sub_depth = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            sensor_qos,
        )
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
        self.get_logger().info(
            f'  Tracker: IoU={self.tracker_iou:.2f}, max_age={int(self.tracker_max_age)}'
        )
        if self.use_dummy:
            self.get_logger().warn('*** DUMMY DETECTION MODE — bypassing model inference ***')

    def depth_callback(self, msg: Image):
        """Store latest depth image (16UC1, millimetres)."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def rgb_callback(self, msg: Image):
        """Main processing trigger – runs on every RGB frame."""

        # Guard: need a depth image
        if self.depth_image is None:
            self.get_logger().warn(
                'Waiting for Depth image…', throttle_duration_sec=5.0,
            )
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

        # Check pixel alignment: depth and RGB should share resolution
        depth_h, depth_w = self.depth_image.shape[:2]
        rgb_h, rgb_w = cv_image.shape[:2]
        depth_scale_x = 1.0
        depth_scale_y = 1.0
        if (depth_h, depth_w) != (rgb_h, rgb_w):
            depth_scale_x = float(depth_w) / float(rgb_w)
            depth_scale_y = float(depth_h) / float(rgb_h)
            self.get_logger().warn(
                f'Depth ({depth_w}×{depth_h}) and RGB ({rgb_w}×{rgb_h}) '
                f'resolutions differ — using scaled depth sampling fallback.',
                throttle_duration_sec=10.0,
            )

        # ── Step 1: Model detection on RGB ────────────────────────────
        raw_detections = self.detector.detect(cv_image)
        if not raw_detections:
            self._publish_empty(msg.header.stamp)
            self.debug._publish_debug_overlay(cv_image, [], msg.header)
            return

        detections_3d = localize_detections(
            raw_detections, self.depth_image, cv_image.shape[:2],
            (CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY), self.nms_merge_dist_m,
        )

        # ── Step 6: Track across frames (IoU on RGB bboxes) ─────────
        tracked = self.tracker.update(detections_3d)

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

def main(args=None):
    rclpy.init(args=args)
    node = UGVPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
