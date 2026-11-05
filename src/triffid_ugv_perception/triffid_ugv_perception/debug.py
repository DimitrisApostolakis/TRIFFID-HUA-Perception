"""Thermal snapshots, semantic labels and track overlays; no inference or localization."""
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image

class DebugPublisher:
    def __init__(self, node, bridge):
        self.node, self.bridge = node, bridge
        self._last_debug_save_stamp = None
        self._last_thermal_save_stamp = None
        self.thermal_frames_received = 0
        self.thermal_stamp = None

    def thermal_callback(self, msg: Image):
        """Confirm FLIR reception and optionally save debug-only snapshots."""
        self.thermal_stamp = msg.header.stamp
        self.thermal_frames_received += 1
        self.node.get_logger().info(
            f'Thermal stream active: received frame #{self.thermal_frames_received} '
            f'({msg.width}x{msg.height}, encoding={msg.encoding})',
            throttle_duration_sec=5.0,
        )

        if not self.node.save_thermal_frames:
            return

        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        period_s = max(0.0, float(self.node.thermal_frame_period_s))
        should_save = (
            self._last_thermal_save_stamp is None
            or period_s == 0.0
            or stamp_s < self._last_thermal_save_stamp
            or (stamp_s - self._last_thermal_save_stamp) >= period_s
        )
        if not should_save:
            return

        try:
            thermal = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Debug snapshots are normalized to viewable 8-bit when the
            # FLIR stream is a higher-bit-depth single-channel image.
            if thermal.ndim == 2 and thermal.dtype != np.uint8:
                finite = np.isfinite(thermal)
                if not np.any(finite):
                    self.node.get_logger().warn(
                        'Thermal frame contains no finite pixels; skipping snapshot.',
                        throttle_duration_sec=5.0,
                    )
                    return
                valid = thermal[finite].astype(np.float32)
                lo = float(np.min(valid))
                hi = float(np.max(valid))
                if hi > lo:
                    debug_thermal = np.clip(
                        (thermal.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                        0.0, 255.0,
                    ).astype(np.uint8)
                else:
                    debug_thermal = np.zeros(thermal.shape, dtype=np.uint8)
            else:
                debug_thermal = thermal

            # ROS rgb8 is RGB, while OpenCV imwrite expects BGR channel order.
            # Convert only for RGB-encoded 3-channel debug frames so the saved
            # PNG preserves the FLIR palette colours correctly.
            encoding = str(msg.encoding).lower()
            if (
                debug_thermal.ndim == 3
                and debug_thermal.shape[2] == 3
                and encoding in ('rgb8', 'rgb16')
            ):
                debug_thermal = cv2.cvtColor(debug_thermal, cv2.COLOR_RGB2BGR)

            filename = (
                f'thermal_{msg.header.stamp.sec}_'
                f'{msg.header.stamp.nanosec:09d}.png'
            )
            path = os.path.join(self.node.thermal_frames_dir, filename)
            if cv2.imwrite(path, debug_thermal):
                self._last_thermal_save_stamp = stamp_s
            else:
                self.node.get_logger().warn(
                    f'Failed to save thermal debug frame: {path}',
                    throttle_duration_sec=5.0,
                )
        except Exception as e:
            self.node.get_logger().warn(
                f'Thermal debug frame conversion/save failed: {e}',
                throttle_duration_sec=5.0,
            )

    def _publish_segmentation(self, cv_image, detections, header):
        """Publish a mono8 label map (pixel = 1-based class id, 0 = background);
        overlapping masks resolve to the higher-confidence detection."""
        if self.node.pub_seg.get_subscription_count() == 0:
            return  # no subscribers, skip

        h, w = cv_image.shape[:2]
        label_img = np.zeros((h, w), dtype=np.uint8)

        # Sort ascending by confidence so higher-conf overwrites lower
        sorted_dets = sorted(detections,
                             key=lambda d: d['confidence'])
        for det in sorted_dets:
            mask = det.get('mask')
            if mask is None:
                continue
            # class_id is 0-based from YOLO; store as 1-based (0 = bg)
            label_img[mask] = det['class_id'] + 1

        seg_msg = self.bridge.cv2_to_imgmsg(label_img, encoding='mono8')
        seg_msg.header = header
        self.node.pub_seg.publish(seg_msg)

    def _publish_debug_overlay(self, cv_image, tracked, header):
        """Publish and optionally save RGB debug image with detection overlay."""
        if not self.node.publish_debug_image and not self.node.save_debug_frames:
            return

        dbg = cv_image.copy()
        for t in tracked:
            bbox = t.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            tid = int(t.get('track_id', -1))
            cls = t.get('class_name', 'unknown')
            conf = float(t.get('confidence', 0.0))
            npts = int(t.get('n_depth_pts', 0))

            # Deterministic per-track color makes ID switches obvious.
            color = (
                int((37 * tid) % 255),
                int((97 * tid) % 255),
                int((173 * tid) % 255),
            )
            cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
            label = f'{cls} #{tid} {conf:.2f} d={npts}'
            ytxt = max(18, y1 - 6)
            cv2.putText(
                dbg, label, (x1, ytxt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

        # Persist only frames that contain at least one tracked detection.
        if self.node.save_debug_frames and tracked:
            stamp_s = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
            period_s = max(0.0, float(self.node.debug_frame_period_s))
            should_save = (
                self._last_debug_save_stamp is None
                or period_s == 0.0
                or stamp_s < self._last_debug_save_stamp
                or (stamp_s - self._last_debug_save_stamp) >= period_s
            )
            if should_save:
                filename = (
                    f'detection_{header.stamp.sec}_'
                    f'{header.stamp.nanosec:09d}.png'
                )
                path = os.path.join(self.node.debug_frames_dir, filename)
                if cv2.imwrite(path, dbg):
                    self._last_debug_save_stamp = stamp_s
                else:
                    self.node.get_logger().warn(
                        f'Failed to save debug detection frame: {path}',
                        throttle_duration_sec=5.0,
                    )

        if self.node.publish_debug_image and self.node.pub_debug.get_subscription_count() > 0:
            msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            msg.header = header
            self.node.pub_debug.publish(msg)
