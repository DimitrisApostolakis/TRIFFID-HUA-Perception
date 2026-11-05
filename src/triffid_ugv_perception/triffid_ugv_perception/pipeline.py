"""Existing RGB-D localization and spatial filtering as a ROS-free stage."""
import numpy as np
from .geometry import (sample_depth_for_detection, camera_to_base, camera_to_base_batch, bbox_to_3d_corners, nms_3d)

def localize_detections(raw_detections, depth_image, rgb_shape, intrinsics, merge_dist_m=0.5):
    rgb_h, rgb_w = rgb_shape
    depth_h, depth_w = depth_image.shape[:2]
    depth_scale_x, depth_scale_y = depth_w / rgb_w, depth_h / rgb_h
    fx, fy, cx, cy = intrinsics
    detections_3d = []

    for det in raw_detections:
        x1, y1, x2, y2 = det['bbox']
        mask = det.get('mask')

        # Sample depth at detection pixels (pixel-aligned: same coords)
        matched_pts = sample_depth_for_detection(
            det, depth_image, fx, fy, cx, cy,
            depth_scale_x=depth_scale_x,
            depth_scale_y=depth_scale_y,
            rgb_shape=(rgb_h, rgb_w),
        )
        if matched_pts is None or len(matched_pts) == 0:
            continue

        n_inside = len(matched_pts)

        # Median position in camera_optical_frame
        median_pt = np.median(matched_pts, axis=0)  # (3,)

        # ── Step 5: camera optical axes → b2/base_link ──────────
        # Assumption: the front RealSense is rigidly aligned with the UGV
        # and its translational offset from base_link is neglected.
        pt_base = camera_to_base(tuple(median_pt))

        # ── Compute 3D bbox extent in base_link ─────────────────
        pt_min_cam = np.min(matched_pts, axis=0)
        pt_max_cam = np.max(matched_pts, axis=0)
        cloud_extent = pt_max_cam - pt_min_cam

        if np.any(cloud_extent > 1e-3):
            corners_cam = np.array([
                [pt_min_cam[0], pt_min_cam[1], pt_min_cam[2]],
                [pt_min_cam[0], pt_min_cam[1], pt_max_cam[2]],
                [pt_min_cam[0], pt_max_cam[1], pt_min_cam[2]],
                [pt_min_cam[0], pt_max_cam[1], pt_max_cam[2]],
                [pt_max_cam[0], pt_min_cam[1], pt_min_cam[2]],
                [pt_max_cam[0], pt_min_cam[1], pt_max_cam[2]],
                [pt_max_cam[0], pt_max_cam[1], pt_min_cam[2]],
                [pt_max_cam[0], pt_max_cam[1], pt_max_cam[2]],
            ])
        else:
            corners_cam = bbox_to_3d_corners(
                x1, y1, x2, y2, median_pt[2], fx, fy, cx, cy,
            )

        corners_base = camera_to_base_batch(corners_cam)
        extent_base = tuple(
            float(v) for v in np.ptp(corners_base, axis=0)
        )

        detections_3d.append({
            'position': pt_base,
            'extent': extent_base,
            'class_id': det['class_id'],
            'class_name': det['class_name'],
            'confidence': det['confidence'],
            'bbox': det['bbox'],
            'n_depth_pts': n_inside,
        })

    # ── Step 5.5: 3-D NMS — deduplicate overlapping detections ───
    detections_3d = nms_3d(
        detections_3d, dist_thresh=float(merge_dist_m),
    )

    return detections_3d
