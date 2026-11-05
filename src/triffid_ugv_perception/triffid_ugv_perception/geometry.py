"""Depth sampling and fixed optical-to-body geometry, without ROS/model dependencies."""
import numpy as np
from .config import _MAX_DEPTH_SAMPLES

def sample_depth_for_detection(
    det,
    depth_img,
    fx,
    fy,
    cx,
    cy,
    depth_scale_x=1.0,
    depth_scale_y=1.0,
    rgb_shape=None,
):
    """Sample depth at the detection's mask/bbox pixels and back-project
    to camera_optical_frame (X right, Y down, Z forward). Returns
    (M, 3) or None; RGB/depth share pixel grid on this camera."""
    depth_h, depth_w = depth_img.shape[:2]
    if rgb_shape is None:
        rgb_h, rgb_w = depth_h, depth_w
    else:
        rgb_h, rgb_w = rgb_shape
    mask = det.get('mask')
    x1, y1, x2, y2 = det['bbox']

    if mask is not None:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        if len(xs) > _MAX_DEPTH_SAMPLES:
            step = max(1, len(xs) // _MAX_DEPTH_SAMPLES)
            xs = xs[::step]
            ys = ys[::step]
    else:
        # Bbox grid fallback (dummy mode / no mask)
        ix1 = max(0, min(int(x1), rgb_w - 1))
        ix2 = max(ix1 + 1, min(int(x2), rgb_w))
        iy1 = max(0, min(int(y1), rgb_h - 1))
        iy2 = max(iy1 + 1, min(int(y2), rgb_h))
        side = max(1, min(ix2 - ix1, iy2 - iy1) // 10)
        xs_g, ys_g = np.meshgrid(
            np.arange(ix1, ix2, max(1, side)),
            np.arange(iy1, iy2, max(1, side)),
        )
        xs = xs_g.ravel()
        ys = ys_g.ravel()
        if len(xs) > _MAX_DEPTH_SAMPLES:
            step = max(1, len(xs) // _MAX_DEPTH_SAMPLES)
            xs = xs[::step]
            ys = ys[::step]

    # Map RGB coordinates onto depth indices for non-aligned fallback.
    xs_depth = np.rint(xs.astype(np.float64) * depth_scale_x).astype(np.int32)
    ys_depth = np.rint(ys.astype(np.float64) * depth_scale_y).astype(np.int32)

    # Clip to depth image bounds
    ok = (
        (xs_depth >= 0) & (xs_depth < depth_w) &
        (ys_depth >= 0) & (ys_depth < depth_h)
    )
    xs = xs[ok]
    ys = ys[ok]
    xs_depth = xs_depth[ok]
    ys_depth = ys_depth[ok]
    if len(xs_depth) == 0:
        return None

    # Read depth (uint16 mm)
    z_mm = depth_img[ys_depth, xs_depth].astype(np.float64)
    valid = z_mm > 0
    if not np.any(valid):
        return None

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z_m = z_mm[valid] / 1000.0

    # Pinhole back-projection → camera_optical_frame
    X = (xs - cx) * z_m / fx
    Y = (ys - cy) * z_m / fy
    Z = z_m
    return np.column_stack([X, Y, Z])

def camera_to_base(point_cam):
    """Map RealSense optical axes to b2/base_link axes.

    camera optical: X=right, Y=down, Z=forward
    b2/base_link:    X=forward, Y=left, Z=up
    """
    x_cam, y_cam, z_cam = point_cam
    return (
        float(z_cam),
        float(-x_cam),
        float(-y_cam),
    )

def camera_to_base_batch(points_cam):
    """Vectorized camera-optical → b2/base_link axis mapping."""
    points_cam = np.asarray(points_cam)
    return np.column_stack([
        points_cam[:, 2],
        -points_cam[:, 0],
        -points_cam[:, 1],
    ])

def bbox_to_3d_corners(u1, v1, u2, v2, depth, fx, fy, cx, cy):
    """Back-project a 2D bbox to 8 3D corners at a fixed depth (Z)."""
    x_left  = (u1 - cx) * depth / fx
    x_right = (u2 - cx) * depth / fx
    y_top   = (v1 - cy) * depth / fy
    y_bot   = (v2 - cy) * depth / fy

    x_min, x_max = min(x_left, x_right), max(x_left, x_right)
    y_min, y_max = min(y_top, y_bot), max(y_top, y_bot)

    z = float(depth)
    return np.array([
        [x_min, y_min, z],
        [x_min, y_max, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_min, z],
        [x_min, y_max, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
    ])

def nms_3d(detections, dist_thresh=0.5):
    """Keep the highest-confidence detection within dist_thresh metres —
    overlapping classes (e.g. Building/Destroyed building) can land on
    the same depth points and produce near-identical positions."""
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending — greedy NMS
    dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    keep = []
    for det in dets:
        pos = np.array(det['position'])
        suppressed = False
        for kept in keep:
            if np.linalg.norm(pos - np.array(kept['position'])) < dist_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
    return keep
