"""Validated camera calibration and rigid transforms, independent of ROS."""
from collections import deque
import numpy as np


def rigid_matrix(translation, quaternion_xyzw):
    """Return target-from-source matrix for a ROS transform quaternion."""
    translation = np.asarray(translation, dtype=np.float64)
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or q.shape != (4,):
        raise ValueError('Expected a 3-vector translation and xyzw quaternion')
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(q)):
        raise ValueError('Nonfinite transform')
    norm = np.linalg.norm(q)
    if abs(norm - 1.) > 1e-3:
        raise ValueError('Quaternion is not unit length')
    x, y, z, w = q / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = translation
    return matrix


def validate_rigid_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('Expected a finite 4x4 transform')
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Transform is not a rigid rotation/translation')
    return matrix


def transform_points(points, matrix):
    matrix = validate_rigid_matrix(matrix)
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def resolve_static_transform(transforms, target, source):
    """Compose recorded parent-from-child transforms, in either direction."""
    graph = {}
    for record in transforms:
        matrix = rigid_matrix(record['translation_m'], record['quaternion_xyzw'])
        parent, child = record['parent'], record['child']
        graph.setdefault(parent, []).append((child, matrix))
        graph.setdefault(child, []).append((parent, np.linalg.inv(matrix)))
    pending, seen = deque([(target, np.eye(4))]), set()
    while pending:
        frame, matrix = pending.popleft()
        if frame == source:
            return validate_rigid_matrix(matrix)
        if frame in seen:
            continue
        seen.add(frame)
        pending.extend((child, matrix @ edge) for child, edge in graph.get(frame, []) if child not in seen)
    raise ValueError(f'No recorded transform from {source} to {target}')


def calibrated_intrinsics(info, rgb_shape):
    """Pinhole intrinsics for this exact undistorted/aligned pixel grid.

    Reject unhandled distortion or resolution changes instead of guessing a
    resize/crop. The recorded PUI color images have zero distortion.
    """
    height, width = rgb_shape
    if (info.height, info.width) != (height, width):
        raise ValueError('CameraInfo resolution does not match RGB pixels')
    k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    d = np.asarray(info.d, dtype=np.float64)
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    if not np.all(np.isfinite(d)) or np.any(np.abs(d) > 1e-8):
        raise ValueError('Raw image distortion requires undistortion before pinhole projection')
    if not np.allclose(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1], atol=1e-8):
        raise ValueError('Unsupported camera matrix')
    return tuple(float(x) for x in (k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
