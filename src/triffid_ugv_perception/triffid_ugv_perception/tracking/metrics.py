import numpy as np

def _iou_batch(bboxes_a, bboxes_b):
    """Vectorised (M, N) IoU matrix between two sets of (x1,y1,x2,y2) boxes."""
    a = np.asarray(bboxes_a, dtype=np.float64)
    b = np.asarray(bboxes_b, dtype=np.float64)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    union = np.maximum(union, 1e-12)

    return inter / union

def _pos_distance_batch(positions_a, positions_b):
    """(M, N) Euclidean distance matrix; inf where either position is None."""
    M = len(positions_a)
    N = len(positions_b)
    dist = np.full((M, N), np.inf, dtype=np.float64)
    for i, pa in enumerate(positions_a):
        if pa is None:
            continue
        pa = np.asarray(pa, dtype=np.float64)
        for j, pb in enumerate(positions_b):
            if pb is None:
                continue
            pb = np.asarray(pb, dtype=np.float64)
            dist[i, j] = np.linalg.norm(pa - pb)
    return dist
