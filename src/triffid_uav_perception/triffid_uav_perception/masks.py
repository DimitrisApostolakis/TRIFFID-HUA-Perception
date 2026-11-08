"""Mask-derived observations and exact, cropped row-major binary RLE for Unity."""
import cv2
import numpy as np


def encode_rle(mask: np.ndarray) -> dict:
    """Alternating background/foreground runs; first run is always background.
    Row-major order (x changes fastest). This is NOT COCO's column-major RLE.
    """
    flat = np.asarray(mask, dtype=np.uint8).ravel(order='C')
    if flat.size == 0:
        raise ValueError('Cannot encode an empty-size mask')
    changes = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {'encoding': 'binary_rle_row_major', 'size_hw': list(mask.shape), 'counts': counts}


def decode_rle(encoded: dict) -> np.ndarray:
    if encoded.get('encoding') != 'binary_rle_row_major':
        raise ValueError('Unknown mask encoding')
    height, width = encoded['size_hw']
    counts = encoded['counts']
    if min(height, width) <= 0 or not counts or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
        raise ValueError('Invalid mask RLE')
    if sum(counts) != height * width:
        raise ValueError('RLE run lengths do not match mask dimensions')
    return np.repeat(np.arange(len(counts)) % 2, counts).astype(bool).reshape(height, width)


def observations_from_prediction(prediction, item, observation_prefix, *, min_area_px=1):
    mask = np.asarray(prediction.mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Backend mask must be a 2D boolean array in original-image pixels')
    if prediction.class_id != item.id:
        raise ValueError('Backend class does not match requested class')
    if min_area_px < 1:
        raise ValueError('min_area_px must be positive')
    probabilities = prediction.probabilities
    if probabilities is not None and (probabilities.shape != mask.shape or
            not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('Probability map must match mask dimensions and lie in [0, 1]')
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    observations = []
    for component in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[component])
        if area < min_area_px:
            continue
        crop = labels[y:y + height, x:x + width] == component
        cx, cy = (float(v) for v in centroids[component])
        # A mask centroid can lie in a hole or outside a concave region. Keep
        # the true centroid and a separate foreground pixel suitable for raycasting.
        ys, xs = np.nonzero(crop)
        closest = int(np.argmin((xs + x - cx) ** 2 + (ys + y - cy) ** 2))
        encoded = encode_rle(crop)
        encoded['offset_xy'] = [x, y]
        observations.append({
            'observation_id': f'{observation_prefix}:class_{item.id}:component_{component}',
            'class_id': item.id, 'class_name': item.name,
            'source_class_ids': list(item.source_ids), 'target_geometry': item.geometry,
            'mask': encoded, 'bbox_xyxy': [x, y, x + width, y + height],
            'centroid_xy': [cx, cy], 'raycast_point_xy': [int(xs[closest] + x), int(ys[closest] + y)],
            'area_px': area,
            'score': None if probabilities is None else float(
                probabilities[y:y + height, x:x + width][crop].mean()),
            'score_type': None if probabilities is None else 'mean_foreground_probability_uncalibrated',
        })
    return observations
