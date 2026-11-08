"""Run cached PET/MASA observations through recorded RGB-D and GeoJSON stages.

Run in the ROS container. No model rerun or external API calls. The existing
pipeline's north-facing fallback is explicit when absolute heading is absent.
"""
from bisect import bisect_left, bisect_right
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np
import rosbag2_py as bag
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo, NavSatFix
from ugv_status_interfaces.msg import UnitreeStatus

from triffid_ugv_perception.pipeline import localize_detections
from triffid_ugv_perception.position import PositionProvider
from triffid_ugv_perception.geojson import GeoJSONFormatter
from triffid_ugv_perception.calibration import calibrated_intrinsics, resolve_static_transform


ROOT = Path('/ws')
RUN = ROOT / 'archive/samples_legacy_20261106/pet_masa_20261106_v2'
DEPTH = '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw'
INFO = '/b2/camera_front_435i/realsense_front_435i/color/camera_info'


def messages(path, topics):
    reader = bag.SequentialReader()
    storage_id = 'sqlite3' if path.suffix == '.db3' else ''
    reader.open(bag.StorageOptions(uri=str(path), storage_id=storage_id), bag.ConverterOptions('', ''))
    reader.set_filter(bag.StorageFilter(topics=topics))
    while reader.has_next():
        yield reader.read_next()


def stamp(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN/'calibrated_projection_v2')
    parser.add_argument('--max-depth-delta-ms', type=float, default=50.)
    args = parser.parse_args()
    if not 0 < args.max_depth_delta_ms <= 200:
        raise ValueError('Depth tolerance must be in (0, 200] ms')
    args.output.mkdir(exist_ok=False)
    audit_path = ROOT/'data/PUI/scenarios/coordinate_validation.json'
    audit = json.loads(audit_path.read_text())['recordings']['objects']
    transforms = audit['robot']['static_transforms']
    rows = [json.loads(x) for x in (RUN / 'tracks.jsonl').read_text().splitlines()]
    targets = [x['source']['header_timestamp_ns'] for x in rows]
    selected = [None] * len(rows)
    bridge = CvBridge()
    fixes, statuses, calibrations = [], [], []
    paired = ROOT / 'data/PUI/original_rosbag/paired_bag'
    for topic, raw, recorded in messages(paired, [DEPTH, '/fix', '/b2/ugv_status']):
        if topic == DEPTH:
            msg = deserialize_message(raw, Image)
            time_ns = stamp(msg)
            index = bisect_left(targets, time_ns)
            image = None
            for i in (index - 1, index):
                if not 0 <= i < len(rows):
                    continue
                delta = abs(time_ns - targets[i])
                if delta > args.max_depth_delta_ms * 1e6 or (selected[i] and selected[i][0] <= delta):
                    continue
                if msg.encoding not in ('16UC1', 'mono16'):
                    raise ValueError(f'Unsupported depth units: {msg.encoding}')
                if image is None:
                    image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').copy()
                selected[i] = (delta, image, time_ns, recorded, msg.header.frame_id)
        elif topic == '/fix':
            msg = deserialize_message(raw, NavSatFix)
            if (msg.status.status >= 0 and math.isfinite(msg.latitude)
                    and math.isfinite(msg.longitude) and -90 <= msg.latitude <= 90
                    and -180 <= msg.longitude <= 180):
                fixes.append((recorded, msg))
        else:
            statuses.append((recorded, deserialize_message(raw, UnitreeStatus)))
    original = ROOT / 'data/PUI/original_rosbag/bags/rosbag2_2026_10_20-16_13_25'
    if not (original / 'rosbag2_2026_10_20-16_13_25_0-002.db3').exists():
        # The downloaded bag metadata names *_0.db3, but the actual split
        # files have upload suffixes. Read those files directly.
        folder = Path('/bags/Objects/rosbag2_2026_10_20-16_13_25/rosbag2_2026_10_20-16_13_25')
        originals = sorted(folder.glob('*.db3'))
    else:
        originals = [original]
    for original_path in originals:
        for topic, raw, recorded in messages(original_path, [INFO]):
            msg = deserialize_message(raw, CameraInfo)
            if msg.k[0] > 0 and msg.k[4] > 0:
                calibrations.append((recorded, msg))
    if not fixes or not calibrations:
        raise RuntimeError('Recorded GPS or camera calibration is unavailable')
    fixes.sort(key=lambda x: x[0])
    calibrations.sort(key=lambda x: x[0])
    fix_times = [x[0] for x in fixes]
    calibration_times = [x[0] for x in calibrations]
    valid_heading = [m for _, m in statuses if m.gnss_valid and m.has_global_position
                     and m.global_position.gnss_valid]
    # Do not silently choose heading units or substitute unreferenced IMU yaw.
    if valid_heading:
        raise RuntimeError('Valid heading found: confirm its convention before projection')
    provider = PositionProvider.__new__(PositionProvider)
    provider._position_lock = threading.Lock()
    provider.gps_valid = True
    provider.heading_valid = False
    provider.robot_heading = 0.0
    formatter = GeoJSONFormatter(provider)
    features, best, skips = [], {}, Counter()
    depth_deltas = []
    for i, row in enumerate(rows):
        frame_time = row['source']['bag_timestamp_ns']
        fix_index = bisect_right(fix_times, frame_time) - 1
        calibration_index = bisect_right(calibration_times, frame_time) - 1
        if selected[i] is None or fix_index < 0 or calibration_index < 0:
            skips['frame_missing_depth_gps_or_calibration'] += 1
            continue
        fix_age = (frame_time - fix_times[fix_index]) / 1e9
        if fix_age > 3:
            skips['frame_stale_gps'] += 1
            continue
        window = fixes[max(0, fix_index - 6):fix_index + 1]
        provider.robot_lat = float(np.median([x.latitude for _, x in window]))
        provider.robot_lon = float(np.median([x.longitude for _, x in window]))
        altitudes = [x.altitude for _, x in window if math.isfinite(x.altitude)]
        provider.robot_alt = float(np.median(altitudes)) if altitudes else 0.
        info = calibrations[calibration_index][1]
        source = row['source']
        intrinsics = calibrated_intrinsics(info, (source['height'], source['width']))
        delta, depth, depth_stamp, _, depth_frame = selected[i]
        if (depth_frame != source['frame_id'] or info.header.frame_id != source['frame_id']
                or depth.shape != (source['height'], source['width'])):
            raise ValueError('Depth, RGB and calibration must share the same optical pixel grid')
        camera_to_base = resolve_static_transform(transforms, 'b2/base_link', source['frame_id'])
        depth_deltas.append(delta / 1e6)
        for track in row['tracks']:
            if not track.get('object_id'):
                continue
            # Preserve MASA's UUIDs: apply localization individually so spatial
            # NMS cannot merge separate tracked identities or nested objects.
            localized = localize_detections([track], depth,
                (source['height'], source['width']), intrinsics, merge_dist_m=0.,
                camera_to_target=camera_to_base)
            if not localized:
                skips['observation_no_valid_depth'] += 1
                continue
            det = localized[0]
            det.update(coordinates=provider.body_to_gps(*det['position']),
                       size=det['extent'], track_id=track['object_id'])
            feature = formatter.detections_to_geojson([det], detection_type='bbox')['features'][0]
            feature['id'] = f"{track['object_id']}:{row['frame']}"
            feature['properties'].update(
                object_id=track['object_id'], track_id=track['track_id'],
                frame=row['frame'], timestamp_ns=frame_time, image=source['file'],
                bbox_xyxy_px=track['bbox'], position_body_m=det['position'],
                depth_samples=det['n_depth_pts'], depth_delta_ms=delta / 1e6,
                depth_header_timestamp_ns=depth_stamp, gps_age_s=fix_age,
                robot_lon_lat=[provider.robot_lon, provider.robot_lat],
                camera_mount_applied=True, camera_to_base_matrix=camera_to_base.tolist(),
                saturated_depth_rejected=True,
                intrinsics_rgb=intrinsics, heading_valid=False,
                heading_assumed_degrees=0., location_quality='approximate_missing_absolute_heading',
                recovered=track.get('recovered', False),
                role_ambiguous=track.get('role_ambiguous', False))
            features.append(feature)
            previous = best.get(track['object_id'])
            if previous is None or track['confidence'] > previous['properties']['confidence']:
                best[track['object_id']] = feature
    description = ('Recorded RGB-D localization and pipeline GPS/GeoJSON conversion; '
        'GPS uses a causal seven-fix median. Absolute heading is missing: existing '
        'north-facing fallback applied. Geographic locations are approximate and '
        'are not validated. Recorded full camera-to-base TF is applied. '
        f'RGB/depth header times differ by at most {args.max_depth_delta_ms:g} ms. '
        'GPS antenna offset and body attitude are not corrected. '
        'Zero and saturated 65535 depth pixels are excluded. '
        'BBox depth can include background. No API calls. MASA UUIDs preserved.')
    registry = json.loads((RUN / 'object_registry.json').read_text())
    missing = sorted(set(registry['objects']) - set(best))
    unlocated = []
    for object_id in missing:
        obj = registry['objects'][object_id]
        unlocated.append(dict(type='Feature', id=object_id, geometry=None, properties=dict(
            object_id=object_id, **{'class': obj['class_name']},
            class_id=obj['class_id'], bbox_xyxy_px=obj['bbox'],
            last_seen_timestamp_ns=obj['last_seen_ns'],
            confidence_at_creation=obj['created_event']['confidence'],
            location_quality='no_reliable_depth_or_time_match')))
    for name, items in [('detections_projected_observations.geojson', features),
                        ('detections_projected.geojson', list(best.values()))]:
        if name == 'detections_projected.geojson':
            items = [dict(f, id=f['properties']['object_id']) for f in items] + unlocated
        collection = dict(type='FeatureCollection', description=description, features=items)
        (args.output / name).write_text(json.dumps(collection, indent=2, allow_nan=False) + '\n')
    summary = dict(frames=len(rows), projected_observations=len(features),
        projected_objects=len(best), total_confirmed_objects=len(registry['objects']), missing_object_ids=missing,
        skips=dict(skips), camera_calibrations=len(calibrations), gps_fixes=len(fixes),
        status_messages=len(statuses), valid_absolute_heading=False,
        camera_mount_applied=True, max_depth_delta_ms=args.max_depth_delta_ms,
        saturated_depth_rejected=True,
        calibration_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        depth_delta_ms_max=max(depth_deltas) if depth_deltas else None,
        projection=description)
    (args.output / 'projection_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
