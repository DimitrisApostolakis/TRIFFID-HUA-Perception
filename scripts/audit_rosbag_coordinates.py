"""Validate available calibration/pose evidence; does not claim surveyed accuracy."""
from bisect import bisect_left
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sqlite3
import struct

import numpy as np
import rosbag2_py as bag
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, Image, NavSatFix

from triffid_ugv_perception.config import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY

ROOT = Path('/ws/data/PUI/scenarios')
COLOR = '/b2/camera_front_435i/realsense_front_435i/color/'
DEPTH = '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/'


def read(path, topics):
    reader = bag.SequentialReader()
    storage = 'sqlite3' if path.suffix == '.db3' else 'mcap' if path.suffix == '.mcap' else ''
    reader.open(bag.StorageOptions(uri=str(path), storage_id=storage), bag.ConverterOptions('', ''))
    reader.set_filter(bag.StorageFilter(topics=topics))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, recorded = reader.read_next()
        yield topic, raw, recorded, types[topic]


def camera_audit(paths):
    calibrations, sample_images, times, header_times = {}, {}, {}, {}
    for path in paths:
        # Direct SQLite queries avoid decoding every multi-megabyte image just
        # to inspect timestamps and calibration.
        connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        for name in (COLOR+'image_raw', DEPTH+'image_raw'):
            entries = list(connection.execute(
                'SELECT timestamp,substr(data,1,12) FROM messages WHERE topic_id IN '
                '(SELECT id FROM topics WHERE name=?) ORDER BY timestamp', (name,)))
            stamps = [x[0] for x in entries]
            times.setdefault(name, []).extend(stamps)
            for _, prefix in entries:
                if prefix[:2] not in (b'\x00\x01', b'\x00\x00'):
                    raise ValueError('Expected plain CDR encapsulation for Image header')
                sec, nanos = struct.unpack_from('<iI' if prefix[1] else '>iI', prefix, 4)
                header_times.setdefault(name, []).append(sec*10**9+nanos)
            first = connection.execute('SELECT data FROM messages WHERE topic_id IN '
                '(SELECT id FROM topics WHERE name=?) ORDER BY timestamp LIMIT 1', (name,)).fetchone()
            if first and name not in sample_images:
                msg = deserialize_message(first[0], Image)
                sample_images[name] = dict(width=msg.width, height=msg.height,
                    encoding=msg.encoding, frame_id=msg.header.frame_id)
        for name in (COLOR+'camera_info', DEPTH+'camera_info'):
            for raw, in connection.execute('SELECT data FROM messages WHERE topic_id IN '
                    '(SELECT id FROM topics WHERE name=?)', (name,)):
                info = deserialize_message(raw, CameraInfo)
                calibration = dict(width=info.width, height=info.height,
                    frame_id=info.header.frame_id, k=list(info.k), p=list(info.p),
                    d=list(info.d), distortion_model=info.distortion_model)
                key = json.dumps(calibration, sort_keys=True)
                calibrations.setdefault(name, {})[key] = calibration
        connection.close()
    rgb_times, depth_times = sorted(times[COLOR+'image_raw']), sorted(times[DEPTH+'image_raw'])
    deltas = []
    for timestamp in rgb_times:
        index = bisect_left(depth_times, timestamp)
        candidates = depth_times[max(0,index-1):index+1]
        if candidates:
            deltas.append(min(abs(x-timestamp) for x in candidates)/1e6)
    results = dict(images=sample_images,
        rgb_count=len(rgb_times), depth_count=len(depth_times),
        rgb_start_ns=rgb_times[0] if rgb_times else None,
        rgb_end_ns=rgb_times[-1] if rgb_times else None,
        rgb_max_recorded_gap_s=max(np.diff(rgb_times))/1e9 if len(rgb_times)>1 else None,
        calibrations={name:list(values.values()) for name,values in calibrations.items()},
        nearest_depth_recorded_time_delta_ms=dict(
            median=float(np.median(deltas)), p95=float(np.percentile(deltas,95)),
            max=max(deltas), over_200ms=sum(x>200 for x in deltas)) if deltas else None)
    rgb_headers = sorted(header_times.get(COLOR+'image_raw', []))
    depth_headers = sorted(header_times.get(DEPTH+'image_raw', []))
    header_deltas = []
    for timestamp in rgb_headers:
        index = bisect_left(depth_headers, timestamp)
        candidates = depth_headers[max(0,index-1):index+1]
        if candidates:
            header_deltas.append(min(abs(x-timestamp) for x in candidates)/1e6)
    results['nearest_depth_header_time_delta_ms'] = dict(
        median=float(np.median(header_deltas)), p95=float(np.percentile(header_deltas,95)),
        max=max(header_deltas), over_200ms=sum(x>200 for x in header_deltas)) if header_deltas else None
    colors = list(calibrations.get(COLOR+'camera_info', {}).values())
    if colors:
        c = colors[0]
        k = c['k']
        pixels = np.array([[0.,0.], [c['width']-1.,0.],
                           [0.,c['height']-1.], [c['width']-1.,c['height']-1.],
                           [c['width']/2.,c['height']/2.]])
        actual = (pixels-np.array([k[2],k[5]]))/np.array([k[0],k[4]])
        fixed = (pixels-np.array([CAMERA_CX,CAMERA_CY]))/np.array([CAMERA_FX,CAMERA_FY])
        # This measures calibration mismatch only, at an illustrative Z=10 m.
        results['hardcoded_vs_recorded_intrinsics'] = dict(
            hardcoded=[CAMERA_FX,CAMERA_FY,CAMERA_CX,CAMERA_CY],
            recorded=[k[0],k[4],k[2],k[5]],
            lateral_error_at_10m_test_pixels_m=np.linalg.norm((actual-fixed)*10,axis=1).tolist(),
            test_pixels=pixels.tolist())
    return results


def pose_audit(paths):
    counts, fix_status = Counter(), Counter()
    valid_gps, headings, status_headings, yaw, covariances = [], [], [], [], []
    roll, pitch = [], []
    transforms = {}
    gps_frames = set()
    status_valid = global_valid = 0
    selected = ['/fix', '/b2/ugv_status', '/tf_static',
                '/b2/nicla/magnetometer/heading', '/b2/nicla/magnetometer/heading_broadcasted']
    decoders = {}
    for path in paths:
        for topic, raw, recorded, message_type in read(path, selected):
            if message_type not in decoders:
                decoders[message_type] = get_message(message_type)
            msg = deserialize_message(raw, decoders[message_type])
            counts[topic] += 1
            if topic == '/fix':
                gps_frames.add(msg.header.frame_id)
                fix_status[str(msg.status.status)] += 1
                if msg.status.status >= 0 and math.isfinite(msg.latitude) and math.isfinite(msg.longitude):
                    valid_gps.append([msg.longitude,msg.latitude])
                    if msg.position_covariance_type:
                        covariances.append(list(msg.position_covariance))
            elif topic == '/b2/ugv_status':
                status_valid += int(msg.gnss_valid)
                global_valid += int(msg.has_global_position and msg.global_position.gnss_valid)
                status_headings.append(float(msg.heading))
                yaw.append(float(msg.yaw))
                roll.append(float(msg.roll))
                pitch.append(float(msg.pitch))
            elif topic == '/tf_static':
                for t in msg.transforms:
                    transforms[(t.header.frame_id,t.child_frame_id)] = dict(
                        parent=t.header.frame_id, child=t.child_frame_id,
                        translation_m=[t.transform.translation.x,t.transform.translation.y,t.transform.translation.z],
                        quaternion_xyzw=[t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w])
            else:
                headings.append(float(msg.data))
    graph = {}
    for parent, child in transforms:
        graph.setdefault(parent,set()).add(child)
        graph.setdefault(child,set()).add(parent)
    reachable, pending = set(), ['b2/base_link']
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        reachable.add(node)
        pending.extend(graph.get(node,set())-reachable)
    result = dict(topic_counts=dict(counts), valid_fix_count=len(valid_gps),
        gps_status_counts=dict(fix_status), status_valid_gnss_count=status_valid,
        status_valid_global_position_count=global_valid,
        heading_range=[min(headings),max(headings)] if headings else None,
        status_heading_range=[min(status_headings),max(status_headings)] if status_headings else None,
        yaw_range_rad=[min(yaw),max(yaw)] if yaw else None,
        roll_range_rad=[min(roll),max(roll)] if roll else None,
        pitch_range_rad=[min(pitch),max(pitch)] if pitch else None,
        static_transforms=list(transforms.values()),
        static_frames_connected_to_base=sorted(reachable),
        gps_frame_ids=sorted(gps_frames),
        gps_frames_connected_to_base={frame:frame in reachable for frame in sorted(gps_frames)},
        gps_reported_horizontal_sigma_m=None)
    if covariances:
        variances = [max(c[0],c[4]) for c in covariances if c[0]>=0 and c[4]>=0]
        result['gps_reported_horizontal_sigma_m'] = dict(
            median=float(np.median(np.sqrt(variances))),
            max=float(np.max(np.sqrt(variances))))
    return result


def transform_validation(camera, robot):
    def matrix(t):
        x,y,z,w = t['quaternion_xyzw']
        norm = math.sqrt(x*x+y*y+z*z+w*w)
        if abs(norm-1.) > 1e-5:
            raise ValueError('Invalid recorded transform quaternion')
        x,y,z,w = np.array([x,y,z,w])/norm
        m = np.eye(4)
        m[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
        m[:3,3] = t['translation_m']
        return m
    graph = {}
    for t in robot['static_transforms']:
        m = matrix(t)
        graph.setdefault(t['parent'],[]).append((t['child'],m))
        graph.setdefault(t['child'],[]).append((t['parent'],np.linalg.inv(m)))
    target = camera['images'][COLOR+'image_raw']['frame_id']
    pending, seen, found = [('b2/base_link',np.eye(4))], set(), None
    while pending:
        frame, m = pending.pop()
        if frame == target:
            found = m
            break
        if frame in seen:
            continue
        seen.add(frame)
        pending.extend((child,m@edge) for child,edge in graph.get(frame,[]) if child not in seen)
    if found is None:
        return dict(camera_frame=target, connected_to_base=False)
    optical = np.array([[0.,0.,0.,1.], [0.,0.,10.,1.], [2.,1.,10.,1.]])
    recorded = (found@optical.T).T
    axis_only = np.column_stack([optical[:,2],-optical[:,0],-optical[:,1]])
    roundtrip = (np.linalg.inv(found)@recorded.T).T
    return dict(camera_frame=target, connected_to_base=True,
        camera_optical_to_base_matrix=found.tolist(),
        mount_translation_m=found[:3,3].tolist(),
        rotation_orthogonality_error=float(np.max(np.abs(found[:3,:3].T@found[:3,:3]-np.eye(3)))),
        rotation_determinant=float(np.linalg.det(found[:3,:3])),
        transform_roundtrip_max_error_m=float(np.max(np.abs(roundtrip-optical))),
        test_optical_points_m=optical[:,:3].tolist(),
        recorded_base_points_m=recorded[:,:3].tolist(),
        existing_axis_only_error_m=np.linalg.norm(recorded[:,:3]-axis_only,axis=1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', choices=['objects','scenario_1','scenario_2'], action='append')
    args = parser.parse_args()
    output = ROOT/'coordinate_validation.json'
    report = json.loads(output.read_text()) if output.exists() else dict(
        verdict='Absolute geolocation precision is not validated.', recordings={})
    sources = {
        'objects': (
            sorted(Path('/bags/Objects/rosbag2_2026_10_20-16_13_25/rosbag2_2026_10_20-16_13_25').glob('*.db3')),
            sorted(Path('/bags/Objects/rosbag2_2026_10_20-16_13_20/rosbag2_2026_10_20-16_13_20').glob('*.mcap'))),
    }
    for scenario in ('scenario_1','scenario_2'):
        if args.recording and scenario not in args.recording:
            continue
        folders = sorted((ROOT/scenario/'bags').iterdir())
        sources[scenario] = ([p for f in folders for p in sorted(f.glob('*.db3'))],
                             [p for f in folders for p in sorted(f.glob('*.mcap'))])
    for name,(camera,nuc) in sources.items():
        if args.recording and name not in args.recording:
            continue
        if not camera:
            raise RuntimeError(f'Missing camera files: {name}')
        print(f'Auditing {name}',flush=True)
        result = dict(camera=camera_audit(camera), robot=pose_audit(nuc))
        result['transform_validation'] = transform_validation(result['camera'],result['robot'])
        report['recordings'][name] = result
        (ROOT/'coordinate_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'Audited {name}',flush=True)


if __name__ == '__main__':
    main()
