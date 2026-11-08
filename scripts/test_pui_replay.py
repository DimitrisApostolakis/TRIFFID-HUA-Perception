"""Replay paired bags on one ROS timeline and run SANSA without external APIs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml

root = Path(os.environ.get('PUI_RUN', '/ws/data/PUI/original_rosbag'))
root.mkdir(parents=True, exist_ok=True)
merged = root/'paired_bag'
if (merged/'metadata.yaml').exists():
    metadata=yaml.safe_load((merged/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    names={t['topic_metadata']['name'] for t in metadata['topics_with_message_count'] if t['message_count']}
    required={'/b2/camera_front_435i/realsense_front_435i/color/camera_info','/tf_static'}
    if not required.issubset(names):
        raise RuntimeError('Paired bag lacks recorded camera calibration/TF. Rebuild it with the updated merge script, or set PUI_RUN to data/PUI/scenarios/scenario_1.')
classes = os.environ.get('PUI_CLASSES') or ','.join(json.loads(Path('/ws/ugv_supports/PUI/import_report.json').read_text())['one_shot_classes'])
cmd = ['ros2','launch','triffid_ugv_perception','ugv_perception.launch.py',
    'backend:=sansa','support_bank:=/ws/ugv_supports/PUI','shots:=1','classes:='+classes,
    'sansa_repo:=/models/ThirdParty/SANSA',
    'checkpoint:=/models/Checkpoints/SANSA/adapter_generalist.pth',
    'sam2_checkpoint:=/models/Checkpoints/SANSA/sam2_hiera_large.pt',
    'device:='+os.environ.get('PUI_DEVICE','cuda'),
    'rgb_image_topic:=/b2/camera_front_435i/realsense_front_435i/color/image_raw',
    'depth_image_topic:=/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw',
    'use_position_api:=false','publish_to_api:=false',
    'output_dir:='+str(root/'geojson'),
    'debug_frames_dir:='+str(root/'overlays')]
(root/'launch_command.json').write_text(json.dumps(cmd,indent=2))
with (root/'perception.log').open('w') as log:
    node = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    recorder = None
    try:
        deadline=time.monotonic()+300
        while 'Detector: sansa' not in (root/'perception.log').read_text():
            if 'process has died' in (root/'perception.log').read_text(): raise RuntimeError('Perception process failed; see perception.log')
            if node.poll() is not None: raise RuntimeError('Perception launch failed; see perception.log')
            if time.monotonic()>deadline: raise RuntimeError('Model initialization timed out')
            time.sleep(1)
        time.sleep(3)
        record_log = (root/'record.log').open('w')
        recorder = subprocess.Popen(['ros2','bag','record','-o',str(root/'output_bag'),
            '/ugv/detections/front/detections_3d','/ugv/detections/front/geojson',
            '/b2/ugv_status','/fix'], stdout=record_log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2)
        bags = [merged] if (merged/'metadata.yaml').exists() else sorted((root/'bags').iterdir())
        replay=['ros2','bag','play']
        for bag in bags: replay += ['-i',str(bag)]
        replay += ['--clock','--read-ahead-queue-size','20','--disable-keyboard-controls','--qos-profile-overrides-path','/ws/scripts/pui_replay_qos.yaml','--topics',
            '/b2/camera_front_435i/realsense_front_435i/color/image_raw',
            '/b2/camera_front_435i/realsense_front_435i/color/camera_info',
            '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw',
            '/b2/ugv_status','/fix','/tf','/tf_static',
            '/b2/nicla/magnetometer/heading']
        (root/'replay_command.json').write_text(json.dumps(replay,indent=2))
        with (root/'replay.log').open('w') as replay_log:
            subprocess.run(replay,stdout=replay_log,stderr=subprocess.STDOUT,check=True,timeout=300)
        time.sleep(20)
        if node.poll() is not None or 'process has died' in (root/'perception.log').read_text():
            raise RuntimeError('Perception exited during replay')
    finally:
        if recorder is not None:
            os.killpg(recorder.pid,signal.SIGINT)
            recorder.wait(timeout=30)
            record_log.close()
        if node.poll() is None:
            os.killpg(node.pid,signal.SIGINT)
            try: node.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(node.pid,signal.SIGTERM)
print('Paired replay finished; inspect overlays and GeoJSON for detection results.')
