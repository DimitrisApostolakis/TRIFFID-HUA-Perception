"""Central defaults; existing calibration, topic, tracker and delivery settings retained."""

BASE_FRAME = 'b2/base_link'
DEFAULT_RGB_TOPIC = '/b2/camera_front_435i/realsense_front_435i/color/image_raw_broadcasted'
DEFAULT_DEPTH_TOPIC = '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw_broadcasted'
DEFAULT_THERMAL_TOPIC = '/b2/camera_flir/image_raw_broadcasted'
CAMERA_FRAME = 'camera_color_optical_frame'
CAMERA_FX = 606.883056640625
CAMERA_FY = 606.7998657226562
CAMERA_CX = 323.86395263671875
CAMERA_CY = 244.37435913085938
_MAX_DEPTH_SAMPLES = 500
NODE_PARAMETERS = {
    'model_path': '/ws/best.pt',
    'confidence_threshold': 0.35,
    'target_frame': 'b2/base_link',
    'use_dummy_detections': False,
    'yolo_imgsz': 1280,
    'tracker_iou_threshold': 0.3,
    'tracker_iou_threshold_low': 0.15,
    'tracker_conf_high': 0.4,
    'tracker_max_age': 30,
    'tracker_n_init': 3,
    'tracker_pos_gate': 2.0,
    'nms_merge_dist_m': 0.5,
    'publish_debug_image': True,
    'save_debug_frames': True,
    'debug_frames_dir': '/ws/detection_frames',
    'debug_frame_period_s': 1.0,
    'save_thermal_frames': False,
    'thermal_frames_dir': '/ws/thermal_frames',
    'thermal_frame_period_s': 1.0,
    'rgb_image_topic': '/b2/camera_front_435i/realsense_front_435i/color/image_raw_broadcasted',
    'depth_image_topic': '/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw_broadcasted',
    'thermal_image_topic': '/b2/camera_flir/image_raw_broadcasted',
    'backend': 'sansa',
    'sansa_repo': '/ws/models/SANSA',
    'checkpoint': '/ws/models/adapter_generalist.pth',
    'sam2_checkpoint': '/ws/models/sam2_hiera_large.pt',
    'support_bank': '/ws/ugv_supports',
    'shots': 5,
    'device': 'cuda',
    'image_size': 640,
    'mask_threshold': 0.5,
    'min_area_px': 1,
    'classes': '',
}

BRIDGE_PARAMETERS = {
    'api_url': 'https://crispres.com/wp-json/map-manager/v1/features',
    'position_api_url': 'https://crispres.com/wp-json/mqtt/v1/latest',
    'use_position_api': True,
    'ugv_status_topic': '/b2/ugv_status',
    'ugv_status_timeout_s': 3.0,
    'publish_to_api': False,
    'save_to_disk': True,
    'gps_origin_lat': 0.0,
    'gps_origin_lon': 0.0,
    'gps_origin_alt': 0.0,
    'dedup_radius_m': 3.0,
    'publish_period_s': 5.0,
    'output_dir': '/ws/samples',
    'target_geojson_stamp': 0,
}



def declare_parameters(node, defaults):
    for name, default in defaults.items():
        node.declare_parameter(name, default)
