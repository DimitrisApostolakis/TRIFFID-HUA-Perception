from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Shared arguments
        DeclareLaunchArgument('model_path', default_value='/ws/best.pt'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.35'),
        DeclareLaunchArgument('target_frame', default_value='b2/base_link'),

        # Configurable topic names (pixel-aligned RGB-D camera)
        DeclareLaunchArgument('rgb_image_topic',
                              default_value='/b2/camera_front_435i/realsense_front_435i/color/image_raw_broadcasted'),
        DeclareLaunchArgument('depth_image_topic',
                              default_value='/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw_broadcasted'),
        DeclareLaunchArgument('thermal_image_topic',
                              default_value='/b2/camera_flir/image_raw_broadcasted'),
        DeclareLaunchArgument('use_dummy_detections', default_value='false'),
        DeclareLaunchArgument('save_debug_frames', default_value='true'),
        DeclareLaunchArgument('debug_frames_dir', default_value='/ws/detection_frames'),
        DeclareLaunchArgument('debug_frame_period_s', default_value='1.0'),
        DeclareLaunchArgument('save_thermal_frames', default_value='false'),
        DeclareLaunchArgument('thermal_frames_dir', default_value='/ws/thermal_frames'),
        DeclareLaunchArgument('thermal_frame_period_s', default_value='1.0'),

        # GeoJSON bridge arguments
        DeclareLaunchArgument('use_position_api', default_value='true'),
        DeclareLaunchArgument('ugv_status_topic', default_value='/b2/ugv_status'),
        DeclareLaunchArgument('ugv_status_timeout_s', default_value='3.0'),
        DeclareLaunchArgument('publish_to_api', default_value='false'),
        DeclareLaunchArgument('save_to_disk', default_value='true'),
        DeclareLaunchArgument('publish_period_s', default_value='5.0'),
        DeclareLaunchArgument('output_dir', default_value='/ws/samples'),
        DeclareLaunchArgument('target_geojson_stamp', default_value='0'),

        # UGV Perception Node
        Node(
            package='triffid_ugv_perception',
            executable='ugv_node',
            name='ugv_perception_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'target_frame': LaunchConfiguration('target_frame'),
                'rgb_image_topic': LaunchConfiguration('rgb_image_topic'),
                'depth_image_topic': LaunchConfiguration('depth_image_topic'),
                'thermal_image_topic': LaunchConfiguration('thermal_image_topic'),
                'use_dummy_detections': LaunchConfiguration('use_dummy_detections'),
                'save_debug_frames': LaunchConfiguration('save_debug_frames'),
                'debug_frames_dir': LaunchConfiguration('debug_frames_dir'),
                'debug_frame_period_s': LaunchConfiguration('debug_frame_period_s'),
                'save_thermal_frames': LaunchConfiguration('save_thermal_frames'),
                'thermal_frames_dir': LaunchConfiguration('thermal_frames_dir'),
                'thermal_frame_period_s': LaunchConfiguration('thermal_frame_period_s'),
            }],
        ),

        # GeoJSON Bridge Node
        Node(
            package='triffid_ugv_perception',
            executable='geojson_bridge',
            name='geojson_bridge',
            output='screen',
            parameters=[{
                'use_position_api': LaunchConfiguration('use_position_api'),
                'ugv_status_topic': LaunchConfiguration('ugv_status_topic'),
                'ugv_status_timeout_s': LaunchConfiguration('ugv_status_timeout_s'),
                'publish_to_api': LaunchConfiguration('publish_to_api'),
                'save_to_disk': LaunchConfiguration('save_to_disk'),
                'publish_period_s': LaunchConfiguration('publish_period_s'),
                'output_dir': LaunchConfiguration('output_dir'),
                'target_geojson_stamp': LaunchConfiguration('target_geojson_stamp'),
            }],
        ),
    ])
