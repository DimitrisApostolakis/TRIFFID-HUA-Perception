"""Launch both UGV nodes; defaults live in one configuration module."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from triffid_ugv_perception.config import NODE_PARAMETERS, BRIDGE_PARAMETERS


def generate_launch_description():
    defaults = dict(NODE_PARAMETERS, **BRIDGE_PARAMETERS)
    arguments = [DeclareLaunchArgument(name, default_value=str(value).lower() if isinstance(value, bool) else str(value))
                 for name, value in defaults.items()]
    def parameters(values):
        return {name: ParameterValue(LaunchConfiguration(name), value_type=type(value))
                for name, value in values.items()}
    return LaunchDescription(arguments + [
        Node(package='triffid_ugv_perception', executable='ugv_node',
             name='ugv_perception_node', output='screen', parameters=[parameters(NODE_PARAMETERS)]),
        Node(package='triffid_ugv_perception', executable='geojson_bridge',
             name='geojson_bridge', output='screen', parameters=[parameters(BRIDGE_PARAMETERS)]),
    ])
