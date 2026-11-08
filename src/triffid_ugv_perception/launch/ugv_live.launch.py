"""YOLOE-v8-L + MASA live profile with explicit appearance transport."""
import os
import json
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from triffid_ugv_perception.config import NODE_PARAMETERS,BRIDGE_PARAMETERS

def generate_launch_description():
    profile=json.loads((Path(__file__).resolve().parents[1]/'config/ugv_live_pui.json').read_text())
    perception=dict(NODE_PARAMETERS,**profile['node'])
    bridge=dict(BRIDGE_PARAMETERS,**profile['bridge'])
    defaults=dict(perception,**bridge)
    args=[DeclareLaunchArgument(k,default_value=str(v).lower() if isinstance(v,bool) else str(v)) for k,v in defaults.items()]
    args += [DeclareLaunchArgument(k,default_value=v) for k,v in profile['runtime'].items()]
    def params(values):
        return {k:ParameterValue(LaunchConfiguration(k),value_type=type(v)) for k,v in values.items()}
    runtime=dict(PYTHONPATH=[LaunchConfiguration('inference_pythonpath'),':',os.environ.get('PYTHONPATH','')],
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false',
        MPLCONFIGDIR='/tmp/triffid-live-mpl',YOLO_CONFIG_DIR='/tmp/triffid-live-yolo',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return LaunchDescription(args+[
        Node(package='triffid_ugv_perception',executable='ugv_node',name='ugv_perception_node',output='screen',
            prefix=[LaunchConfiguration('inference_python')],additional_env=runtime,parameters=[params(perception)]),
        Node(package='triffid_ugv_perception',executable='geojson_bridge',name='geojson_bridge',output='screen',parameters=[params(bridge)])])
