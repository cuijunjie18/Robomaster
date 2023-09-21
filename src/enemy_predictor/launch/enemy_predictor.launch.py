import os

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('enemy_predictor'), 'config', 'enemy_predictor_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(name='params_file',
                              default_value=params_file),
        Node(
            package='enemy_predictor',
            executable='enemy_predictor_node',
            output='both',
            emulate_tty=True,
            parameters=[LaunchConfiguration('params_file')],
        )
    ])