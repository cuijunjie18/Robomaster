import os

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('rm_detector'), 'config', 'detector_params.yaml')

    detector_dir = get_package_share_directory('rm_detector')

    return LaunchDescription([
        DeclareLaunchArgument(name='params_file',
                              default_value=params_file),
        DeclareLaunchArgument(name='detector_dir',
                              default_value=detector_dir),

        Node(
            package='rm_detector',
            executable='rm_detector_node',
            output='both',
            emulate_tty=True,
            parameters=[LaunchConfiguration('params_file'), {
                'detector_dir': LaunchConfiguration('detector_dir'),
            }],
        )
    ])
