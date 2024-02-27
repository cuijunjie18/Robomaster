import os
import yaml
import sys

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

sys.path.append(os.path.join(get_package_share_directory("rmcv_bringup"), "launch"))

# 根据机器人修改
robot_dir = "balance1"
use_can = True
bag_dir = "/home/ubuntu/rosbag_autoaim/rosbag2_4_rotate"


def generate_launch_description():
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription
    from launch.actions import ExecuteProcess

    node_params = os.path.join(
        get_package_share_directory("rmcv_bringup"),
        "config",
        robot_dir,
        "node_params.yaml",
    )

    launch_params = yaml.safe_load(
        open(
            os.path.join(
                get_package_share_directory("rmcv_bringup"),
                "config",
                robot_dir,
                "launch_params.yaml",
            )
        )
    )

    robot_description = Command(
        [
            "xacro ",
            os.path.join(
                get_package_share_directory("rmcv_bringup"),
                "urdf",
                "rm_gimbal.urdf.xacro",
            ),
            " xyz:=",
            launch_params["camera2gimbal"]["xyz"],
            " rpy:=",
            launch_params["camera2gimbal"]["rpy"],
        ]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description, "publish_frequency": 1000.0}
        ],
    )

    detector_dir = os.path.join(get_package_share_directory("rm_detector"))
    camera_info_url = os.path.join(
        "package://rmcv_bringup/config", robot_dir, "camera_info.yaml"
    )

    def get_intra_container():
        return ComposableNodeContainer(
            name="intra_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="rm_republish",
                    plugin="rm_republish::RmRepublishNode",
                    name="rm_republish",
                    parameters=[node_params],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="rm_detector",
                    plugin="rm_detector::DetectorNode",
                    name="rm_detector",
                    parameters=[node_params, {"detector_dir": detector_dir}],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                # ComposableNode(
                #     package='enemy_predictor',
                #     plugin='enemy_predictor::EnemyPredictorNode',
                #     name='enemy_predictor',
                #     parameters=[node_params],
                #     extra_arguments=[{'use_intra_process_comms': True}]
                # )
            ],
            output="both",
            emulate_tty=True,
            #            ros_arguments=['--ros-args', '--log-level',
            #                           'armor_detector:='+launch_params['detector_log_level']],
            on_exit=Shutdown(),
        )

    republish = Node(
        package="rm_republish",
        executable="rm_republish_node",
        name="rm_republish",
        output="both",
        emulate_tty=True,
        parameters=[node_params],
    )

    detector = Node(
        package="rm_detector",
        executable="rm_detector_node",
        name="rm_detector",
        output="both",
        emulate_tty=True,
        parameters=[node_params, {"detector_dir": detector_dir}],
    )

    predictor = Node(
        package="enemy_predictor",
        executable="enemy_predictor_node",
        name="enemy_predictor",
        output="both",
        emulate_tty=True,
        parameters=[node_params],
    )

    intra_container = get_intra_container()

    # delay_tracker_node = TimerAction(
    #     period=2.0,
    #     actions=[tracker_node],
    # )

    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
    )

    rosbag = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_dir,
            "-r",
            "0.2",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            # robot_state_publisher,
            republish,
            detector,
            predictor,
            foxglove,
            # rosbag,
        ]
    )
