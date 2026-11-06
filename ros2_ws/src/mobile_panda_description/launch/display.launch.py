"""View the URDF alone in RViz (no simulator). Joints are at zero unless gui:=true (requires ros-humble-
joint-state-publisher-gui): ros2 launch mobile_panda_description display.launch.py [base:=wheels|planar]"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def nodes(context):
    share = Path(get_package_share_directory("mobile_panda_description"))
    gui = LaunchConfiguration("gui").perform(context).lower() == "true"
    urdf = share / f"urdf/mobile_panda_{LaunchConfiguration('base').perform(context)}.urdf"
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf.read_text()}]),
        Node(package="joint_state_publisher_gui" if gui else "joint_state_publisher",
             executable="joint_state_publisher_gui" if gui else "joint_state_publisher"),
        Node(package="rviz2", executable="rviz2"),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("base", default_value="wheels"),
                              DeclareLaunchArgument("gui", default_value="false"), OpaqueFunction(function=nodes)])
