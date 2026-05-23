"""Launch file for arm_controller."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('arm_controller'),
        'config',
        'controllers.yaml',
    )

    arm_controller_node = Node(
        package='arm_controller',
        executable='arm_controller_node',
        name='arm_controller_node',
        parameters=[config],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        arm_controller_node,
    ])
