"""Bring up the full autonomous-racing stack.

  ros2 launch decision auto_race.launch.py course:=out
  ros2 launch decision auto_race.launch.py course:=in model_path:=/path/best.pt

Reuses the kit's camera_node and control_node; adds lane/aruco/yolo/decision.
control_node runs in AUTO mode (use_joystick_control:=False -> listens /control).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    course = LaunchConfiguration('course')
    model_path = LaunchConfiguration('model_path')

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='out',
                              description="'out' (S-curve+fork) or 'in' (roundabout)"),
        DeclareLaunchArgument('model_path',
                              default_value='/home/topst/D-Racer/models/best.pt'),

        # --- kit: camera ---
        Node(package='camera', executable='camera_node', name='camera_node', output='screen'),

        # --- new: perception (OpenCV) ---
        Node(package='perception', executable='lane_node', name='lane_node', output='screen'),
        Node(package='perception', executable='aruco_node', name='aruco_node', output='screen'),

        # --- new: inference (YOLO) ---
        Node(package='inference', executable='yolo_node', name='yolo_node', output='screen',
             parameters=[{'model_path': model_path}]),

        # --- new: decision (state machine + PID) ---
        Node(package='decision', executable='decision_node', name='decision_node', output='screen',
             parameters=[{'course': course}]),

        # --- kit: control in AUTO mode ---
        Node(package='control', executable='control_node', name='control_node', output='screen',
             parameters=[{'use_joystick_control': False, 'control_topic': '/control'}]),

        # --- kit: joystick kept alive for E-STOP safety ---
        Node(package='joystick', executable='joystick_node', name='joystick_node', output='screen'),
    ])
