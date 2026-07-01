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
    race_dir = LaunchConfiguration('race_dir')
    aruco_dict = LaunchConfiguration('aruco_dict')
    aruco_inverted = LaunchConfiguration('aruco_inverted')
    model_param = LaunchConfiguration('model_param')
    model_bin = LaunchConfiguration('model_bin')

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='out',
                              description="'out' (S-curve+fork) or 'in' (roundabout)"),
        DeclareLaunchArgument('race_dir', default_value='left',
                              description="START direction set on race day: 'left' (CCW) or 'right' (CW). "
                                          "Flips roundabout turn + junction side together."),
        # Dynamic-obstacle marker. Best guess from the marker photo: 4X4_50, white-on-black.
        # Confirm with tools/identify_aruco.py on the board and override here if different.
        DeclareLaunchArgument('aruco_dict', default_value='DICT_4X4_50'),
        DeclareLaunchArgument('aruco_inverted', default_value='true'),
        DeclareLaunchArgument('model_param',
                              default_value='/home/topst/D-Racer/models/model.ncnn.param'),
        DeclareLaunchArgument('model_bin',
                              default_value='/home/topst/D-Racer/models/model.ncnn.bin'),

        # --- kit: camera ---
        Node(package='camera', executable='camera_node', name='camera_node', output='screen'),

        # --- new: perception (OpenCV) ---
        Node(package='perception', executable='lane_node', name='lane_node', output='screen',
             parameters=[{'race_dir': race_dir}]),
        Node(package='perception', executable='aruco_node', name='aruco_node', output='screen',
             parameters=[{'dictionary': aruco_dict, 'inverted': aruco_inverted}]),

        # --- new: inference (YOLO via NCNN, ARM-friendly) ---
        Node(package='inference', executable='yolo_ncnn_node', name='yolo_node', output='screen',
             parameters=[{'model_param': model_param, 'model_bin': model_bin}]),

        # --- new: decision (state machine + PID) ---
        Node(package='decision', executable='decision_node', name='decision_node', output='screen',
             parameters=[{'course': course, 'race_dir': race_dir}]),

        # --- kit: control in AUTO mode ---
        Node(package='control', executable='control_node', name='control_node', output='screen',
             parameters=[{'use_joystick_control': False, 'control_topic': '/control'}]),

        # --- kit: joystick kept alive for E-STOP safety ---
        Node(package='joystick', executable='joystick_node', name='joystick_node', output='screen'),
    ])
