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
    skip_missions = LaunchConfiguration('skip_missions')
    vehicle_config = LaunchConfiguration('vehicle_config')

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
        DeclareLaunchArgument('skip_missions', default_value='false',
                              description='true = pure lane-following test (no green light / roundabout / '
                                          'obstacle missions); starts driving immediately.'),
        # Absolute path so the camera finds it regardless of which workspace is sourced.
        # Without it, camera_node cannot locate src/config/vehicle_config.yaml and silently
        # falls back to MIPI 640x480 defaults -> USB camera fails to open and the node dies.
        DeclareLaunchArgument('vehicle_config',
                              default_value='/home/topst/SEAME_2026_Hackathon-clone/D-Racer-Kit/'
                                            'src/config/vehicle_config.yaml',
                              description='Vehicle/camera config (USB vs MIPI, device, resolution).'),

        # --- kit: camera ---
        Node(package='camera', executable='camera_node', name='camera_node', output='screen',
             parameters=[{'vehicle_config_file': vehicle_config}]),

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
             parameters=[{'course': course, 'race_dir': race_dir, 'skip_missions': skip_missions}]),

        # --- kit: control in AUTO mode ---
        # respawn: control_node 가 죽으면 모터가 조용히 멈추므로 자동 재시작한다.
        # (재시작 시 ESC 아밍 3초가 다시 돌지만 안전상 문제 없음.)
        Node(package='control', executable='control_node', name='control_node', output='screen',
             respawn=True, respawn_delay=1.0,
             parameters=[{'use_joystick_control': False, 'control_topic': '/control'}]),

        # --- kit: joystick kept alive for E-STOP safety ---
        Node(package='joystick', executable='joystick_node', name='joystick_node', output='screen'),

        # --- kit: battery (publishes battery_status for the monitor) ---
        Node(package='battery', executable='battery_node', name='battery_node', output='screen',
             respawn=True, respawn_delay=2.0),

        # --- kit: web monitor (shows camera + battery status) ---
        Node(package='monitor', executable='monitor_node', name='monitor_node', output='screen'),
    ])
