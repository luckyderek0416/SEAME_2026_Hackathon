"""Bring up the full autonomous-racing stack (camera -> perception -> inference
-> decision -> control), plus battery + joystick(E-STOP).

  ros2 launch decision auto_race.launch.py course:=out
  ros2 launch decision auto_race.launch.py course:=in
  ros2 launch decision auto_race.launch.py course:=out lane_mode:=yellow steer_scale:=-1.0

Pipeline:
  camera_node ---/camera/image/compressed--->
      lane_node  (HSV)  --> /perception/lane      (offset)
      aruco_node        --> /perception/aruco      (obstacle)
      yolo_node  (NCNN) --> /inference/detections  (lights/signs)
          --> decision_node (state machine + PID) --> /control --> control_node --> car

control_node runs in AUTO mode (use_joystick_control:=False -> listens /control).
joystick_node stays alive so its E-STOP still works on the track.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def get_default_model_path():
    # YOLO NCNN export folder (best_ncnn_model), produced by training/.
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'training' / 'best_ncnn_model'
        if candidate.exists():
            return str(candidate)
    return str(Path.home() / 'SEAME_2026_Hackathon' / 'training' / 'best_ncnn_model')


def generate_launch_description():
    course = LaunchConfiguration('course')
    model_path = LaunchConfiguration('model_path')
    lane_mode = LaunchConfiguration('lane_mode')
    steer_center = LaunchConfiguration('steer_center')
    steer_scale = LaunchConfiguration('steer_scale')

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='out',
                              description="'out' (S-curve+fork) or 'in' (roundabout)"),
        DeclareLaunchArgument('model_path', default_value=get_default_model_path(),
                              description='YOLO NCNN model folder used by yolo_node'),
        DeclareLaunchArgument('lane_mode', default_value='white',
                              description="lane color: 'white' (boundary) or 'yellow' (center)"),
        DeclareLaunchArgument('steer_center', default_value='0.10',
                              description='steering neutral (= STEER_TRIM in vehicle_config.yaml)'),
        DeclareLaunchArgument('steer_scale', default_value='1.0',
                              description='flip to -1.0 if the car steers the wrong way'),

        # --- kit: camera ---
        Node(package='camera', executable='camera_node', name='camera_node', output='screen'),

        # --- perception (OpenCV: HSV lane + ArUco) ---
        Node(package='perception', executable='lane_node', name='lane_node', output='screen',
             parameters=[{'mode': lane_mode}]),
        Node(package='perception', executable='aruco_node', name='aruco_node', output='screen'),

        # --- inference (YOLO NCNN) ---
        Node(package='inference', executable='yolo_node', name='yolo_node', output='screen',
             parameters=[{'model_path': model_path}]),

        # --- decision (state machine + PID) ---
        Node(package='decision', executable='decision_node', name='decision_node', output='screen',
             parameters=[{
                 'course': course,
                 'steer_center': ParameterValue(steer_center, value_type=float),
                 'steer_scale': ParameterValue(steer_scale, value_type=float),
             }]),

        # --- kit: control in AUTO mode ---
        Node(package='control', executable='control_node', name='control_node', output='screen',
             parameters=[{'use_joystick_control': False, 'control_topic': '/control'}]),

        # --- kit: battery monitor ---
        Node(package='battery', executable='battery_node', name='battery_node', output='screen'),

        # --- kit: joystick kept alive for E-STOP safety ---
        Node(package='joystick', executable='joystick_node', name='joystick_node', output='screen'),
    ])
