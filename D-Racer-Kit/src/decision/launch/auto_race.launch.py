"""Bring up the full autonomous-racing stack.

  ros2 launch decision auto_race.launch.py course:=in
  ros2 launch decision auto_race.launch.py course:=in model_path:=/path/best.pt

Reuses the kit's camera_node and control_node; adds lane/aruco/yolo/decision.
control_node runs in AUTO mode (use_joystick_control:=False -> listens /control).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _lane_node(context, *args, **kwargs):
    """course 값에 따라 lane_node 를 만든다. Out 코스는 노란 차선이 없으므로(회전교차로
    없음) use_yellow=False 로 흰색만 검출해, 바닥/조명의 노란기 픽셀이 차선 마스크를
    오염시키는 걸 원천 차단한다. In 코스는 노란 링 추종을 위해 노란색을 유지한다.
    (Out 갈림길 방향은 표지판 기반이라 노란색을 꺼도 분기 선택은 그대로 동작.)"""
    course_val = LaunchConfiguration('course').perform(context).strip().lower()
    race_dir_val = LaunchConfiguration('race_dir').perform(context)
    use_yellow = course_val != 'out'
    return [Node(package='perception', executable='lane_node', name='lane_node',
                 output='screen',
                 parameters=[{'race_dir': race_dir_val, 'use_yellow': use_yellow}])]


def generate_launch_description():
    course = LaunchConfiguration('course')
    race_dir = LaunchConfiguration('race_dir')
    aruco_dict = LaunchConfiguration('aruco_dict')
    aruco_inverted = LaunchConfiguration('aruco_inverted')
    skip_missions = LaunchConfiguration('skip_missions')
    vehicle_config = LaunchConfiguration('vehicle_config')

    return LaunchDescription([
        DeclareLaunchArgument('course', default_value='in',
                              description="'out' (S-curve+fork) or 'in' (roundabout)"),
        DeclareLaunchArgument('race_dir', default_value='left',
                              description="START direction set on race day: 'left' (CCW) or 'right' (CW). "
                                          "Flips roundabout turn + junction side together."),
        # Dynamic-obstacle marker. Best guess from the marker photo: 4X4_50, white-on-black.
        # Confirm with tools/identify_aruco.py on the board and override here if different.
        DeclareLaunchArgument('aruco_dict', default_value='DICT_6X6_50'),
        DeclareLaunchArgument('aruco_inverted', default_value='true'),
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
        # lane_node 는 course 에 따라 use_yellow 를 바꾸므로 OpaqueFunction 으로 생성.
        OpaqueFunction(function=_lane_node),
        Node(package='perception', executable='aruco_node', name='aruco_node', output='screen',
             parameters=[{'dictionary': aruco_dict, 'inverted': aruco_inverted}]),

        # --- new: inference (YOLO via NCNN, ARM-friendly) ---
        # 보드에 ultralytics(torch) 가 없어 yolo_node 는 모델 로드 실패 -> 빈 검출.
        # 순수 ncnn 경로(yolo_ncnn_node, run80plus 검증)로 loun v3 가중치를 문다.
        # 파일은 리포 내 training/best_ncnn_model 의 model.ncnn.{param,bin} (동일 가중치).
        Node(package='inference', executable='yolo_ncnn_node', name='yolo_node', output='screen',
             parameters=[{
                 'model_param': '/home/topst/SEAME_2026_Hackathon-clone/training/'
                                'best_ncnn_model/model.ncnn.param',
                 'model_bin': '/home/topst/SEAME_2026_Hackathon-clone/training/'
                              'best_ncnn_model/model.ncnn.bin',
             }]),

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
        # 자율주행에선 조향/스로틀은 안 쓰고 X버튼 E-STOP만 쓰므로, 5Hz 디버그 로그
        # (SSH I/O 부하)는 끈다. 비상정지 기능은 그대로 살아있다.
        Node(package='joystick', executable='joystick_node', name='joystick_node', output='screen',
             parameters=[{'debug_log_enable': False}]),

        # --- kit: battery (publishes battery_status for the monitor) ---
        Node(package='battery', executable='battery_node', name='battery_node', output='screen',
             respawn=True, respawn_delay=2.0),

        # --- kit: web monitor (shows camera + battery status) ---
        Node(package='monitor', executable='monitor_node', name='monitor_node', output='screen'),
    ])
