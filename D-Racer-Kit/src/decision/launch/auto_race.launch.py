"""자율주행 레이싱 전체 스택을 기동한다.

  ros2 launch decision auto_race.launch.py course:=in
  ros2 launch decision auto_race.launch.py course:=in model_path:=/path/best.pt

키트의 camera_node 와 control_node 를 재사용하고 lane/aruco/yolo/decision 을 추가한다.
control_node 는 AUTO 모드로 동작한다 (use_joystick_control:=False -> /control 구독).
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
    # course 도 전달: In 코스에서만 색상 추종 상태머신(흰->노랑->흰)이 활성화된다.
    return [Node(package='perception', executable='lane_node', name='lane_node',
                 output='screen',
                 parameters=[{'race_dir': race_dir_val, 'use_yellow': use_yellow,
                              'course': course_val}])]


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
        DeclareLaunchArgument('course', default_value='in',
                              description="'out' (S-curve+fork) or 'in' (roundabout)"),
        DeclareLaunchArgument('race_dir', default_value='left',
                              description="START direction set on race day: 'left' (CCW) or 'right' (CW). "
                                          "Flips roundabout turn + junction side together."),
        # 동적 장애물 마커. 마커 사진 기준 최선의 추정: 4X4_50, 검정 바탕에 흰색.
        # 실제 보드에 tools/identify_aruco.py 로 확인하고 다르면 여기서 override 할 것.
        DeclareLaunchArgument('aruco_dict', default_value='DICT_4X4_50'),
        DeclareLaunchArgument('aruco_inverted', default_value='true'),
        DeclareLaunchArgument('model_param',
                              default_value='/home/topst/D-Racer/models/model.ncnn.param'),
        DeclareLaunchArgument('model_bin',
                              default_value='/home/topst/D-Racer/models/model.ncnn.bin'),
        DeclareLaunchArgument('skip_missions', default_value='false',
                              description='true = pure lane-following test (no green light / roundabout / '
                                          'obstacle missions); starts driving immediately.'),
        # 어느 워크스페이스를 source 했든 카메라가 찾을 수 있도록 절대 경로 사용.
        # 없으면 camera_node 가 src/config/vehicle_config.yaml 을 못 찾고 조용히
        # MIPI 640x480 기본값으로 떨어져 -> USB 카메라 열기에 실패하고 노드가 죽는다.
        DeclareLaunchArgument('vehicle_config',
                              default_value='/home/topst/SEAME_2026_Hackathon-clone/D-Racer-Kit/'
                                            'src/config/vehicle_config.yaml',
                              description='Vehicle/camera config (USB vs MIPI, device, resolution).'),

        # --- 키트: 카메라 ---
        Node(package='camera', executable='camera_node', name='camera_node', output='screen',
             parameters=[{'vehicle_config_file': vehicle_config}]),

        # --- 신규: perception (OpenCV) ---
        # lane_node 는 course 에 따라 use_yellow 를 바꾸므로 OpaqueFunction 으로 생성.
        OpaqueFunction(function=_lane_node),
        Node(package='perception', executable='aruco_node', name='aruco_node', output='screen',
             parameters=[{'dictionary': aruco_dict, 'inverted': aruco_inverted}]),

        # --- 신규: inference (NCNN 기반 YOLO, ARM 친화적) ---
        Node(package='inference', executable='yolo_ncnn_node', name='yolo_node', output='screen',
             parameters=[{'model_param': model_param, 'model_bin': model_bin}]),

        # --- 신규: decision (상태머신 + PID) ---
        Node(package='decision', executable='decision_node', name='decision_node', output='screen',
             parameters=[{'course': course, 'race_dir': race_dir, 'skip_missions': skip_missions}]),

        # --- 키트: AUTO 모드 control ---
        # respawn: control_node 가 죽으면 모터가 조용히 멈추므로 자동 재시작한다.
        # (재시작 시 ESC 아밍 3초가 다시 돌지만 안전상 문제 없음.)
        Node(package='control', executable='control_node', name='control_node', output='screen',
             respawn=True, respawn_delay=1.0,
             parameters=[{'use_joystick_control': False, 'control_topic': '/control'}]),

        # --- 키트: E-STOP 안전용으로 조이스틱 유지 ---
        # 자율주행에선 조향/스로틀은 안 쓰고 X버튼 E-STOP만 쓰므로, 5Hz 디버그 로그
        # (SSH I/O 부하)는 끈다. 비상정지 기능은 그대로 살아있다.
        Node(package='joystick', executable='joystick_node', name='joystick_node', output='screen',
             parameters=[{'debug_log_enable': False}]),

        # --- 키트: 배터리 (모니터용 battery_status publish) ---
        Node(package='battery', executable='battery_node', name='battery_node', output='screen',
             respawn=True, respawn_delay=2.0),

        # --- 키트: 웹 모니터 (카메라 + 배터리 상태 표시) ---
        Node(package='monitor', executable='monitor_node', name='monitor_node', output='screen'),
    ])
