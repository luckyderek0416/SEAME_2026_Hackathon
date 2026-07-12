"""자율주행 레이싱 전체 스택을 기동한다.

  ros2 launch decision auto_race.launch.py course:=in
  ros2 launch decision auto_race.launch.py course:=in model_path:=/path/best.pt

키트의 camera_node 와 control_node 를 재사용하고 lane/aruco/yolo/decision 을 추가한다.
control_node 는 AUTO 모드로 동작한다 (use_joystick_control:=False -> /control 구독).
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_model(filename):
    """리포 안의 models/ 를 위쪽으로 올라가며 찾는다. 워크스페이스를 어디에 클론하든
    동작하게 하려는 것 — 예전엔 /home/topst/D-Racer/models 로 하드코딩되어 있어서
    다른 경로에 클론하면 YOLO 가 조용히 빈 detections 만 발행했다."""
    for base in Path(__file__).resolve().parents:
        candidate = base / 'models' / filename
        if candidate.exists():
            return str(candidate)
    return str(Path('/home/topst/D-Racer/models') / filename)   # 예전 기본값 폴백


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


def _battery_node(context, *args, **kwargs):
    """use_battery:=false 면 battery_node 를 아예 띄우지 않는다.

    07-10: auto_race 기동 직후 i2c-3 에서 'Arbitration lost' 가 초당 수십 회 쏟아지며
    콘솔 printk 폭주 -> 시스템 정지(대시보드/네트워크 멈춤)가 재현됐다. battery_node 는
    INA219(0x42)를 10Hz 로 읽고, control_node 는 PCA9685(0x40)에 20Hz 로 쓴다.
    배터리 표시는 주행에 불필요하므로(저전압 보호는 undervolt_*=0 으로 비활성) 원인
    분리 실험과 실주행에서 끌 수 있게 한다. respawn=True 라 i2c 오류 시 2초마다
    되살아나며 버스를 다시 여는 것도 폭주를 키운다.
    """
    val = LaunchConfiguration('use_battery').perform(context).strip().lower()
    if val in ('false', '0', 'no', 'off'):
        return []
    return [Node(package='battery', executable='battery_node', name='battery_node',
                 output='screen', respawn=True, respawn_delay=2.0)]


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
        DeclareLaunchArgument('aruco_dict', default_value='DICT_6X6_50'),
        DeclareLaunchArgument('aruco_inverted', default_value='true'),
        # 07-10: 배터리 표시는 주행에 불필요하고(저전압 보호는 undervolt_*=0 으로 비활성)
        # i2c-3 마스터를 하나 줄이려고 기본 OFF. 필요하면 use_battery:=true.
        DeclareLaunchArgument('use_battery', default_value='false',
                              description='true 면 battery_node 를 띄운다 (대시보드 배터리% 표시)'),
        DeclareLaunchArgument('model_param',
                              default_value=_find_model('model.ncnn.param')),
        DeclareLaunchArgument('model_bin',
                              default_value=_find_model('model.ncnn.bin')),
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
        # respawn: camera_node 가 죽으면 lane_node 가 이미지를 못 받아 주행 로직 전체가
        # 조용히 정지한다(decision 은 마지막 명령을 유지). 카메라는 상태가 없어 재시작이
        # 안전하므로 자동 복구시킨다. USB 재열거에 시간이 걸리므로 delay 2초.
        Node(package='camera', executable='camera_node', name='camera_node', output='screen',
             respawn=True, respawn_delay=2.0,
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

        # --- 키트: 배터리 (모니터용 battery_status publish; use_battery:=false 로 끔) ---
        OpaqueFunction(function=_battery_node),

        # --- 키트: 웹 모니터 (카메라 + 배터리 상태 표시) ---
        Node(package='monitor', executable='monitor_node', name='monitor_node', output='screen'),
    ])
