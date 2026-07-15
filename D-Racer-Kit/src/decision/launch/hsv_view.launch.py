"""Camera + HSV lane preprocessing + web dashboard, WITHOUT driving.

Brings up only the perception/monitor path so you can check the HSV
preprocessed image on the web dashboard. No control/decision/joystick
nodes -> the motor and steering never move (safe, no i2c actuation).

  ros2 launch decision hsv_view.launch.py

Then open the dashboard from your PC (SSH port-forward avoids hotspot isolation):
  ssh -L 5000:localhost:5000 topst@<board-ip>
  # browser -> http://localhost:5000   (see the "HSV Lane" panel)

Nodes:
  camera_node  -> /camera/image/compressed
  lane_node    -> /perception/lane/debug   (HSV mask overlay; monitor shows this)
  monitor_node -> web dashboard on WEB_HOST:WEB_PORT (vehicle_config.yaml)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # vehicle_config 미전달 시 camera_node 가 MIPI 기본값으로 폴백해 USB 카메라를
    # 못 열고 죽는다 (auto_race.launch.py 와 동일 처방). battery 는 확인용
    # 대시보드에 불필요 + respawn 소음이라 제외 (07-15 run80 백포트).
    vehicle_config = LaunchConfiguration('vehicle_config')
    return LaunchDescription([
        DeclareLaunchArgument('vehicle_config',
                              default_value='/home/topst/SEAME_2026_Hackathon-clone/D-Racer-Kit/'
                                            'src/config/vehicle_config.yaml'),
        Node(package='camera', executable='camera_node', name='camera_node', output='screen',
             parameters=[{'vehicle_config_file': vehicle_config}]),
        Node(package='perception', executable='lane_node', name='lane_node', output='screen'),
        Node(package='monitor', executable='monitor_node', name='monitor_node', output='screen'),
    ])
