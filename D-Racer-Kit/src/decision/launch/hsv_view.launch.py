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
(07-15: battery_node 제외 — i2c 폭주 이력 + 캘리에 불필요. joystick 은 원래 없음.)
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # vehicle_config 명시 전달 (auto_race 와 동일 이유): 없으면 camera_node 가
        # 조용히 MIPI 기본값으로 떨어져 USB 카메라에서 죽는다 (07-15 run80 백포트).
        Node(package='camera', executable='camera_node', name='camera_node', output='screen',
             respawn=True, respawn_delay=2.0,
             parameters=[{'vehicle_config_file':
                          '/home/topst/SEAME_2026_Hackathon-clone/D-Racer-Kit/'
                          'src/config/vehicle_config.yaml'}]),
        Node(package='perception', executable='lane_node', name='lane_node', output='screen'),
        Node(package='monitor', executable='monitor_node', name='monitor_node', output='screen'),
    ])
