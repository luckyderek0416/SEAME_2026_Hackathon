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
  battery_node -> battery_status           (battery % on the dashboard)
  monitor_node -> web dashboard on WEB_HOST:WEB_PORT (vehicle_config.yaml)
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='camera', executable='camera_node', name='camera_node', output='screen'),
        Node(package='perception', executable='lane_node', name='lane_node', output='screen'),
        Node(package='battery', executable='battery_node', name='battery_node', output='screen'),
        Node(package='monitor', executable='monitor_node', name='monitor_node', output='screen'),
    ])
