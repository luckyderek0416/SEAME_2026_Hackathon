"""decision_node: subscribes to lane + aruco + YOLO, runs the state machine,
and publishes Control to /control (which the kit's control_node actuates).

Run control_node with use_joystick_control:=False so it listens to /control.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from perception_msgs.msg import LaneState, ArucoState
from inference_msgs.msg import Detections
from control_msgs.msg import Control

from decision.state_machine import RaceStateMachine


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        # ----- topics / loop rate -----
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('aruco_topic', '/perception/aruco')
        self.declare_parameter('detections_topic', '/inference/detections')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('control_hz', 30.0)

        # ----- strategy -----
        self.declare_parameter('course', 'out')     # 'out' = S-curve+fork, 'in' = roundabout

        # ----- lane PID + steering mapping -----
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.15)
        self.declare_parameter('steer_center', 0.0)  # = STEER_TRIM from vehicle_config.yaml
        self.declare_parameter('steer_scale', 1.0)   # set NEGATIVE if steering is inverted

        # ----- throttle levels (kit's set_throttle_percent convention) -----
        self.declare_parameter('drive_throttle', 0.18)
        self.declare_parameter('slow_throttle', 0.12)
        self.declare_parameter('stop_throttle', 0.0)

        # ----- mission tuning -----
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('green_frames', 3)
        self.declare_parameter('red_frames', 3)
        self.declare_parameter('marker_area_trigger', 0.02)
        self.declare_parameter('marker_clear_frames', 5)
        self.declare_parameter('fork_bias', 0.4)
        self.declare_parameter('roundabout_seconds', 8.0)

        g = self.get_parameter
        cfg = {
            'course': str(g('course').value),
            'kp': float(g('kp').value), 'ki': float(g('ki').value), 'kd': float(g('kd').value),
            'steer_center': float(g('steer_center').value),
            'steer_scale': float(g('steer_scale').value),
            'drive_throttle': float(g('drive_throttle').value),
            'slow_throttle': float(g('slow_throttle').value),
            'stop_throttle': float(g('stop_throttle').value),
            'conf_threshold': float(g('conf_threshold').value),
            'green_frames': int(g('green_frames').value),
            'red_frames': int(g('red_frames').value),
            'marker_area_trigger': float(g('marker_area_trigger').value),
            'marker_clear_frames': int(g('marker_clear_frames').value),
            'fork_bias': float(g('fork_bias').value),
            'roundabout_seconds': float(g('roundabout_seconds').value),
        }
        self.sm = RaceStateMachine(cfg)

        # latest inputs (safe defaults)
        self.lane = LaneState()
        self.aruco = ArucoState()
        self.aruco.marker_id = -1
        self.dets = []

        self.create_subscription(LaneState, str(g('lane_topic').value), self.on_lane, 10)
        self.create_subscription(ArucoState, str(g('aruco_topic').value), self.on_aruco, 10)
        self.create_subscription(Detections, str(g('detections_topic').value), self.on_dets, 10)
        self.pub = self.create_publisher(Control, str(g('control_topic').value), 10)

        self.dt = 1.0 / float(g('control_hz').value)
        self.timer = self.create_timer(self.dt, self.on_timer)
        self.get_logger().info(f"decision_node up. course={cfg['course']}")

    def on_lane(self, msg):
        self.lane = msg

    def on_aruco(self, msg):
        self.aruco = msg

    def on_dets(self, msg):
        self.dets = list(msg.detections)

    def on_timer(self):
        steer, throttle, state = self.sm.step(self.lane, self.aruco, self.dets, self.dt)
        out = Control()
        out.header = Header()
        out.header.stamp = self.get_clock().now().to_msg()
        out.steering = float(steer)
        out.throttle = float(throttle)
        self.pub.publish(out)
        self.get_logger().debug(f'[{state}] steer={steer:.3f} thr={throttle:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
