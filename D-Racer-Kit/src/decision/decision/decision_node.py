"""decision_node: subscribes to lane + aruco + YOLO, runs the state machine,
and publishes Control to /control (which the kit's control_node actuates).

Run control_node with use_joystick_control:=False so it listens to /control.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
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
        self.declare_parameter('skip_missions', False)  # True = pure lane-following test (no missions)

        # ----- lane PID + steering mapping -----
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.15)
        self.declare_parameter('steer_center', 0.2)  # steering bias to correct drift
        self.declare_parameter('steer_scale', 1.0)   # set NEGATIVE if steering is inverted

        # ----- throttle levels (kit's set_throttle_percent convention) -----
        self.declare_parameter('drive_throttle', 0.2)
        self.declare_parameter('slow_throttle', 0.16)
        self.declare_parameter('stop_throttle', 0.0)
        self.declare_parameter('curve_slow', 0.5)     # DRIVE: slow on curves (per |curvature|)

        # ----- mission tuning -----
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('green_frames', 3)
        self.declare_parameter('red_frames', 3)
        self.declare_parameter('marker_area_trigger', 0.02)
        self.declare_parameter('marker_clear_frames', 5)
        self.declare_parameter('fork_bias', 0.4)

        # ----- roundabout (junction-count, no IMU) -----
        self.declare_parameter('enter_curvature', 0.45)   # |lane offset| above this ...
        self.declare_parameter('enter_sustain_s', 0.6)    # ... for this long -> enter circle
        # race_dir MASTER (set on the day): 'left' (CCW) or 'right' (CW). Derives the
        # roundabout turn_direction here AND lane_node's junction_side. One value flips
        # everything direction-dependent. turn_direction below is used only if race_dir
        # is not left/right.
        self.declare_parameter('race_dir', 'left')
        self.declare_parameter('turn_direction', -1.0)    # +1 CW (steer right), -1 CCW
        # In/Out branch (AUTO, direction-agnostic): at the fork, steer toward the yellow
        # path for the In course / away from it for Out. Uses where the yellow is, so it
        # works whichever side it appears -> no day setup needed.
        self.declare_parameter('branch_bias', 0.35)        # steer bias at the fork
        self.declare_parameter('branch_yellow_min', 0.03)  # yellow_ratio above this => fork in view
        self.declare_parameter('circle_steer_bias', 0.15) # hold the ring, don't exit early
        self.declare_parameter('target_loops', 1)
        self.declare_parameter('min_loop_time_s', 3.0)    # cannot finish a lap faster (HARD floor)
        self.declare_parameter('max_loop_time_s', 20.0)   # failsafe exit if all estimates fail
        self.declare_parameter('junction_cooldown_s', 2.0)
        # --- lap voting (junction + steering-integral + time), no IMU/marker ---
        self.declare_parameter('yaw_lap_threshold', 6.0)   # steering-integral per lap; CALIBRATE on track
        self.declare_parameter('nominal_loop_time_s', 8.0) # measured one-lap time at race speed
        self.declare_parameter('lap_votes_needed', 2)      # of {junction, yaw, time} to exit
        # yellow gates roundabout entry (the roundabout is yellow; the outer loop is white)
        self.declare_parameter('use_yellow_entry', True)
        self.declare_parameter('yellow_enter_ratio', 0.06)  # ROI yellow fraction => in roundabout

        g = self.get_parameter
        race_dir = str(g('race_dir').value).lower()
        if race_dir == 'right':
            turn_direction = 1.0
        elif race_dir == 'left':
            turn_direction = -1.0
        else:
            turn_direction = float(g('turn_direction').value)
        cfg = {
            'course': str(g('course').value),
            'skip_missions': bool(g('skip_missions').value),
            'race_dir': race_dir,
            'kp': float(g('kp').value), 'ki': float(g('ki').value), 'kd': float(g('kd').value),
            'steer_center': float(g('steer_center').value),
            'steer_scale': float(g('steer_scale').value),
            'drive_throttle': float(g('drive_throttle').value),
            'slow_throttle': float(g('slow_throttle').value),
            'stop_throttle': float(g('stop_throttle').value),
            'curve_slow': float(g('curve_slow').value),
            'conf_threshold': float(g('conf_threshold').value),
            'green_frames': int(g('green_frames').value),
            'red_frames': int(g('red_frames').value),
            'marker_area_trigger': float(g('marker_area_trigger').value),
            'marker_clear_frames': int(g('marker_clear_frames').value),
            'fork_bias': float(g('fork_bias').value),
            'enter_curvature': float(g('enter_curvature').value),
            'enter_sustain_s': float(g('enter_sustain_s').value),
            'turn_direction': turn_direction,
            'branch_bias': float(g('branch_bias').value),
            'branch_yellow_min': float(g('branch_yellow_min').value),
            'circle_steer_bias': float(g('circle_steer_bias').value),
            'target_loops': int(g('target_loops').value),
            'min_loop_time_s': float(g('min_loop_time_s').value),
            'max_loop_time_s': float(g('max_loop_time_s').value),
            'junction_cooldown_s': float(g('junction_cooldown_s').value),
            'yaw_lap_threshold': float(g('yaw_lap_threshold').value),
            'nominal_loop_time_s': float(g('nominal_loop_time_s').value),
            'lap_votes_needed': int(g('lap_votes_needed').value),
            'use_yellow_entry': bool(g('use_yellow_entry').value),
            'yellow_enter_ratio': float(g('yellow_enter_ratio').value),
        }
        self.sm = RaceStateMachine(cfg)

        # ----- live-tunable params (ros2 param set 으로 주행 중 변경 가능) -----
        # state_machine 이 매 step 마다 self.sm.cfg 를 읽으므로 값만 갱신하면 즉시 반영됨.
        self.live_tunable = {
            'drive_throttle', 'slow_throttle', 'stop_throttle', 'curve_slow',
        }
        self.add_on_set_parameters_callback(self._on_set_parameters)

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

    def _on_set_parameters(self, params):
        """ros2 param set 요청을 받아 state_machine 설정을 라이브로 갱신."""
        for p in params:
            if p.name in self.live_tunable and p.name in self.sm.cfg:
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} 는 숫자여야 합니다',
                    )
                self.sm.cfg[p.name] = value
                self.get_logger().info(f'[live] {p.name} -> {value:g}')
        return SetParametersResult(successful=True)

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
