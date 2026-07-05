"""decision_node: subscribes to lane + aruco + YOLO, runs the state machine,
and publishes Control to /control (which the kit's control_node actuates).

Run control_node with use_joystick_control:=False so it listens to /control.
"""

from collections import deque

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Header, String

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
        self.declare_parameter('fork_dir_topic', '/decision/fork_dir')  # 확정 방향을 perception 에 전달
        self.declare_parameter('control_hz', 30.0)

        # ----- strategy -----
        self.declare_parameter('course', 'in')     # 'out' = S-curve+fork, 'in' = roundabout
        self.declare_parameter('skip_missions', False)  # True = pure lane-following test (no missions)

        # ----- lane PID + steering mapping -----
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.15)
        self.declare_parameter('steer_center', 0.2)  # steering bias to correct drift
        self.declare_parameter('steer_scale', -1.0)   # NEGATIVE: 트랙 검증 결과 조향 부호가 반대였음(오른쪽 치우침 offset<0 -> 더 왼쪽 조향해야 교정)

        # ----- throttle levels (kit's set_throttle_percent convention) -----
        self.declare_parameter('drive_throttle', 0.20)
        self.declare_parameter('slow_throttle', 0.12)
        self.declare_parameter('stop_throttle', 0.0)
        self.declare_parameter('curve_slow', 0.5)     # DRIVE: slow on curves (per |curvature|)
        # ----- look-ahead 보조 (기본 0 = 기존 동작 그대로) -----
        self.declare_parameter('max_steering_delta', 0.0)  # 틱당 조향 변화 상한; 0=off
        self.declare_parameter('steer_slow', 0.0)          # |조향|에 비례한 감속 게인; 0=off

        # ----- mission tuning -----
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('green_frames', 3)
        self.declare_parameter('red_frames', 3)
        self.declare_parameter('marker_area_trigger', 0.02)
        self.declare_parameter('marker_clear_frames', 5)
        self.declare_parameter('fork_bias', 0.2)    # 브랜치 선택(perception)이 주역이라 진입 보조용으로 축소
        self.declare_parameter('fork_hold_s', 8.0)  # latch failsafe: fork 신호가 계속 켜져 있어도 이 시간 지나면 해제(초)
        # ----- 갈림길 표지판 방향 표결 (decision_node 로컬; on_dets 에서 사용) -----
        self.declare_parameter('fork_sign_min_conf', 0.50)  # 이 confidence 미만 표지판은 무시
        self.declare_parameter('fork_sign_vote_window', 5)  # 최근 detection 메시지 표결 창(int)
        self.declare_parameter('fork_sign_vote_min', 2)     # 같은 방향 이 개수 이상이면 확정(int)
        self.declare_parameter('fork_vote_clear_s', 1.0)    # 표지판 끊긴 뒤 표결 초기화까지(떨림방지창; 홀드 아님)

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
        # --- 회전교차로 탈출 = 갈림길과 동일한 브랜치 락온 ---
        # 링을 돌며 게이트(노란 가로선 or 점선 개구부)의 상승엣지를 세고, 이 횟수에
        # 도달하면 출구 브랜치로 락온해 명시적으로 빠져나간다(암묵적 바이어스 해제 대신).
        self.declare_parameter('roundabout_exit_gates', 2)   # 게이트 통과 이 횟수면 탈출(엣지 카운트)
        self.declare_parameter('roundabout_exit_side', '')   # 출구 브랜치 side; '' 이면 race_dir 파생

        g = self.get_parameter
        race_dir = str(g('race_dir').value).lower()
        if race_dir == 'right':
            turn_direction = 1.0
        elif race_dir == 'left':
            turn_direction = -1.0
        else:
            turn_direction = float(g('turn_direction').value)
        # 회전교차로 출구 브랜치 side: 명시값 없으면 race_dir 에서 파생(junction_side 와 동일
        # 원리). 정방향(race_dir=left/CCW)=출구 오른쪽, 역방향(right/CW)=출구 왼쪽.
        exit_side = str(g('roundabout_exit_side').value).lower()
        if exit_side not in ('left', 'right'):
            exit_side = 'right' if race_dir == 'left' else 'left'
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
            'max_steering_delta': float(g('max_steering_delta').value),
            'steer_slow': float(g('steer_slow').value),
            'conf_threshold': float(g('conf_threshold').value),
            'green_frames': int(g('green_frames').value),
            'red_frames': int(g('red_frames').value),
            'marker_area_trigger': float(g('marker_area_trigger').value),
            'marker_clear_frames': int(g('marker_clear_frames').value),
            'fork_bias': float(g('fork_bias').value),
            'fork_hold_s': float(g('fork_hold_s').value),
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
            'roundabout_exit_gates': int(g('roundabout_exit_gates').value),
            'roundabout_exit_side': exit_side,
        }
        self.sm = RaceStateMachine(cfg)

        # ----- live-tunable params (ros2 param set 으로 주행 중 변경 가능) -----
        # state_machine 이 매 step 마다 self.sm.cfg 를 읽으므로 값만 갱신하면 즉시 반영됨.
        # state_machine 이 매 step 마다 self.sm.cfg 에서 읽는 값들만 라이브 변경 가능.
        # (steer_center/steer_scale 는 _lane_steer 에서 매번 cfg 를 읽으므로 즉시 반영됨)
        self.live_tunable = {
            'drive_throttle', 'slow_throttle', 'stop_throttle', 'curve_slow',
            'steer_center', 'steer_scale',
            'kp', 'ki', 'kd',   # PID 게인 (조향 세기) 트랙에서 라이브 튜닝
            'max_steering_delta', 'steer_slow',   # look-ahead 보조 (rate limit / 조향 감속)
            'fork_bias', 'fork_hold_s',           # 갈림길 편향 세기 / 유지 시간
            'roundabout_exit_gates',              # 회전교차로 탈출 게이트 카운트 (트랙 실측)
        }
        self._prev_steer = None   # rate limit 용 직전 조향값
        # 갈림길 표지판 표결 상태 (decision_node 로컬)
        self.fork_sign_min_conf = float(g('fork_sign_min_conf').value)
        self.fork_sign_vote_window = int(g('fork_sign_vote_window').value)
        self.fork_sign_vote_min = int(g('fork_sign_vote_min').value)
        self.fork_vote_clear_s = float(g('fork_vote_clear_s').value)
        self._fork_votes = deque(maxlen=self.fork_sign_vote_window)
        self.last_fork_sign_time = None
        self.fork_sign_confidence = 0.0
        self._last_fork_dir = None            # 이번 프레임 감지된 방향(디버그)
        self.confirmed_fork_direction = None  # 표결로 확정된 방향
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
        # 확정된 갈림길 방향을 perception(lane_node)에 전달 -> 브랜치 선택(guide 시드).
        self.fork_dir_pub = self.create_publisher(String, str(g('fork_dir_topic').value), 10)

        self.dt = 1.0 / float(g('control_hz').value)
        self.timer = self.create_timer(self.dt, self.on_timer)
        self.get_logger().info(f"decision_node up. course={cfg['course']}")

    def _on_set_parameters(self, params):
        """ros2 param set 요청을 받아 state_machine 설정을 라이브로 갱신."""
        for p in params:
            # 갈림길 표결 파라미터는 sm.cfg 가 아니라 decision_node 로컬이라 별도 처리.
            if p.name == 'fork_sign_min_conf':
                self.fork_sign_min_conf = float(p.value)
                self.get_logger().info(f'[live] fork_sign_min_conf -> {self.fork_sign_min_conf:g}')
                continue
            if p.name == 'fork_sign_vote_min':
                self.fork_sign_vote_min = int(p.value)
                self.get_logger().info(f'[live] fork_sign_vote_min -> {self.fork_sign_vote_min}')
                continue
            if p.name == 'fork_vote_clear_s':
                self.fork_vote_clear_s = float(p.value)
                self.get_logger().info(f'[live] fork_vote_clear_s -> {self.fork_vote_clear_s:g}')
                continue
            if p.name == 'fork_sign_vote_window':
                self.fork_sign_vote_window = int(p.value)
                self._fork_votes = deque(self._fork_votes, maxlen=self.fork_sign_vote_window)
                self.get_logger().info(f'[live] fork_sign_vote_window -> {self.fork_sign_vote_window}')
                continue
            if p.name in self.live_tunable and p.name in self.sm.cfg:
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} 는 숫자여야 합니다',
                    )
                self.sm.cfg[p.name] = value
                # PID 게인은 state_machine 의 PID 객체에도 반영해야 즉시 적용됨.
                if p.name in ('kp', 'ki', 'kd'):
                    setattr(self.sm.pid, p.name, value)
                self.get_logger().info(f'[live] {p.name} -> {value:g}')
        return SetParametersResult(successful=True)

    def on_lane(self, msg):
        self.lane = msg

    def on_aruco(self, msg):
        self.aruco = msg

    def on_dets(self, msg):
        self.dets = list(msg.detections)
        self._vote_fork_sign(msg)

    def _vote_fork_sign(self, msg):
        """갈림길 표지판 방향 표결 (메시지당 1표). 한 프레임 오검출로 방향을 확정하지
        않도록, 최근 fork_sign_vote_window 개 detection 메시지에서 left/right 를 누적해
        fork_sign_vote_min 이상인 방향만 confirmed_fork_direction 으로 확정한다.
        step() 이 30Hz 로 도는 것과 무관하게, 표는 detection 메시지가 올 때만 쌓인다."""
        best = None  # (direction, confidence)
        for d in msg.detections:
            if d.label in ('left_sign', 'right_sign') and d.confidence >= self.fork_sign_min_conf:
                direction = 'left' if d.label == 'left_sign' else 'right'
                if best is None or d.confidence > best[1]:
                    best = (direction, d.confidence)
        self._last_fork_dir = best[0] if best else None
        if best is not None:
            self._fork_votes.append(best[0])
            self.last_fork_sign_time = self.get_clock().now()
            self.fork_sign_confidence = best[1]

        # 표지판이 fork_vote_clear_s 이상 안 들어오면 표결 초기화(오래된 표는 버린다).
        # 이건 '떨림 방지창'일 뿐, 갈림길 통과까지의 홀드는 state_machine 이 lane.fork
        # (도로 재수렴) 기준으로 단독 담당한다 => 예전 홀드 이중적용(≈2×) 제거.
        if self.last_fork_sign_time is not None:
            age = (self.get_clock().now() - self.last_fork_sign_time).nanoseconds / 1e9
            if age >= self.fork_vote_clear_s:
                self._fork_votes.clear()
                self.confirmed_fork_direction = None
                self.fork_sign_confidence = 0.0
                return

        left = self._fork_votes.count('left')
        right = self._fork_votes.count('right')
        if max(left, right) >= self.fork_sign_vote_min:
            if left > right:
                self.confirmed_fork_direction = 'left'
            elif right > left:
                self.confirmed_fork_direction = 'right'
            else:                                   # 동수 -> 최근 표
                self.confirmed_fork_direction = self._fork_votes[-1]

    def on_timer(self):
        # 표결로 확정된 갈림길 방향을 상태머신에 전달(step 전). None 이면 무편향.
        self.sm.confirmed_fork_direction = self.confirmed_fork_direction
        steer, throttle, state = self.sm.step(self.lane, self.aruco, self.dets, self.dt)
        # 래치된 방향을 perception 에 publish -> lane_node 가 브랜치 선택(guide 시드)에 사용.
        # 표지판이 시야에서 사라진 뒤에도 latch 가 유지되는 동안(갈림길 통과 전) 계속
        # 그 브랜치를 좇게 하려고, confirmed 가 아니라 sm.turn_latch 를 보낸다.
        self.fork_dir_pub.publish(String(data=self.sm.turn_latch or ''))
        cfg = self.sm.cfg
        # 조향 rate limit: 틱당 변화량 제한으로 프레임 간 조향 급변(튐)을 막는다.
        # 0 이면 완전 비활성(기존 동작). roundabout 의 yaw 적분은 제한 전 값을 쓰지만
        # 편향이 '늦게 나가기' 쪽이라 랩 카운트 안전에는 문제 없음.
        delta = float(cfg.get('max_steering_delta', 0.0))
        if delta > 0.0 and self._prev_steer is not None:
            steer = max(self._prev_steer - delta, min(self._prev_steer + delta, steer))
        self._prev_steer = steer
        # 조향이 클수록 감속(steer_slow). curvature 기반 감속(curve_slow)과 별개로,
        # 실제 출력 조향각 기준으로 한 번 더 줄인다. slow_throttle 아래로는 안 내림.
        ss = float(cfg.get('steer_slow', 0.0))
        if ss > 0.0 and throttle > 0.0:
            cut = max(0.0, 1.0 - ss * min(1.0, abs(steer - float(cfg['steer_center']))))
            throttle = max(min(throttle, float(cfg['slow_throttle'])), throttle * cut)
        out = Control()
        out.header = Header()
        out.header.stamp = self.get_clock().now().to_msg()
        out.steering = float(steer)
        out.throttle = float(throttle)
        self.pub.publish(out)
        self.get_logger().debug(
            f'[{state}] steer={steer:.3f} thr={throttle:.3f} '
            f'off={self.lane.offset:+.3f} curv={self.lane.curvature:+.3f} '
            f'xline={int(self.lane.yellow_crossline)} junc={int(self.lane.junction)} '
            f'fork={int(self.lane.fork)} '
            f'yr={self.lane.yellow_ratio:.2f} '
            f'entryV={self.sm._entry_votes} exitV={self.sm._exit_votes} gate={self.sm._gate_count} '
            f'forkDet={self._last_fork_dir} forkFix={self.confirmed_fork_direction} '
            f'forkConf={self.fork_sign_confidence:.2f} latch={self.sm.turn_latch}'
        )


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
