"""decision_node: lane + aruco + YOLO 를 구독하고 상태머신을 돌려서
Control 을 /control 로 publish 한다 (키트의 control_node 가 이를 구동).

control_node 는 use_joystick_control:=False 로 실행해야 /control 을 구독한다.
"""

from collections import deque

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Header, String

from perception_msgs.msg import LaneState, ArucoState
from inference_msgs.msg import Detections
from control_msgs.msg import Control
from battery_msgs.msg import Battery

from decision.state_machine import RaceStateMachine


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        # ----- 토픽 / 루프 주기 -----
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('aruco_topic', '/perception/aruco')
        self.declare_parameter('detections_topic', '/inference/detections')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('fork_dir_topic', '/decision/fork_dir')  # 확정 방향을 perception 에 전달
        self.declare_parameter('state_topic', '/decision/state')        # 주행 모드 -> BEV 표시/모니터
        self.declare_parameter('control_hz', 30.0)

        # ----- 전략 -----
        self.declare_parameter('course', 'in')     # 'out' = S커브+갈림길, 'in' = 회전교차로
        self.declare_parameter('skip_missions', False)  # True = 순수 차선 추종 테스트 (미션 없음)

        # ----- 차선 PID + 조향 매핑 -----
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.15)
        self.declare_parameter('steer_center', 0.26)  # 쏠림(drift) 보정용 조향 bias (07-07 수동주행 실측: 0.26에서 정확히 직진)
        self.declare_parameter('steer_scale', -1.0)   # NEGATIVE: 트랙 검증 결과 조향 부호가 반대였음(오른쪽 치우침 offset<0 -> 더 왼쪽 조향해야 교정)
        # 좌우 조향 비대칭 보정: 차가 좌조향 시 실제로 덜 꺾여서, 좌조향(steer_center 위쪽
        # 편차)만 이 배율로 증폭한다. 1.0=off. 좌조향 = steer > steer_center (PID 부호 검증).
        self.declare_parameter('steer_left_gain', 1.0)  # 07-07 원 그리기 실측: center 0.26 기준
        # 좌/우 원 지름 동일(187cm) = 대칭 확인. 예전 "좌측 덜 꺾임"은 center 오프셋(0.2)
        # 증상이었음. center 재캘리 시 이 값도 원 테스트로 재확인할 것.
        # 급커브 feed-forward: DRIVE 에서 곡률에 비례한 조향 편향을 추가한다. 급좌커브에서
        # 가까운 왼선을 놓쳐도 curvature(밴드 간 차이)는 살아있어 미리·더 꺾게 해준다.
        # 0=off. 부호는 offset 과 동일 규약이라 target 에 그대로 더한다(좌커브 curvature<0).
        self.declare_parameter('curve_steer_bias', 0.0)

        # ----- throttle 단계 (키트의 set_throttle_percent 규약) -----
        # 07-10 실측(만충 8.2V, 바닥): ESC 는 1580us(0.16)에서 바퀴가 돌기 시작하지만
        # 차체 무게를 밀고 실제로 '전진'하는 임계는 1600us(0.20)이다. 배터리가 닳으면
        # 이 임계가 올라가므로 두 값 모두 임계 위로 여유를 두고 잡는다.
        #   throttle -> pulse:  1500us + throttle*500us  (neutral 1500, fwd 2000)
        # 07-10 바닥 실측(만충 8.22V). PCA9685 12bit/50Hz 라 1틱 = 4.88us = throttle 0.0098:
        #   정지->출발 임계(정지마찰): 0.20 (1602us, 틱328)
        #   굴러가는 중 유지 임계(운동마찰): 0.175 (1587us, 틱325) 로도 계속 주행
        # 배터리가 닳으면 두 임계 모두 올라간다 -> 반쯤 닳은 상태에서 재검증할 것.
        self.declare_parameter('drive_throttle', 0.20)   # 1602us: DRIVE 기본(=직선 최고 속도).
                                                         # curve_slow/steer_slow 가 여기서 깎는다.
        self.declare_parameter('slow_throttle', 0.175)   # 1587us: ROUNDABOUT 주행 + 감속 바닥.
                                                         # 유지는 되지만 정지에서 출발은 불가.
        self.declare_parameter('stop_throttle', 0.0)     # 1500us: 중립
        # 출발 킥: 정지마찰 때문에 slow_throttle(0.175)로는 정지 상태에서 출발할 수 없다.
        # 스로틀이 0에서 살아나는 모든 순간(초록불 출발, 장애물 해제 후 재출발, 정지선
        # 재출발)에만 잠시 킥 값으로 올렸다가 목표치로 되돌린다. 특히 링 안에서 장애물을
        # 만나 선 뒤 ROUNDABOUT(=slow_throttle)로 복귀할 때 못 나가는 것을 막는다. 0=off.
        self.declare_parameter('start_kick_throttle', 0.20)   # 출발 임계값
        self.declare_parameter('start_kick_s', 0.4)           # 킥 유지 시간(초)
        self.declare_parameter('curve_slow', 0.5)     # DRIVE: 커브에서 감속 (|curvature| 비례)
        # ----- look-ahead 보조 (기본 0 = 기존 동작 그대로) -----
        self.declare_parameter('max_steering_delta', 0.0)  # 틱당 조향 변화 상한; 0=off
        self.declare_parameter('steer_slow', 0.0)          # |조향|에 비례한 감속 게인; 0=off

        # ----- 전원 보호 (브라운아웃 완화) -----
        # 스로틀 상승 rate limit: 정지->출발 순간 스로틀이 한 틱에 점프하면 모터 돌입
        # 전류가 배터리 전압을 끌어내려 보드가 리셋/행에 빠진다. 상승만 제한하고 하강은
        # 즉시 반영한다(정지 명령은 절대 늦추면 안 됨). 30Hz 기준 0.02 => 0->0.20 에 ~0.33s.
        self.declare_parameter('max_throttle_delta', 0.02)   # 틱당 스로틀 상승 상한; 0=off
        # 저전압 가드: battery_node 의 퍼센트를 전압으로 역산해, 임계 아래면 스로틀을
        # slow_throttle 로 묶고(1단계), 더 떨어지면 정지(2단계)해 스스로를 보호한다.
        # 0 = off. 배터리 하한 6.4V, 만충 8.4V (battery_node 의 선형 매핑과 동일).
        self.declare_parameter('undervolt_slow_v', 0.0)      # 이 전압 미만 -> slow_throttle 상한
        self.declare_parameter('undervolt_stop_v', 0.0)      # 이 전압 미만 -> 정지
        self.declare_parameter('battery_topic', '/battery_status')
        # battery_node 의 선형 매핑 역산용. battery_node 의 min/max_voltage 와 일치해야 한다.
        self.declare_parameter('batt_min_v', 6.4)
        self.declare_parameter('batt_max_v', 8.4)

        # ----- 미션 튜닝 -----
        self.declare_parameter('conf_threshold', 0.5)  # YOLO 인식 confidence 문턱
        # 07-08 실측: 기동 직후 노출 안정화 전 프레임에서 순간 초록 오인식으로
        # 3프레임(0.1s)이 뚫려 조기 출발한 사례 -> 10프레임(0.33s) 연속 요구.
        # 진짜 초록불은 상시 점등이라 출발 지연 영향은 ~0.2s 뿐.
        self.declare_parameter('green_frames', 10)
        self.declare_parameter('red_frames', 3)
        # 빨간불 대기 무장: 주 조건은 장애물 미션 완료(obstacle_done, 코스 순서 고정).
        # finish_min_drive_s 는 아루코를 통째로 놓친 비상 주행용 예비 무장 —
        # 이 시간(초) 경과 시 장애물 미완이어도 빨간불 인식을 켠다.
        # 실측 코스 소요시간보다 여유 있게 길게 설정할 것. 신호등 bbox 최소 면적
        # (정규화 w*h, 0=off)도 함께 — 멀리 있는 작은 오검출 박스 필터.
        self.declare_parameter('finish_min_drive_s', 60.0)
        self.declare_parameter('light_min_area', 0.0)
        self.declare_parameter('marker_area_trigger', 0.02)
        self.declare_parameter('marker_clear_frames', 5)
        self.declare_parameter('fork_bias', 0.2)    # 브랜치 선택(perception)이 주역이라 진입 보조용으로 축소
        self.declare_parameter('fork_hold_s', 8.0)  # latch failsafe: fork 신호가 계속 켜져 있어도 이 시간 지나면 해제(초)
        # ----- 갈림길 표지판 방향 표결 (decision_node 로컬; on_dets 에서 사용) -----
        self.declare_parameter('fork_sign_min_conf', 0.50)  # 이 confidence 미만 표지판은 무시
        self.declare_parameter('fork_sign_vote_window', 5)  # 최근 detection 메시지 표결 창(int)
        self.declare_parameter('fork_sign_vote_min', 2)     # 같은 방향 이 개수 이상이면 확정(int)
        self.declare_parameter('fork_vote_clear_s', 1.0)    # 표지판 끊긴 뒤 표결 초기화까지(떨림방지창; 홀드 아님)

        # ----- 회전교차로 (junction 카운트, IMU 없음) -----
        self.declare_parameter('enter_curvature', 0.45)   # (미사용) 진입이 가로선 단독 트리거로 바뀜
        self.declare_parameter('enter_sustain_s', 0.3)    # 가로선이 이 시간 지속 보이면 -> 회전 진입
        # race_dir 마스터 (당일 설정): 'left' (CCW) 또는 'right' (CW). 여기서 회전교차로
        # turn_direction 을, 그리고 lane_node 의 junction_side 를 파생한다. 값 하나로
        # 방향 의존적인 모든 것이 뒤집힌다. 아래 turn_direction 은 race_dir 이
        # left/right 가 아닐 때만 쓰인다.
        self.declare_parameter('race_dir', 'left')
        self.declare_parameter('turn_direction', -1.0)    # +1 CW (우조향), -1 CCW
        # In/Out 브랜치 (자동, 방향 무관): 갈림길에서 In 코스는 노란 경로 쪽으로,
        # Out 은 반대쪽으로 조향. 노란색의 위치를 이용하므로 어느 쪽에 나타나든
        # 동작한다 -> 당일 설정 불필요.
        # FOLLOW-Y(노란 전용 추종)가 갈림길 선택을 담당하게 되면서 이 bias 는 중복이
        # 됐고, 노란 구간 주행 내내 ±branch_bias 타깃 지터/끌림을 만들어 기본 OFF.
        # 갈림길에서 FOLLOW-Y 전환이 늦는 게 실측되면 트랙에서 라이브로 켜서 보조.
        self.declare_parameter('branch_bias', 0.0)         # 0=off (07-08: FOLLOW-Y 로 대체)
        self.declare_parameter('branch_yellow_min', 0.03)  # yellow_ratio 가 이 값 이상 => 갈림길이 보임
        self.declare_parameter('circle_steer_bias', 0.225) # 링 유지, 조기 탈출 방지
        self.declare_parameter('min_loop_time_s', 3.0)    # 이보다 빨리 한 바퀴 완료 불가 (절대 하한)
        self.declare_parameter('max_loop_time_s', 20.0)   # 모든 추정치 실패 시 failsafe 탈출
        self.declare_parameter('crossline_cooldown_s', 2.0)  # 게이트 카운트 간 최소 간격 (재카운트 디바운스)
        # --- 탈출 failsafe 3-표결 (조향 적분 + 시간 + 가로선 재등장), IMU/마커 없음 ---
        self.declare_parameter('yaw_lap_threshold', 6.0)   # 한 바퀴당 조향 적분값; 트랙에서 캘리브레이션할 것
        self.declare_parameter('nominal_loop_time_s', 8.0) # 레이스 속도에서 실측한 한 바퀴 시간
        self.declare_parameter('lap_votes_needed', 2)      # {yaw, time, crossline} 중 탈출에 필요한 표 수
        # 노란색이 회전교차로 진입을 게이트한다 (회전교차로는 노란색, 외곽 루프는 흰색)
        self.declare_parameter('use_yellow_entry', True)
        self.declare_parameter('yellow_enter_ratio', 0.06)  # ROI 노란색 비율 => 회전교차로 안
        # --- 회전교차로 진입/탈출 = 갈림길과 동일한 브랜치 락온 ---
        # 진입: 가로선 첫 감지 -> RA on + 진입측 one-shot 락온(링 순환 방향).
        # 탈출: 링을 돌며 가로선 상승엣지를 세고, 도달 시 출구 브랜치로 락온.
        self.declare_parameter('roundabout_exit_gates', 1)   # 진입 후 가로선을 이 횟수 더 만나면 탈출
                                                             # (진입 가로선은 블랭크+재무장으로 제외; 1=한 바퀴)
        self.declare_parameter('roundabout_exit_side', '')   # 출구 브랜치 side; '' 이면 race_dir 파생
        self.declare_parameter('entry_lock_release_s', 2.0)  # 진입측 락온 강제 해제 시간 (one-shot, 짧게)
        # RA 진입 후 게이트 카운트 금지 시간(진입선 오카운트 방어). 길어서 손해는
        # "한 바퀴 더"뿐(과회전 허용), 짧으면 조기 탈출=실격 위험 -> 길게 잡는다.
        # 단, 실측 한 바퀴 시간보다는 반드시 짧아야 함 (트랙에서 라이브 조정).
        self.declare_parameter('gate_blank_s', 6.0)
        self.declare_parameter('gate_rearm_s', 0.5)          # 가로선이 이 시간 연속 OFF 여야 다음 카운트 무장

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
        # 진입측은 항상 그 반대(링 순환 방향) — 하드코딩 금지, 역방향 자동 반전.
        exit_side = str(g('roundabout_exit_side').value).lower()
        if exit_side not in ('left', 'right'):
            exit_side = 'right' if race_dir == 'left' else 'left'
        entry_side = 'left' if exit_side == 'right' else 'right'
        cfg = {
            'course': str(g('course').value),
            'skip_missions': bool(g('skip_missions').value),
            'race_dir': race_dir,
            'kp': float(g('kp').value), 'ki': float(g('ki').value), 'kd': float(g('kd').value),
            'steer_center': float(g('steer_center').value),
            'steer_scale': float(g('steer_scale').value),
            'steer_left_gain': float(g('steer_left_gain').value),
            'curve_steer_bias': float(g('curve_steer_bias').value),
            'drive_throttle': float(g('drive_throttle').value),
            'slow_throttle': float(g('slow_throttle').value),
            'stop_throttle': float(g('stop_throttle').value),
            'curve_slow': float(g('curve_slow').value),
            'max_steering_delta': float(g('max_steering_delta').value),
            'steer_slow': float(g('steer_slow').value),
            'max_throttle_delta': float(g('max_throttle_delta').value),
            'undervolt_slow_v': float(g('undervolt_slow_v').value),
            'undervolt_stop_v': float(g('undervolt_stop_v').value),
            'start_kick_throttle': float(g('start_kick_throttle').value),
            'start_kick_s': float(g('start_kick_s').value),
            'conf_threshold': float(g('conf_threshold').value),
            'green_frames': int(g('green_frames').value),
            'red_frames': int(g('red_frames').value),
            'finish_min_drive_s': float(g('finish_min_drive_s').value),
            'light_min_area': float(g('light_min_area').value),
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
            'min_loop_time_s': float(g('min_loop_time_s').value),
            'max_loop_time_s': float(g('max_loop_time_s').value),
            'crossline_cooldown_s': float(g('crossline_cooldown_s').value),
            'yaw_lap_threshold': float(g('yaw_lap_threshold').value),
            'nominal_loop_time_s': float(g('nominal_loop_time_s').value),
            'lap_votes_needed': int(g('lap_votes_needed').value),
            'use_yellow_entry': bool(g('use_yellow_entry').value),
            'yellow_enter_ratio': float(g('yellow_enter_ratio').value),
            'roundabout_exit_gates': int(g('roundabout_exit_gates').value),
            'roundabout_exit_side': exit_side,
            'roundabout_entry_side': entry_side,
            'entry_lock_release_s': float(g('entry_lock_release_s').value),
            'gate_blank_s': float(g('gate_blank_s').value),
            'gate_rearm_s': float(g('gate_rearm_s').value),
        }
        self.sm = RaceStateMachine(cfg)

        # ----- live-tunable params (ros2 param set 으로 주행 중 변경 가능) -----
        # state_machine 이 매 step 마다 self.sm.cfg 를 읽으므로 값만 갱신하면 즉시 반영됨.
        # state_machine 이 매 step 마다 self.sm.cfg 에서 읽는 값들만 라이브 변경 가능.
        # (steer_center/steer_scale 는 _lane_steer 에서 매번 cfg 를 읽으므로 즉시 반영됨)
        self.live_tunable = {
            'drive_throttle', 'slow_throttle', 'stop_throttle', 'curve_slow',
            'steer_center', 'steer_scale', 'steer_left_gain',
            'kp', 'ki', 'kd',   # PID 게인 (조향 세기) 트랙에서 라이브 튜닝
            'max_steering_delta', 'steer_slow',   # look-ahead 보조 (rate limit / 조향 감속)
            'fork_bias', 'fork_hold_s',           # 갈림길 편향 세기 / 유지 시간
            'branch_bias',                        # In/Out 색상 편향 (기본 0=off, 현장 재활성용)
            'roundabout_exit_gates',              # 회전교차로 탈출 게이트 카운트 (트랙 실측)
            'enter_sustain_s',                    # 진입 가로선 지속 debounce (트랙 실측)
            'entry_lock_release_s',               # 진입 락온 강제 해제 시간
            'gate_blank_s', 'gate_rearm_s',       # 게이트 블랭크 / 재무장 시간
            'crossline_cooldown_s',               # 게이트 카운트 간 최소 간격
            'finish_min_drive_s',                 # 빨간불 무시 최소 주행시간 (오인식 가드)
            'light_min_area',                     # 신호등 bbox 최소 면적 (오검출 필터)
            'conf_threshold',                     # YOLO confidence 문턱 (현장 조정)
            'circle_steer_bias',                  # 회전교차로 링 유지 편향 세기
            'curve_steer_bias',                   # DRIVE 급커브 feed-forward 편향
            'max_throttle_delta',                 # 스로틀 상승 rate limit (돌입전류 완화)
            'undervolt_slow_v', 'undervolt_stop_v',   # 저전압 가드 임계
            'start_kick_throttle', 'start_kick_s',    # 출발 킥 (정지마찰 극복)
        }
        self._prev_steer = None   # rate limit 용 직전 조향값
        self._prev_throttle = 0.0   # 스로틀 rate limit 용 직전 출력값
        self._kick_left = 0.0       # 출발 킥 남은 시간(초)
        self._battery_v = None      # 최근 배터리 전압 (None = 아직 수신 전 -> 가드 비활성)
        self._prev_state = None   # 상태 전환 INFO 로그용
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

        # 최신 입력값 (안전한 기본값)
        self.lane = LaneState()
        self.aruco = ArucoState()
        self.aruco.marker_id = -1
        self.dets = []

        self.create_subscription(LaneState, str(g('lane_topic').value), self.on_lane, 10)
        self.create_subscription(ArucoState, str(g('aruco_topic').value), self.on_aruco, 10)
        self.create_subscription(Detections, str(g('detections_topic').value), self.on_dets, 10)
        self.create_subscription(Battery, str(g('battery_topic').value), self.on_battery, 10)
        self.pub = self.create_publisher(Control, str(g('control_topic').value), 10)
        # 확정된 갈림길 방향을 perception(lane_node)에 전달 -> 브랜치 선택(guide 시드).
        self.fork_dir_pub = self.create_publisher(String, str(g('fork_dir_topic').value), 10)
        # 현재 주행 모드를 publish -> lane_node 가 BEV 디버그 화면에 표시.
        self.state_pub = self.create_publisher(String, str(g('state_topic').value), 10)

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

    def on_battery(self, msg):
        """battery_node 는 퍼센트만 발행하므로 선형 매핑을 역산해 전압으로 되돌린다.
        0%/100% 에서 clamp 되므로 batt_min_v 아래는 구분이 안 된다(가드 임계는 그 위로 잡을 것)."""
        lo = float(self.get_parameter('batt_min_v').value)
        hi = float(self.get_parameter('batt_max_v').value)
        self._battery_v = lo + (float(msg.battery_status) / 100.0) * (hi - lo)

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
        # 현재 주행 모드를 publish -> BEV 디버그 화면 표시.
        self.state_pub.publish(String(data=state))
        # 상태 전환은 INFO 로 남긴다 (디버그 레벨 없이도 전환 사유 추적 가능):
        # ROUNDABOUT 탈출이 게이트(gate)였는지 표결(exitV)이었는지 수치로 판별.
        if state != self._prev_state:
            self.get_logger().info(
                f'state {self._prev_state} -> {state} '
                f'(race_t={self.sm.race_t:.1f}s circle_t={self.sm.circle_t:.1f}s '
                f'gate={self.sm._gate_count} exitV={self.sm._exit_votes} '
                f'yaw={self.sm.yaw_proxy:.1f} latch={self.sm.turn_latch})')
            self._prev_state = state
        cfg = self.sm.cfg
        # 좌우 조향 비대칭 보정: 차가 좌조향 시 실제로 덜 꺾이므로, steer_center 위쪽
        # 편차(=좌조향)만 steer_left_gain 배로 증폭한다. rate limit 전에 적용.
        lg = float(cfg.get('steer_left_gain', 1.0))
        center = float(cfg['steer_center'])
        if lg != 1.0 and steer > center:
            steer = max(-1.0, min(1.0, center + (steer - center) * lg))
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

        # ----- 저전압 가드: 전압이 무너지면 스스로 부하를 줄인다 -----
        # 배터리 전압이 낮을 때 모터 돌입 전류가 겹치면 보드가 리셋/행에 빠진다.
        # 전압 미수신(None)이면 가드는 동작하지 않는다(배터리 노드 없이도 주행 가능).
        v = self._battery_v
        if v is not None:
            v_stop = float(cfg.get('undervolt_stop_v', 0.0))
            v_slow = float(cfg.get('undervolt_slow_v', 0.0))
            if v_stop > 0.0 and v < v_stop:
                throttle = float(cfg['stop_throttle'])
            elif v_slow > 0.0 and v < v_slow:
                throttle = min(throttle, float(cfg['slow_throttle']))

        # ----- 출발 킥: 정지 상태에서 출발하는 순간 정지마찰을 넘긴다 -----
        # 직전 출력이 0(정지)인데 이번에 스로틀이 살아나면 킥 타이머를 건다. 킥이 도는
        # 동안에는 목표치와 킥 값 중 큰 쪽을 쓴다(=목표가 이미 크면 무해). rate limit 전에
        # 적용해야 램프가 킥 값까지 올라간다.
        kick = float(cfg.get('start_kick_throttle', 0.0))
        kick_s = float(cfg.get('start_kick_s', 0.0))
        if kick > 0.0 and kick_s > 0.0:
            if throttle > 0.0 and self._prev_throttle <= 1e-3:
                self._kick_left = kick_s
            if self._kick_left > 0.0:
                if throttle > 0.0:
                    throttle = max(throttle, kick)
                    self._kick_left -= self.dt
                else:
                    self._kick_left = 0.0   # 다시 멈췄으면 킥 취소

        # ----- 스로틀 상승 rate limit: 돌입 전류 피크 억제 -----
        # 상승만 제한한다. 하강(감속/정지)은 안전 기능이므로 절대 늦추지 않는다.
        tdelta = float(cfg.get('max_throttle_delta', 0.0))
        if tdelta > 0.0 and throttle > self._prev_throttle:
            throttle = min(throttle, self._prev_throttle + tdelta)
        self._prev_throttle = throttle

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
