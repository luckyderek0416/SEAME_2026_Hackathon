"""decision_node: lane + aruco + YOLO 를 구독하고 상태머신을 돌려서
Control 을 /control 로 publish 한다 (키트의 control_node 가 이를 구동).

control_node 는 use_joystick_control:=False 로 실행해야 /control 을 구독한다.
"""

from collections import deque

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, Header, String

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
        self.declare_parameter('drive_throttle', 0.19)   # 흰 구간 순항 (07-11 오후: 팩 열화로 전 구간 +0.01)   # 1602us: DRIVE 기본(=직선 최고 속도).
                                                         # curve_slow/steer_slow 가 여기서 깎는다.
        self.declare_parameter('slow_throttle', 0.165)
        # 노란 구간(DRIVE[Y]) 전용 상한: 접근/갈림길에서 저속·정밀 주행 (0=기능 off).
        # 07-12 run47 실증 + 사용자 확정: 노란 구간은 0.165 고정. (0.17 이던 시절에도
        # 커브 감속이 하한 0.165 로 깎아 링에선 사실상 0.165 였음 — 카메라 재캘리 후
        # 곡률 추정이 정확해지며 상시 하한 도달. 변동 요소를 없애고 상수로 못 박는다.)
        self.declare_parameter('yellow_drive_throttle', 0.165)
        self.declare_parameter('yellow_slow_ratio', 0.03)   # 노란 구간 판정 문턱 (FOLLOW-Y 와 동일 값 유지 — 07-11 run8 후 0.03 복원에 동기화)   # 1587us: ROUNDABOUT 주행 + 감속 바닥.
                                                         # 유지는 되지만 정지에서 출발은 불가.
        self.declare_parameter('stop_throttle', 0.0)     # 1500us: 중립
        # 출발 킥: 스로틀이 0에서 살아나는 모든 순간(초록불 출발, ArUco 장애물 해제 후
        # 재출발)에만 잠시 킥 값까지 올렸다가 목표치로 되돌린다.
        # 킥 값은 정지->출발 임계(0.20)보다 확실히 위여야 한다. 정지에서 출발할 때의 목표는
        # 항상 drive_throttle(0.20)인데 그게 곧 임계값이라 여유가 0이고, 배터리가 닳으면
        # 임계가 올라가 아예 못 나간다. 0.24(1620us)는 실측상 확실히 출발하는 값이다.
        # (순항 속도로는 빠르지만 0.4s 만 유지하므로 무해)  0 = off.
        self.declare_parameter('start_kick_throttle', 0.23)   # 1620us: 임계(0.20)보다 확실히 위
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
        # 빨간 도로(ArUco 장애물 구간) 감지 임계. ROI 중 빨간 픽셀 비율이 이 값 이상이면
        # (1) DRIVE 스로틀을 slow_throttle 로 묶어 미리 감속하고,
        # (2) 그 구간 안에서는 marker_area_trigger 를 무시하고 마커가 보이는 즉시 정지한다.
        # 멀리서 빨간 도로가 '보이기 시작하는' 수준으로 낮게 잡을 것. 0 = 기능 off.
        self.declare_parameter('red_slow_ratio', 0.05)
        self.declare_parameter('marker_area_trigger', 0.02)
        self.declare_parameter('marker_clear_frames', 8)   # run80: 정지중 검출 깜빡임 흡수 + 제거 후 ~0.8s 재출발
        self.declare_parameter('fork_bias', 0.2)    # (미사용 — 07-11 방안1: 조향 경로에서 제거, 탈출 락으로 대체)
        self.declare_parameter('fork_hold_s', 8.0)  # latch failsafe: fork 신호가 계속 켜져 있어도 이 시간 지나면 해제(초)
        # ----- 갈림길 표지판 방향 표결 (decision_node 로컬; on_dets 에서 사용) -----
        self.declare_parameter('fork_sign_min_conf', 0.50)  # 이 confidence 미만 표지판은 무시
        self.declare_parameter('fork_sign_vote_window', 5)  # 최근 detection 메시지 표결 창(int)
        self.declare_parameter('fork_sign_vote_min', 2)     # 같은 방향 이 개수 이상이면 확정(int)
        self.declare_parameter('fork_vote_clear_s', 1.0)    # 표지판 끊긴 뒤 표결 초기화까지(떨림방지창; 홀드 아님)

        # ----- 회전교차로 (junction 카운트, IMU 없음) -----
        self.declare_parameter('enter_sustain_s', 0.2)
        self.declare_parameter('ra_min_drive_s', 7.5)    # 출발 후 이 시간 전엔 RA 진입 금지 (입구 오진입 차단)
                                                         # 07-11 run12: 1L 진입 성공으로 진짜 정지선 도착이 9.3s 로
                                                         # 당겨져 10.0 이 진짜 진입을 차단 -> 7.5 (가짜 5.9s +1.6 여유,
                                                         # 진짜 9.3s -1.8 여유)    # 가로선이 이 시간 지속 보이면 -> 회전 진입
        # 곡률 게이트 (0=off): 새 정지선 검출기(Hough+직교)가 차선 오인을 원리적으로
        # 차단해 불필요 -> 0. (0.5 게이트는 곡선 위 진짜 정지선 접근까지 막았었음)
        self.declare_parameter('enter_max_curvature', 0.0)
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
        # 부호 규약: 음수 = 링 안쪽(좌). PID 이후 가산이라 steer_scale 반전에 안 뒤집힘
        # (실측 수정 완료). run13: 상시 +0.13 가산이 중앙 섬 진입 유발 -> 0 으로.
        # 링 '바깥' 이탈 재발 시 -0.05 부터 소량 재도입 (부호 반전 금지).
        self.declare_parameter('circle_steer_bias', 0.0)  # 링 유지 편향 (0=off)
        # 재획득 규칙 무장 지연 (07-12): RA 진입 후 이 시간이 지나면 perception 의
        # "nl 0->1 재획득 = 우측(바깥) 경계" 분류를 무장 (/decision/merge_zone).
        # 실측 (런19~26, RA 에피소드 8개): 진입 잔재 전이는 전부 <2.6s, 합류부 0->1 은
        # 최속 가정에도 ~14s 이후 -> 5s 는 양쪽 모두에 큰 여유. 상한 없음 (탈출 시 자동 해제).
        self.declare_parameter('reacq_arm_s', 5.0)
        # RA 맹목 폴백 (07-11 run21): RA 중 차선 소실 프레임의 조향을 직진 대신
        # 링 유지 호로. 진입 락과 중복 가산 안 함. 음수 = 안쪽(좌).
        self.declare_parameter('ra_blind_bias', -0.15)
        # 골든런 실측: 랩 19.7s, 진입 정지선 군집 3.9s, 가짜 0건. 자율 1차에서 8.6s
        # 링 안 직각 실선 오인 조기 탈출 -> 17s 까지 블랭크 (진짜 탈출 ~20s+ 무지장).
        # 주의: 시간 기반이라 속도 밴드에 민감 — run55 고속 랩(20.5s)에선 랩의 90%.
        self.declare_parameter('min_loop_time_s', 17.0)   # 이보다 빨리 한 바퀴 완료 불가 (절대 하한)
        # 속도 스케일 기준: G->RA 접근 소요시간이 이 값일 때 스케일 1.0 (run54=20.5s 밴드,
        # 절대 임계 3.6/18.5 가 실측 정합했던 밴드). 시간/적분 임계 전부에 s 를 곱한다 —
        # run53(16.8s)/54(20.5)/55(15.0) 오프라인 재검산 전부 가짜 차단+진짜 통과.
        self.declare_parameter('ra_ref_drive_s', 20.5)
        # 스로틀 동적 보정 (G->Y래치 소요시간 -> 순항 스로틀 가산). gain 0=off.
        self.declare_parameter('throttle_ref_latch_s', 4.6)   # 적정 속도 기준 (run53/54/59 실측 4.4~4.7)
        self.declare_parameter('throttle_adapt_gain', 0.06)   # 보정 = gain x (실측/기준 - 1)
        self.declare_parameter('throttle_adapt_max', 0.015)   # 보정 절대 상한 (안전 클램프)
        self.declare_parameter('y_latch_ratio', 0.02)         # 래치 감지 yr 문턱 (perception 과 동일값 권장)
        self.declare_parameter('y_latch_frames', 10)          # 래치 감지 연속 틱
        # 비상구 (07-15 사용자 결정): 구 3-표결(yaw/시간/가로선 재등장) + max_loop
        # 전부 삭제 — 속도 의존 재캘리 + 조기 탈출 오발(=실격) 이력. 유일 failsafe 는
        # 링 체류 절대 상한 하나 (속도 스케일 없음). 실측 랩타임 기반 '2랩 근처
        # (=A 부근에서 발동)'로 캘리할 것. 참고: 현대 스택 실측 랩 18.6s@0.185.
        self.declare_parameter('ra_failsafe_exit_s', 40.0)
        self.declare_parameter('crossline_cooldown_s', 2.0)  # 게이트 카운트 간 최소 간격 (재카운트 디바운스)
        # 노란색이 회전교차로 진입을 게이트한다 (회전교차로는 노란색, 외곽 루프는 흰색)
        self.declare_parameter('use_yellow_entry', True)
        self.declare_parameter('yellow_enter_ratio', 0.06)  # ROI 노란색 비율 => 회전교차로 안
        # --- 회전교차로 진입/탈출 = 갈림길과 동일한 브랜치 락온 ---
        # 진입: 가로선 첫 감지 -> RA on + 진입측 one-shot 락온(링 순환 방향).
        # 탈출: 링을 돌며 가로선 상승엣지를 세고, 도달 시 출구 브랜치로 락온.
        self.declare_parameter('roundabout_exit_gates', 2)   # 군집 순서: 반대편 입구(1) -> 우리 입구(2)=탈출
                                                             # (진입 가로선은 블랭크+재무장으로 제외; 1=한 바퀴)
        self.declare_parameter('roundabout_exit_side', '')   # 출구 브랜치 side; '' 이면 race_dir 파생
        self.declare_parameter('entry_lock_release_s', 3.0)  # 진입측 락온 강제 해제 시간 (07-11 오후: 저속 대응 2->3s)
        # 탈출 락 (07-11 run20, 방안1): RA 탈출 순간부터 이 시간 동안 조향에 직접
        # (post-PID) 우측 편향 -> 개구부의 차선 소실에도 우회전 유지. fork_bias 대체.
        self.declare_parameter('exit_steer_bias', 0.18)      # 양수 = 탈출측(우)
        self.declare_parameter('exit_lock_release_s', 2.5)
        # 진입 락온 중 조향 피드포워드 (음수 = 링 안쪽/좌). 진입 급좌회전 언더스티어
        # 대응 — 총 편향 0.15 는 언더스티어 재발(run15), -0.22 사용자 지정.
        self.declare_parameter('entry_steer_bias', -0.15)
        # RA 진입 후 게이트 카운트 금지 시간(진입선 오카운트 방어). 길어서 손해는
        # "한 바퀴 더"뿐(과회전 허용), 짧으면 조기 탈출=실격 위험 -> 길게 잡는다.
        # 단, 실측 한 바퀴 시간보다는 반드시 짧아야 함 (트랙에서 라이브 조정).
        self.declare_parameter('gate_blank_s', 7.0)   # 잔상 '출생' 덮개 (군집 v2 — 무효 판정용, 무스케일)
        self.declare_parameter('gate_rearm_s', 0.5)          # 가로선이 이 시간 연속 OFF 여야 다음 카운트 무장
        self.declare_parameter('gate_cluster_gap_s', 1.8)    # 목격 간격 이 미만 = 같은 군집 (파편/2조각 병합)
        self.declare_parameter('gate_cluster_on_s', 0.25)    # 군집 누적 ON 카운트 문턱 (run76 약피처 0.19 배제, 실선 0.5+ 통과)
        # 이 조향적분 전에는 탈출 게이트 잠금 (링 중간 가짜선 차단).
        # 07-12 재배치 4.2 -> 3.6 (사용자 승인, B안): 4.2 는 진짜 탈출선 실측
        # (4.12~4.28)과 여유 2%뿐이라 손 개입/빠른 랩에서 yaw 가 모자라 불발 —
        # run29 진짜 3연타 3.68~3.81 불발, run31 진짜 3.65 불발 -> 둘 다 페일세이프
        # 강제 탈출 후 이탈. 가짜 실선 실측 최대 3.11 (run31 2.57~2.65) 이라
        # 3.6 은 가짜/진짜 사이 균형점 (여유 각각 ~0.5).
        self.declare_parameter('yaw_gate_min', 1.0)   # 위생 하한 (위치 판별은 군집 순서가 담당)

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
            'yellow_drive_throttle': float(g('yellow_drive_throttle').value),
            'yellow_slow_ratio': float(g('yellow_slow_ratio').value),
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
            'red_slow_ratio': float(g('red_slow_ratio').value),
            'marker_clear_frames': int(g('marker_clear_frames').value),
            'fork_bias': float(g('fork_bias').value),
            'fork_hold_s': float(g('fork_hold_s').value),
            'enter_sustain_s': float(g('enter_sustain_s').value),
            'ra_min_drive_s': float(g('ra_min_drive_s').value),
            'enter_max_curvature': float(g('enter_max_curvature').value),
            'turn_direction': turn_direction,
            'branch_bias': float(g('branch_bias').value),
            'branch_yellow_min': float(g('branch_yellow_min').value),
            'circle_steer_bias': float(g('circle_steer_bias').value),
            'reacq_arm_s': float(g('reacq_arm_s').value),
            'ra_blind_bias': float(g('ra_blind_bias').value),
            'min_loop_time_s': float(g('min_loop_time_s').value),
            'ra_ref_drive_s': float(g('ra_ref_drive_s').value),
            'throttle_ref_latch_s': float(g('throttle_ref_latch_s').value),
            'throttle_adapt_gain': float(g('throttle_adapt_gain').value),
            'throttle_adapt_max': float(g('throttle_adapt_max').value),
            'y_latch_ratio': float(g('y_latch_ratio').value),
            'y_latch_frames': int(g('y_latch_frames').value),
            'ra_failsafe_exit_s': float(g('ra_failsafe_exit_s').value),
            'crossline_cooldown_s': float(g('crossline_cooldown_s').value),
            'use_yellow_entry': bool(g('use_yellow_entry').value),
            'yellow_enter_ratio': float(g('yellow_enter_ratio').value),
            'roundabout_exit_gates': int(g('roundabout_exit_gates').value),
            'roundabout_exit_side': exit_side,
            'roundabout_entry_side': entry_side,
            'entry_lock_release_s': float(g('entry_lock_release_s').value),
            'exit_steer_bias': float(g('exit_steer_bias').value),
            'exit_lock_release_s': float(g('exit_lock_release_s').value),
            'entry_steer_bias': float(g('entry_steer_bias').value),
            'gate_blank_s': float(g('gate_blank_s').value),
            'gate_cluster_gap_s': float(g('gate_cluster_gap_s').value),
            'gate_cluster_on_s': float(g('gate_cluster_on_s').value),
            'yaw_gate_min': float(g('yaw_gate_min').value),
            'gate_rearm_s': float(g('gate_rearm_s').value),
        }
        self.sm = RaceStateMachine(cfg)

        # ----- live-tunable params (ros2 param set 으로 주행 중 변경 가능) -----
        # state_machine 이 매 step 마다 self.sm.cfg 를 읽으므로 값만 갱신하면 즉시 반영됨.
        # state_machine 이 매 step 마다 self.sm.cfg 에서 읽는 값들만 라이브 변경 가능.
        # (steer_center/steer_scale 는 _lane_steer 에서 매번 cfg 를 읽으므로 즉시 반영됨)
        self.live_tunable = {
            'drive_throttle', 'slow_throttle', 'stop_throttle', 'curve_slow',
            'yellow_drive_throttle', 'yellow_slow_ratio',
            'steer_center', 'steer_scale', 'steer_left_gain',
            'kp', 'ki', 'kd',   # PID 게인 (조향 세기) 트랙에서 라이브 튜닝
            'max_steering_delta', 'steer_slow',   # look-ahead 보조 (rate limit / 조향 감속)
            'fork_bias', 'fork_hold_s',           # 갈림길 편향 세기 / 유지 시간
            'branch_bias',                        # In/Out 색상 편향 (기본 0=off, 현장 재활성용)
            'roundabout_exit_gates',              # 회전교차로 탈출 게이트 카운트 (트랙 실측)
            'enter_sustain_s', 'ra_min_drive_s',  # 진입 debounce / 진입 무장 지연
            'enter_max_curvature',                # 곡선 구간 크로스라인 오인식 차단
            'entry_lock_release_s', 'entry_steer_bias',   # 진입 락온 시간 / 진입 피드포워드
            'exit_lock_release_s', 'exit_steer_bias',     # 탈출 락 시간 / 탈출 피드포워드
            'gate_blank_s', 'gate_rearm_s', 'yaw_gate_min',
            'gate_cluster_gap_s', 'gate_cluster_on_s',
            'ra_ref_drive_s',                     # 속도 스케일 기준 (G->RA 소요시간)
            'throttle_ref_latch_s', 'throttle_adapt_gain', 'throttle_adapt_max',
            'crossline_cooldown_s',               # 게이트 카운트 간 최소 간격
            # 링을 실제 속도로 돌려 실측해야 하므로 주행 중 변경 가능해야 한다.
            # (비상구를 늘리지 못하면 측정 중 강제 탈출해 랩 측정 자체가 불가능하다.)
            'min_loop_time_s', 'ra_failsafe_exit_s',
            'finish_min_drive_s',                 # 빨간불 무시 최소 주행시간 (오인식 가드)
            'light_min_area',                     # 신호등 bbox 최소 면적 (오검출 필터)
            'conf_threshold',                     # YOLO confidence 문턱 (현장 조정)
            'circle_steer_bias',                  # 회전교차로 링 유지 편향 세기
            'reacq_arm_s',                        # 재획득 규칙 무장 지연 (RA 진입 후 초)
            'ra_blind_bias',                      # RA 맹목 폴백 (소실 시 링 호)
            'curve_steer_bias',                   # DRIVE 급커브 feed-forward 편향
            'max_throttle_delta',                 # 스로틀 상승 rate limit (돌입전류 완화)
            'undervolt_slow_v', 'undervolt_stop_v',   # 저전압 가드 임계
            'start_kick_throttle', 'start_kick_s',    # 출발 킥 (정지마찰 극복)
            'red_slow_ratio',                         # 빨간 도로 감지 임계 (트랙 실측 필요)
            'marker_area_trigger',                    # 마커 면적 게이트 (구간 밖에서만 적용)
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
        # 재획득 규칙 무장 플래그 (RA+reacq_arm_s 경과) -> perception 의
        # "nl 0->1 재획득 = 우측 경계" 규칙 스코프. 토픽명은 perception 리빌드를
        # 피하려고 구 이름(merge_zone) 그대로 둠 (07-12 의미 변경).
        self.merge_pub = self.create_publisher(Bool, '/decision/merge_zone', 10)

        self.dt = 1.0 / float(g('control_hz').value)
        self._last_thr_adj = 0.0
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
        # 스로틀 동적 보정 적용 (순항 값에만 — 정지/중립은 제외). 킥/램프 상류.
        adj = float(getattr(self.sm, 'throttle_adj', 0.0))
        if adj != 0.0 and throttle > 0.05:
            throttle = throttle + adj
        if adj != self._last_thr_adj:
            self.get_logger().info(f'throttle_adj -> {adj:+.3f} (G->latch 실측 기반)')
            self._last_thr_adj = adj
        # 래치된 방향을 perception 에 publish -> lane_node 가 브랜치 선택(guide 시드)에 사용.
        # 표지판이 시야에서 사라진 뒤에도 latch 가 유지되는 동안(갈림길 통과 전) 계속
        # 그 브랜치를 좇게 하려고, confirmed 가 아니라 sm.turn_latch 를 보낸다.
        self.fork_dir_pub.publish(String(data=self.sm.turn_latch or ''))
        # 현재 주행 모드를 publish -> BEV 디버그 화면 표시.
        self.state_pub.publish(String(data=state))
        # 재획득 규칙 무장 여부 -> perception (nl 0->1 재획득 = 우측 규칙 스코프)
        self.merge_pub.publish(Bool(data=bool(getattr(self.sm, 'reacq_armed', False))))
        # 상태 전환은 INFO 로 남긴다 (디버그 레벨 없이도 전환 사유 추적 가능):
        # ROUNDABOUT 탈출 = 게이트 카운트 or 비상구(circle_t 로 판별).
        if state != self._prev_state:
            self.get_logger().info(
                f'state {self._prev_state} -> {state} '
                f'(race_t={self.sm.race_t:.1f}s circle_t={self.sm.circle_t:.1f}s '
                f'gate={self.sm._gate_count} '
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
            throttle = max(min(throttle, float(cfg['slow_throttle']) + adj),
                           throttle * cut)

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
        # In 코스에서 정지->출발이 일어나는 지점: 초록불 출발, 빨간 도로의 ArUco 해제 후.
        # 둘 다 목표가 drive_throttle 이고 그 값이 곧 출발 임계라 킥 없이는 여유가 없다.
        # 킥에도 동적 보정 가산 ("각 스로틀 값에 +/-"). 첫 출발 킥은 래치 전이라
        # adj=0 이고, ArUco 재출발 킥부터 보정이 반영된다.
        kick = float(cfg.get('start_kick_throttle', 0.0))
        if kick > 0.0:
            kick = kick + adj
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
            f'entryV={self.sm._entry_votes} gate={self.sm._gate_count} '
            f'sc={self.sm._speed_scale:.2f} '
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
