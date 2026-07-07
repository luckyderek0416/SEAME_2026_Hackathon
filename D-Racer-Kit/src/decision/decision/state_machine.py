"""RaceStateMachine: 인지(perception) 결과를 주행으로 바꾸는 두뇌.

조향은 항상 차선 PID 에서 나온다 (OpenCV offset -> steer).
상태는 throttle 과 어떤 미션 동작을 활성화할지만 바꾼다:

  WAIT_GREEN    -> 정지선에서 정차; YOLO 가 green_light 를 보면 -> 출발
  DRIVE         -> 주행 속도로 차선 추종 (직선, S커브, 갈림길).
                   'in' 코스에서는 회전교차로 진입(지속적인 급커브)도
                   감시하다가 ROUNDABOUT 으로 전환한다.
  ROUNDABOUT    -> 링을 돈다 (차선 추종 + 조기 이탈 방지용 turn bias) 그리고
                   JUNCTION 감지로 바퀴 수를 센다 (링 바깥 라인이 입구/출구에서
                   벌어짐). 반드시 >= target_loops 이고 min_loop_time 이 지난
                   뒤에만 탈출 -- 규정상 한 바퀴를 다 돌기 전에 나가면 실격이다.
  OBSTACLE_STOP -> ArUco 마커가 보이면 -> 사라질 때까지 완전 정지
  FINISH        -> 결승선 통과 후 red_light 를 기다림 -> 완전 정지
  DONE          -> 영구 정지

바퀴 수 카운트 (IMU 없음, 마커 없음): 세 가지 독립 추정치가 투표한다.
  (1) junction  - 링 바깥 라인이 입구/출구 지점에서 다시 열림
  (2) yaw proxy - 조향 편향량의 적분 (IMU 없이 헤딩 추정; 속도가 거의 일정하면
                  누적 yaw 는 이 값에 비례)
  (3) time      - 회전 시간이 실측한 한 바퀴 시간에 도달
min_loop_time 전에는 절대 탈출하지 않고(한 바퀴 미달은 미션 실패), 그 뒤
세 추정치 중 >= lap_votes_needed 개가 동의하면 탈출한다. max_loop_time 은
failsafe. 모든 임계값을 늦게 나가는 쪽으로 치우쳐 잡는다: 규정상 >= 1 바퀴가
허용이라 약간의 과회전은 공짜지만 < 1 바퀴는 실격이다.
yaw_lap_threshold 와 nominal_loop_time_s 는 실제 트랙에서 캘리브레이션할 것.
"""

from enum import Enum

from decision.pid import PID


class State(Enum):
    WAIT_GREEN = 'WAIT_GREEN'
    DRIVE = 'DRIVE'
    ROUNDABOUT = 'ROUNDABOUT'
    OBSTACLE_STOP = 'OBSTACLE_STOP'
    FINISH = 'FINISH'
    DONE = 'DONE'


class RaceStateMachine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pid = PID(cfg['kp'], cfg['ki'], cfg['kd'], out_limit=1.0, i_limit=0.3)
        # skip_missions: 순수 차선 추종 테스트 모드 (신호등/회전교차로/장애물 없음).
        self.state = State.DRIVE if cfg.get('skip_missions') else State.WAIT_GREEN

        self.turn_latch = None        # 방향 표지판 확정 시 'left' / 'right'
        self.turn_latch_age = 0.0     # latch 유지 시간(초) (failsafe 타이머)
        self._fork_seen = False       # 이번 latch 동안 perception 이 fork 기하를 감지했는가
        # decision_node 가 표결(deque)로 확정해 매 틱 넣어주는 갈림길 방향. None = 미확정.
        self.confirmed_fork_direction = None
        self.green_count = 0
        self.red_count = 0
        self.marker_gone = 0
        self._resume_state = State.DRIVE

        # 회전교차로 (junction 카운트) 상태
        self.roundabout_done = False  # 이미 회전을 완료함 (재진입 금지)
        self.enter_acc = 0.0          # 진입 감지용 지속 커브 누적기
        self.circle_t = 0.0           # 현재 회전에서 보낸 시간
        self.cooldown = 0.0           # 진입/카운트 직후에는 junction 무시
        self.loops = 0
        self.side_present = False     # 링 바깥 라인이 현재 보임 (개구부 아님)
        self.yaw_proxy = 0.0          # IMU 없는 헤딩 추정치 (조향 적분)
        self._entry_votes = 0         # 마지막 회전교차로 진입 투표 수 (디버그/로그)
        self._exit_votes = 0          # 마지막 회전교차로 탈출 투표 수 (디버그/로그)
        # 브랜치 락온 탈출용 게이트 통과 카운터 (노란 가로선 or 점선 개구부의
        # 상승엣지 = 게이트 1회 통과). roundabout_exit_gates 회 도달 시 출구 브랜치로 락온.
        self._gate_prev = False       # 직전 프레임의 게이트 신호 (엣지 감지용)
        self._gate_count = 0          # 이번 회전에서 센 게이트 통과 횟수
        self._gate_cd = 0.0           # 카운트 사이의 debounce 쿨다운

    # ---------- 감지 헬퍼 ----------
    def _seen(self, dets, label):
        c = self.cfg['conf_threshold']
        for d in dets:
            if d.label == label and d.confidence >= c:
                return d
        return None

    # ---------- 조향 (항상 차선 PID) ----------
    def _lane_steer(self, lane, dt):
        if lane.lane_found:
            target = lane.offset + self._turn_bias() + self._branch_bias(lane)
        else:
            target = 0.0  # 차선 놓침: 직진하며 재획득을 기다린다
        correction = self.pid.update(target, dt)
        # steer_scale 은 [-1,1] 보정값을 키트의 조향 범위로 매핑한다.
        # 차가 반대 방향으로 조향하면 이 값의 부호를 뒤집을 것.
        return float(self.cfg['steer_center'] + correction * self.cfg['steer_scale'])

    def _turn_bias(self):
        # fork 편향은 주행 중에만 적용; ROUNDABOUT 등에서 남아 있는 stale latch 가
        # 조향을 기울이면 안 된다.
        if self.state != State.DRIVE:
            return 0.0
        if self.turn_latch == 'left':
            return -self.cfg['fork_bias']
        if self.turn_latch == 'right':
            return +self.cfg['fork_bias']
        return 0.0

    def _branch_bias(self, lane):
        """색상 기반 In/Out 갈림길 선택 (방향 무관). 갈림길에서 In 경로는 노란색,
        Out 경로는 흰색이다. DRIVE 상태에서만, 노란 브랜치가 실제로 보일 때만 동작.
        'in' 이면 노란색 쪽으로, 'out' 이면 반대쪽으로 조향한다 — 노란색이 어느 쪽에
        나타나든 동작하므로 left/right 사전 설정이 필요 없다."""
        if self.state != State.DRIVE:
            return 0.0
        if lane.yellow_ratio < self.cfg['branch_yellow_min']:
            return 0.0
        toward_yellow = 1.0 if lane.yellow_offset >= 0.0 else -1.0
        if self.cfg['course'] == 'in':
            return self.cfg['branch_bias'] * toward_yellow      # 노란 브랜치로 진입
        return -self.cfg['branch_bias'] * toward_yellow         # 흰 브랜치 유지

    def _enter(self, state):
        """일반 상태 전이. 회전교차로 진행 상황은 리셋하지 않으므로, 장애물 이후
        ROUNDABOUT 으로 복귀해도 바퀴 수 카운트가 유지된다."""
        self.state = state
        self.pid.reset()
        self.green_count = 0
        self.red_count = 0
        self.marker_gone = 0

    def _start_roundabout(self):
        self.state = State.ROUNDABOUT
        self.pid.reset()
        self.circle_t = 0.0
        self.loops = 0
        self.cooldown = self.cfg['junction_cooldown_s']  # 진입 junction 은 무시
        self.side_present = False
        self.yaw_proxy = 0.0
        self._gate_prev = False
        self._gate_count = 0
        self._gate_cd = self.cfg['junction_cooldown_s']  # 진입 쪽 게이트는 무시

    # ---------- 메인 틱 ----------
    def step(self, lane, aruco, dets, dt):
        """(steering, throttle, state_name) 을 반환한다."""
        center = self.cfg['steer_center']
        stop = self.cfg['stop_throttle']

        # ----- 차선 전용 테스트 모드 (skip_missions) -----
        # 순수 차선 추종: 초록/빨간불, 회전교차로, 장애물, 갈림길 전부 없음.
        # OpenCV offset -> PID -> steer 에 curvature 기반 속도만 적용. 미션 로직 없이
        # 차선 유지(kp/kd/HSV)만 분리해서 튜닝할 때 쓴다.
        if self.cfg.get('skip_missions'):
            target = lane.offset if lane.lane_found else 0.0
            correction = self.pid.update(target, dt)
            steer = max(-1.0, min(1.0, center + correction * self.cfg['steer_scale']))
            curve = abs(getattr(lane, 'curvature', 0.0))
            throttle = self.cfg['drive_throttle'] * (1.0 - self.cfg['curve_slow'] * curve)
            throttle = max(self.cfg['slow_throttle'], throttle)
            return steer, throttle, 'LANE_ONLY'

        if self.cooldown > 0.0:
            self.cooldown = max(0.0, self.cooldown - dt)

        # 갈림길 방향은 decision_node 의 다중 프레임 표결에서 온다(단일 감지가 아님).
        # confirmed_fork_direction 으로 전달된다. 표지판은 두 브랜치 사이에 있으므로
        # "표지판 확정" == "갈림길 도착": 방향을 latch 하고, 기하 조건으로 해제한다 —
        # perception 의 fork 플래그가 한 번 켜졌다가 다시 꺼지면(두 브랜치가 재합류 /
        # 도로가 다시 좁아짐) 해제. 예전의 타이머 전용 홀드(decision_node 의 표결 홀드와
        # 중첩돼 ≈2×fork_hold_s 가 되던 것)를 대체한다. fork_hold_s 는 failsafe 상한으로만
        # 유지한다.
        fork_now = bool(getattr(lane, 'fork', False))
        if self.confirmed_fork_direction is not None:
            self.turn_latch = self.confirmed_fork_direction
            self.turn_latch_age = 0.0
            if fork_now:
                self._fork_seen = True
        elif self.turn_latch is not None:
            self.turn_latch_age += dt
            if fork_now:
                self._fork_seen = True
            reconverged = self._fork_seen and not fork_now
            if reconverged or self.turn_latch_age >= self.cfg.get('fork_hold_s', 8.0):
                self.turn_latch = None
                self._fork_seen = False

        # ----- WAIT_GREEN -----
        if self.state == State.WAIT_GREEN:
            self.green_count = self.green_count + 1 if self._seen(dets, 'green_light') else 0
            if self.green_count >= self.cfg['green_frames']:
                self._enter(State.DRIVE)
            return center, stop, self.state.value

        # ----- 전역 장애물 오버라이드 (미션이 주행보다 우선) -----
        if (self.state != State.OBSTACLE_STOP and aruco.detected
                and aruco.area_ratio >= self.cfg['marker_area_trigger']):
            self._resume_state = self.state
            self._enter(State.OBSTACLE_STOP)

        if self.state == State.OBSTACLE_STOP:
            self.marker_gone = 0 if aruco.detected else self.marker_gone + 1
            if self.marker_gone >= self.cfg['marker_clear_frames']:
                self._enter(self._resume_state)   # 이전 상태로 복귀 (바퀴 수 카운트 유지)
            return center, stop, self.state.value

        steer = self._lane_steer(lane, dt)

        # ----- DRIVE -----
        if self.state == State.DRIVE:
            # 회전교차로 진입 (In 코스, 한 번 완료하기 전까지만).
            # 3중 2 표결: yellow_crossline (갈림길의 노란 가로선), junction (점선
            # 개구부), 급한 curvature. 어떤 단일 신호도 혼자서는 진입시키지 못하고,
            # yellow_ratio 게이트가 흰색 외곽 루프의 코너가 이를 트리거하는 것을
            # 막는다. 기존 enter_acc/enter_sustain_s debounce 는 유지: 표결이
            # enter_sustain_s 동안 유지되어야 진입한다.
            if self.cfg['course'] == 'in' and not self.roundabout_done:
                entry_votes = 0
                if lane.yellow_crossline:
                    entry_votes += 1
                if lane.junction:
                    entry_votes += 1
                # ⚠️ enter_curvature(기본 0.45)는 원래 |offset| 기준값이었다.
                # curvature 는 스케일이 더 작으므로(대략 0.1~0.3) 트랙에서 재튜닝 필요.
                if lane.lane_found and abs(lane.curvature) >= self.cfg['enter_curvature']:
                    entry_votes += 1
                self._entry_votes = entry_votes
                on_yellow = lane.yellow_ratio >= self.cfg['yellow_enter_ratio']
                gate = on_yellow if self.cfg['use_yellow_entry'] else True
                trigger = gate and entry_votes >= 2
                self.enter_acc = self.enter_acc + dt if trigger else 0.0
                if self.enter_acc >= self.cfg['enter_sustain_s']:
                    self._start_roundabout()
                    return steer, self.cfg['slow_throttle'], self.state.value

            self.red_count = self.red_count + 1 if self._seen(dets, 'red_light') else 0
            if self.red_count >= self.cfg['red_frames']:
                self._enter(State.FINISH)
                return center, stop, self.state.value
            # curvature 적응 속도: 커브에서 감속 (규정 권장 사항이며, S커브/코너에서
            # 차선 이탈 위험을 줄여준다).
            curve = abs(getattr(lane, 'curvature', 0.0))
            throttle = self.cfg['drive_throttle'] * (1.0 - self.cfg['curve_slow'] * curve)
            throttle = max(self.cfg['slow_throttle'], throttle)
            return steer, throttle, self.state.value

        # ----- ROUNDABOUT (IMU/마커 없음: 3가지 바퀴 수 추정치로 표결) -----
        if self.state == State.ROUNDABOUT:
            self.circle_t += dt
            # 링 유지: 회전 방향으로 편향을 줘서 차선 추종이 한 바퀴를 다 돌기 전에
            # 출구 브랜치로 빠지지 않게 한다.
            steer = steer + self.cfg['turn_direction'] * self.cfg['circle_steer_bias']
            steer = max(-1.0, min(1.0, steer))

            # (1) 조향 적분 yaw proxy (IMU 대체). 속도가 거의 일정하면 누적 yaw 는
            # 회전 방향 조향 편향량의 합에 비례(∝)한다.
            # 임계값은 실제 트랙에서 캘리브레이션한다 (yaw_lap_threshold).
            defl = self.cfg['turn_direction'] * (steer - center)
            if defl > 0.0:
                self.yaw_proxy += defl * dt

            # (2) JUNCTION 재등장 카운트 (비전)
            if not lane.junction:
                self.side_present = True   # 바깥 라인 보임 -> 링 위에 있음
            elif (self.cooldown <= 0.0 and self.circle_t >= self.cfg['min_loop_time_s']
                  and self.side_present):
                self.loops += 1
                self.cooldown = self.cfg['junction_cooldown_s']
                self.side_present = False

            # (2b) 브랜치 락온 탈출용 게이트 통과 카운터. 게이트 신호(노란 가로선
            # 또는 점선 junction 개구부)의 상승엣지를 쿨다운과 함께 세서, 물리적으로
            # 한 번 지나가면 딱 1회만 카운트되게 한다(프레임마다 1회가 아님). 갈림길이
            # 쓰는 것과 같은 취지의 명시적 탈출 트리거: 출구 게이트를
            # roundabout_exit_gates 번 통과하면 출구 브랜치로 락온해 빠져나간다.
            if self._gate_cd > 0.0:
                self._gate_cd = max(0.0, self._gate_cd - dt)
            gate_now = bool(lane.yellow_crossline or lane.junction)
            if gate_now and not self._gate_prev and self._gate_cd <= 0.0:
                self._gate_count += 1
                self._gate_cd = self.cfg['junction_cooldown_s']
            self._gate_prev = gate_now

            # ----- 바퀴 수 판정 -----
            # 절대 하한: min_loop_time 전에는 절대 탈출하지 않는다 (한 바퀴 미달은
            # 미션 실패이고, 진입 쪽 노란 가로선을 출구로 오인하는 것도 막아준다).
            # 그 뒤에는 네 가지 독립 추정치(junction 바퀴 수, yaw proxy, 시간, 노란
            # 가로선 재등장) 중 >= lap_votes_needed 개가 동의하면 탈출한다.
            # 편향은 늦게 나가는 쪽으로, 절대 일찍 나가지 않는다.
            if self.circle_t >= self.cfg['min_loop_time_s']:
                # 주(PRIMARY) 탈출: 출구 게이트를 충분히 통과함 -> 출구 브랜치로 락온
                # (표지판 갈림길과 같은 메커니즘). 그러면 차선 추종이 링에 다시 붙는
                # 대신 개구부 밖으로 조향한다. turn_latch 는 fork_dir 로 perception 에
                # publish 되며 위의 fork 재수렴 로직으로 해제된다.
                if self._gate_count >= self.cfg['roundabout_exit_gates']:
                    self.turn_latch = self.cfg['roundabout_exit_side']
                    self.turn_latch_age = 0.0
                    self._fork_seen = False
                    self.roundabout_done = True
                    self._enter(State.DRIVE)
                    return steer, self.cfg['slow_throttle'], self.state.value

                # FAILSAFE: 게이트 감지가 통과를 놓칠 수 있다 (포기 없는 미션은
                # 어떻게든 탈출해야 함). 4중 2 추정치 표결을 백업으로 유지한다
                # (브랜치 락온 없이 링 유지 편향만 해제).
                junction_done = self.loops >= self.cfg['target_loops']
                yaw_done = self.yaw_proxy >= self.cfg['yaw_lap_threshold']
                time_done = self.circle_t >= self.cfg['nominal_loop_time_s']
                cross_done = bool(lane.yellow_crossline)   # 출구 갈림길의 노란 가로선 보임
                votes = int(junction_done) + int(yaw_done) + int(time_done) + int(cross_done)
                self._exit_votes = votes
                if votes >= self.cfg['lap_votes_needed']:
                    self.roundabout_done = True
                    self._enter(State.DRIVE)
                    return steer, self.cfg['slow_throttle'], self.state.value

            # failsafe: 모든 추정치가 실패하면 강제 탈출
            if self.circle_t >= self.cfg['max_loop_time_s']:
                self.roundabout_done = True
                self._enter(State.DRIVE)
            return steer, self.cfg['slow_throttle'], self.state.value

        # ----- FINISH -----
        if self.state == State.FINISH:
            if self._seen(dets, 'red_light'):
                self._enter(State.DONE)
            return center, stop, self.state.value

        # ----- DONE -----
        return center, stop, self.state.value
