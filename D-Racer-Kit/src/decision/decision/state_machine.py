"""RaceStateMachine: 인지(perception) 결과를 주행으로 바꾸는 두뇌.

조향은 항상 차선 PID 에서 나온다 (OpenCV offset -> steer).
상태는 throttle 과 어떤 미션 동작을 활성화할지만 바꾼다:

  WAIT_GREEN    -> 정지선에서 정차; YOLO 가 green_light 를 보면 -> 출발
  DRIVE         -> 주행 속도로 차선 추종 (직선, S커브, 갈림길).
                   'in' 코스에서는 회전교차로 진입(노란 가로선=정지선 첫 감지)을
                   감시하다가 ROUNDABOUT 으로 전환한다.
  ROUNDABOUT    -> 링을 돈다 (차선 추종 + 조기 이탈 방지용 turn bias).
                   진입 순간 진입측 브랜치로 one-shot 락온(링에 확실히 올라탐).
                   탈출 = 진입 때 만난 노란 가로선(정지선)을 "한 번 더" 만나면
                   (게이트 카운트) 출구 브랜치로 락온해 나간다. 반드시
                   min_loop_time 이 지난 뒤에만 -- 한 바퀴 미달 탈출은 실격이다.
  OBSTACLE_STOP -> ArUco 마커가 보이면 -> 사라질 때까지 완전 정지
                   (DRIVE 중에만 진입 — 장애물은 코스 순서상 항상 RA 뒤 마지막 미션)
  FINISH        -> 결승선 통과 후 red_light 를 기다림 -> 완전 정지
  DONE          -> 영구 정지

탈출 판정 (IMU 없음, 마커 없음):
  주(PRIMARY) = 가로선 게이트 카운트 (블랭크 + 재무장으로 진입선 재카운트 차단)
  백업(failsafe) = 3-표결: yaw proxy(조향 적분) / 경과시간 / 가로선 재등장
                   중 >= lap_votes_needed 개 동의 (junction 표는 오검출로 제외)
  최후 = max_loop_time 강제 탈출.
min_loop_time 전에는 절대 탈출하지 않는다. 모든 임계값은 늦게 나가는 쪽으로:
규정상 >= 1 바퀴가 허용이라 과회전은 공짜지만 < 1 바퀴는 실격이다.
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
        self.obstacle_done = False    # 동적 장애물 미션 완료(정지->퇴거 1회) -> 빨간불 대기 무장
        self.race_t = 0.0             # 초록불 출발 후 경과 시간 (빨간불 예비 무장용)

        # 회전교차로 (junction 카운트) 상태
        self.roundabout_done = False  # 이미 회전을 완료함 (재진입 금지)
        self.enter_acc = 0.0          # 진입 감지용 지속 커브 누적기
        self.circle_t = 0.0           # 현재 회전에서 보낸 시간
        self.yaw_proxy = 0.0          # IMU 없는 헤딩 추정치 (조향 적분)
        self._entry_votes = 0         # 마지막 회전교차로 진입 투표 수 (디버그/로그)
        self._exit_votes = 0          # 마지막 회전교차로 탈출 투표 수 (디버그/로그)
        self._entry_lock_active = False  # 진입측 락온이 걸려 있는 동안 True (one-shot)
        self._gate_armed = False      # 가로선이 충분히 꺼진 뒤에만 다음 카운트 무장
        self._gate_off_t = 0.0        # 가로선 연속 OFF 시간 (재무장 판정용)
        # 브랜치 락온 탈출용 게이트 통과 카운터 (노란 가로선 or 점선 개구부의
        # 상승엣지 = 게이트 1회 통과). roundabout_exit_gates 회 도달 시 출구 브랜치로 락온.
        self._gate_prev = False       # 직전 프레임의 게이트 신호 (엣지 감지용)
        self._gate_count = 0          # 이번 회전에서 센 게이트 통과 횟수
        self._gate_cd = 0.0           # 카운트 사이의 debounce 쿨다운

    # ---------- 감지 헬퍼 ----------
    def _seen(self, dets, label):
        c = self.cfg['conf_threshold']
        # 신호등은 bbox 면적 게이트 추가(light_min_area, 0=off): 멀리 있는 빨간
        # 물체(트랙 빨간 띠, 옷 등)의 작은 오검출 박스를 걸러낸다.
        min_area = self.cfg.get('light_min_area', 0.0) if label.endswith('_light') else 0.0
        for d in dets:
            if d.label == label and d.confidence >= c:
                if min_area > 0.0 and (d.width * d.height) < min_area:
                    continue
                return d
        return None

    # ---------- 조향 (항상 차선 PID) ----------
    def _lane_steer(self, lane, dt):
        if lane.lane_found:
            target = (lane.offset + self._turn_bias() + self._branch_bias(lane)
                      + self._curve_bias(lane))
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

    def _in_red_zone(self, lane):
        """빨간 노면(ArUco 장애물 구간)에 진입했는가. red_slow_ratio 는 '멀리서 빨간 도로가
        보이기 시작하는' 수준으로 낮게 잡아, 구간에 닿기 전에 미리 감속하도록 한다.
        0 = 기능 off (red_ratio 를 발행하지 않는 구버전 perception 과도 호환)."""
        thr = self.cfg.get('red_slow_ratio', 0.0)
        if thr <= 0.0:
            return False
        return float(getattr(lane, 'red_ratio', 0.0)) >= thr

    def _curve_bias(self, lane):
        """급커브 feed-forward: 곡률에 비례한 조향 편향(DRIVE 에서만). 급좌커브에서
        가까운 왼선이 사라져도 curvature(밴드 간 차이)는 살아있어, 선이 완전히
        사라지기 전에 미리·더 꺾어 바깥으로 튕기는 것을 막는다. 부호는 offset 과 동일
        규약이라 target 에 그대로 더한다(좌커브 curvature<0 -> 좌조향 강화). 0=off."""
        if self.state != State.DRIVE:
            return 0.0
        return self.cfg.get('curve_steer_bias', 0.0) * getattr(lane, 'curvature', 0.0)

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
        """일반 상태 전이 (PID/감지 카운터 리셋; 회전교차로 진행 상황은 건드리지 않음)."""
        self.state = state
        self.pid.reset()
        self.green_count = 0
        self.red_count = 0
        self.marker_gone = 0

    def _start_roundabout(self):
        self.state = State.ROUNDABOUT
        self.pid.reset()
        self.circle_t = 0.0
        self.yaw_proxy = 0.0
        # 게이트(가로선) 카운터: 진입 정지선 재카운트를 3중으로 차단 —
        #  ① 블랭크: gate_blank_s(기본 min_loop 연동) 동안 카운트 금지
        #  ② 재무장(arm): 가로선이 gate_rearm_s 이상 "연속으로 꺼져" 있어야
        #     다음 상승엣지를 셀 수 있음 (한두 프레임 깜빡임으로는 무장 안 됨)
        #  ③ 상승엣지: _gate_prev=True 로 시작(진입선이 보이는 상태 가정)
        self._gate_prev = True
        self._gate_count = 0
        self._gate_cd = float(self.cfg.get('gate_blank_s', self.cfg['min_loop_time_s']))
        self._gate_armed = False
        self._gate_off_t = 0.0
        # 진입측 락온 (one-shot): RA 켜지는 순간 딱 1회, 링 순환 방향 브랜치로
        # 시드를 밀어 진입 갈림길에서 링을 타게 한다. 해제는 fork 재수렴 또는
        # entry_lock_release_s 중 먼저 오는 쪽 (아래 ROUNDABOUT 블록에서).
        # 이후 링 주행 중 fork 가 오발동해도 latch 를 다시 세우는 코드가 없으므로
        # 재락온은 구조적으로 불가능하다.
        entry_side = self.cfg.get('roundabout_entry_side')
        self.turn_latch = entry_side if entry_side in ('left', 'right') else None
        self.turn_latch_age = 0.0
        self._fork_seen = False
        self._entry_lock_active = self.turn_latch is not None

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

        if self.state not in (State.WAIT_GREEN, State.DONE):
            self.race_t += dt   # 출발 후 경과 시간 (FINISH 게이트용)

        # 갈림길 방향은 decision_node 의 다중 프레임 표결에서 온다(단일 감지가 아님).
        # confirmed_fork_direction 으로 전달된다. 표지판은 두 브랜치 사이에 있으므로
        # "표지판 확정" == "갈림길 도착": 방향을 latch 하고, 기하 조건으로 해제한다 —
        # perception 의 fork 플래그가 한 번 켜졌다가 다시 꺼지면(두 브랜치가 재합류 /
        # 도로가 다시 좁아짐) 해제. 예전의 타이머 전용 홀드(decision_node 의 표결 홀드와
        # 중첩돼 ≈2×fork_hold_s 가 되던 것)를 대체한다. fork_hold_s 는 failsafe 상한으로만
        # 유지한다.
        fork_now = bool(getattr(lane, 'fork', False))
        # ROUNDABOUT 동안은 표지판 표결을 무시한다 — 진입/탈출 락온은 RA 로직이
        # 소유하며, 링 위 표지판 오검출이 latch 를 덮어쓰면 perception 시드가 엉뚱한
        # 브랜치로 밀린다 (링 위 재락온 금지 원칙).
        if (self.confirmed_fork_direction is not None
                and self.state != State.ROUNDABOUT):
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

        # ----- 장애물 정지 (DRIVE 전용) -----
        # 코스 순서 고정: 동적 장애물은 항상 회전교차로 "뒤" 마지막 미션이라
        # RA 중에는 마커가 등장하지 않는다. RA 등 다른 상태에서 트리거를 열어두면
        # 링 안에서 다음 구간의 진짜 마커가 멀리 보이는 것만으로 링 한가운데
        # 정지(마커 치울 때까지 무한 대기)할 수 있어 DRIVE 로 제한한다.
        #
        # 빨간 도로 구간(= 마커가 놓이는 곳)에 들어오면 marker_area_trigger 를 무시하고
        # "마커가 보이는 즉시" 정지한다. 면적 게이트는 멀리 있는 마커를 거르려는 장치인데,
        # 이미 해당 구간 안이면 그 마커가 곧 미션 대상이므로 일찍 서는 편이 안전하다.
        # 구간 밖에서는 기존처럼 면적 게이트를 요구해, 트랙 반대편의 마커가 멀리 보이는
        # 것만으로 서버리는 사고를 막는다.
        in_red_zone = self._in_red_zone(lane)
        if self.state == State.DRIVE and aruco.detected and (
                in_red_zone or aruco.area_ratio >= self.cfg['marker_area_trigger']):
            self._enter(State.OBSTACLE_STOP)

        if self.state == State.OBSTACLE_STOP:
            self.marker_gone = 0 if aruco.detected else self.marker_gone + 1
            if self.marker_gone >= self.cfg['marker_clear_frames']:
                self.obstacle_done = True  # 장애물 미션 완료 -> 이제부터 빨간불 대기
                self._enter(State.DRIVE)   # DRIVE 에서만 진입하므로 복귀도 DRIVE
            return center, stop, self.state.value

        steer = self._lane_steer(lane, dt)

        # ----- DRIVE -----
        if self.state == State.DRIVE:
            # 회전교차로 진입 (In 코스, 한 번 완료하기 전까지만).
            # 큰 틀(인코스 플로우): 노란 추종으로 링까지 온 뒤, 정지선처럼 보이는
            # 노란 가로선(yellow_crossline)을 "처음" 만나면 ROUNDABOUT on.
            # 오검 방지 장치 둘은 유지 — yellow_ratio 게이트(흰 외곽 루프에서
            # 안 걸리게) + enter_sustain_s 지속 debounce(한 프레임 깜빡임 무시).
            # 탈출은 ROUNDABOUT 블록의 게이트 카운트: 같은 가로선을 "한 번 더"
            # 만나면 출구 브랜치 락온으로 나간다.
            if self.cfg['course'] == 'in' and not self.roundabout_done:
                on_yellow = lane.yellow_ratio >= self.cfg['yellow_enter_ratio']
                gate = on_yellow if self.cfg['use_yellow_entry'] else True
                # 곡률 게이트: 급커브에서는 노란 차선 자체가 '수평에 가까운 선'으로 보여
                # 크로스라인 검출을 통과한다(실측: 좌회전 중 cv=-1.00 에서 오진입).
                # 진짜 정지선은 차선과 수직이므로 곧은 구간에서 만난다. 0 = 게이트 off.
                cmax = self.cfg.get('enter_max_curvature', 0.0)
                straight = (cmax <= 0.0
                            or abs(getattr(lane, 'curvature', 0.0)) <= cmax)
                trigger = gate and straight and bool(lane.yellow_crossline)
                self._entry_votes = int(trigger)
                self.enter_acc = self.enter_acc + dt if trigger else 0.0
                if self.enter_acc >= self.cfg['enter_sustain_s']:
                    self._start_roundabout()
                    return steer, self.cfg['slow_throttle'], self.state.value

            # 빨간불 = 도착 신호. 코스 순서 고정(... -> 동적 장애물 -> 도착)이므로
            # 장애물 미션 완료(obstacle_done)가 주 무장 조건: 그 전에는 빨간불을
            # 아예 무시해 코스 중간 오인식(빨간 띠·옷 등)으로 FINISH/DONE 되는
            # 사고를 구조적으로 차단한다. 아루코를 통째로 놓친 비상 주행에서도
            # 결승선에서 멈출 수 있게 finish_min_drive_s 경과를 예비 무장으로
            # 남긴다 (실측 코스 소요시간보다 길게 설정할 것).
            red_armed = (self.obstacle_done
                         or self.race_t >= self.cfg.get('finish_min_drive_s', 0.0))
            if red_armed:
                self.red_count = self.red_count + 1 if self._seen(dets, 'red_light') else 0
            else:
                self.red_count = 0
            if self.red_count >= self.cfg['red_frames']:
                self._enter(State.FINISH)
                return center, stop, self.state.value
            # curvature 적응 속도: 커브에서 감속 (규정 권장 사항이며, S커브/코너에서
            # 차선 이탈 위험을 줄여준다).
            curve = abs(getattr(lane, 'curvature', 0.0))
            throttle = self.cfg['drive_throttle'] * (1.0 - self.cfg['curve_slow'] * curve)
            throttle = max(self.cfg['slow_throttle'], throttle)
            # 빨간 도로(장애물 구간)가 보이기 시작하면 저속주행. 마커 앞에서 제동 거리를
            # 줄여 정지 성공률을 높인다. 이 구간은 직선이라 curve_slow 로는 감속되지 않는다.
            if in_red_zone:
                throttle = min(throttle, self.cfg['slow_throttle'])
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

            # 진입측 락온 해제: fork 재수렴(위 공통 latch 블록)이 먼저 오면 그때,
            # 아니면 entry_lock_release_s 경과 시 무조건 해제. 진입 갈림길은 순식간에
            # 지나가므로 오래 유지하면 시드가 안쪽 실선을 파고든다 (변수 5 대응).
            if self._entry_lock_active:
                if (self.turn_latch is None
                        or self.circle_t >= self.cfg.get('entry_lock_release_s', 2.0)):
                    self.turn_latch = None
                    self._fork_seen = False
                    self._entry_lock_active = False

            # (2b) 브랜치 락온 탈출용 게이트 통과 카운터. 노란 가로선(진입 때 만난
            # 그 정지선)의 상승엣지를 세되, 진입선 재카운트를 3중 차단한다:
            # 블랭크(gate_blank_s) + 재무장(가로선이 gate_rearm_s 이상 연속 OFF 여야
            # 다음 엣지 유효 — 한두 프레임 깜빡임은 무장 실패) + 카운트 후 재무장 해제.
            # 링을 돌아 같은 가로선을 roundabout_exit_gates(기본 1)번 더 만나면 탈출.
            if self._gate_cd > 0.0:
                self._gate_cd = max(0.0, self._gate_cd - dt)
            gate_now = bool(lane.yellow_crossline)
            if gate_now:
                self._gate_off_t = 0.0
            else:
                self._gate_off_t += dt
                if self._gate_off_t >= self.cfg.get('gate_rearm_s', 0.5):
                    self._gate_armed = True
            if (gate_now and not self._gate_prev
                    and self._gate_cd <= 0.0 and self._gate_armed):
                self._gate_count += 1
                self._gate_cd = self.cfg.get('crossline_cooldown_s', 2.0)
                self._gate_armed = False
            self._gate_prev = gate_now

            # ----- 바퀴 수 판정 -----
            # 절대 하한: min_loop_time 전에는 절대 탈출하지 않는다 (한 바퀴 미달은
            # 미션 실패이고, 진입 쪽 노란 가로선을 출구로 오인하는 것도 막아준다).
            # 주 탈출 = 게이트 카운트. 백업 = 세 가지 추정치(yaw proxy, 시간, 노란
            # 가로선 재등장) 중 >= lap_votes_needed 개 동의.
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
                # 어떻게든 탈출해야 함). 3중 2 추정치 표결을 백업으로 유지한다.
                # junction 표는 점선 개구부 오검출 문제로 표결에서 제외.
                # 이 경로도 주 탈출과 똑같이 출구측 락온을 건다 — DRIVE 복귀 후에도
                # FOLLOW-Y 는 유지되므로 락온 없이는 링 노란선을 계속 따라 무한
                # 순환할 수 있다. 링 중간에서 걸려도 갈림길 기하가 없으면 시드
                # 밀기는 무해하고, 출구 개구부에 도달하는 순간 그쪽 브랜치를 잡는다.
                yaw_done = self.yaw_proxy >= self.cfg['yaw_lap_threshold']
                time_done = self.circle_t >= self.cfg['nominal_loop_time_s']
                # 가로선 표도 게이트와 같은 재등장 규율 적용: 재무장(_gate_armed,
                # 0.5s 이상 사라졌다 다시 나타남) 상태에서 보일 때만 인정한다.
                # 진입선이 시야에 계속 남아 있는 것(정차/저속 통과)은 표가 아니다.
                cross_done = bool(lane.yellow_crossline) and self._gate_armed
                votes = int(yaw_done) + int(time_done) + int(cross_done)
                self._exit_votes = votes
                if votes >= self.cfg['lap_votes_needed']:
                    self.turn_latch = self.cfg['roundabout_exit_side']
                    self.turn_latch_age = 0.0
                    self._fork_seen = False
                    self.roundabout_done = True
                    self._enter(State.DRIVE)
                    return steer, self.cfg['slow_throttle'], self.state.value

            # 최후 failsafe: 모든 추정치가 실패하면 강제 탈출 (역시 출구측 락온)
            if self.circle_t >= self.cfg['max_loop_time_s']:
                self.turn_latch = self.cfg['roundabout_exit_side']
                self.turn_latch_age = 0.0
                self._fork_seen = False
                self.roundabout_done = True
                self._enter(State.DRIVE)
            return steer, self.cfg['slow_throttle'], self.state.value

        # ----- FINISH -----
        if self.state == State.FINISH:
            # DONE 도 단일 프레임이 아니라 red_frames 연속 인식으로만 확정.
            self.red_count = self.red_count + 1 if self._seen(dets, 'red_light') else 0
            if self.red_count >= self.cfg['red_frames']:
                self._enter(State.DONE)
            return center, stop, self.state.value

        # ----- DONE -----
        return center, stop, self.state.value
