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
        self._fork_absent_t = 0.0     # fork 연속 부재 시간 (재수렴 디바운스)
        # decision_node 가 표결(deque)로 확정해 매 틱 넣어주는 갈림길 방향. None = 미확정.
        self.confirmed_fork_direction = None
        self.green_count = 0
        self.red_count = 0
        self.marker_gone = 0
        self.obstacle_done = False    # 동적 장애물 미션 완료(정지->퇴거 1회) -> 빨간불 대기 무장
        self.race_t = 0.0             # 초록불 출발 후 경과 시간 (빨간불 예비 무장용)
        # 빨간 노면 구간은 미션 순서상 회전교차로를 빠져나온 뒤에만 나타난다
        # (RA -> DRIVE[Y] -> DRIVE[W] -> 빨간 도로). 그 전의 red_ratio 는 전부
        # 오검출이므로(신호등/트랙 밖 물체) ROUNDABOUT 을 한 번 거치기 전엔 무시한다.
        self.roundabout_done = False

        # 회전교차로 (junction 카운트) 상태
        self.roundabout_done = False  # 이미 회전을 완료함 (재진입 금지)
        self.enter_acc = 0.0          # 진입 감지용 지속 커브 누적기
        self.circle_t = 0.0           # 현재 회전에서 보낸 시간
        # 속도 스케일 (07-12 run55 실증): yaw_proxy/블랭크는 시간 기반이라 빠른 랩일수록
        # 작아짐 — 진짜 정지선 yaw 가 run53 3.87 / run54 4.16 / run55 3.21 로 밴드 간
        # 이동, 가짜 최대 3.15 와 겹쳐 절대 임계 불가. G->RA 접근 소요시간(race_t)이
        # 그 런의 속도 밴드를 대표하므로 그 비율로 게이트 임계들을 스케일한다.
        self._speed_scale = 1.0       # RA 진입 시 확정: clamp(race_t/ra_ref_drive_s)
        self._y_run = 0               # Y래치 감지 연속 카운터 (스로틀 보정용)
        self._latch_seen = False      # 런당 1회
        self.throttle_adj = 0.0       # 순항 스로틀 가산 보정 (decision_node 가 적용)
        self.yaw_proxy = 0.0          # IMU 없는 헤딩 추정치 (조향 적분)
        self._merge_bridge_t = 0.0    # 병합 브리지 잔여 시간 (RA 탈출 시 무장)
        self._entry_votes = 0         # 마지막 회전교차로 진입 투표 수 (디버그/로그)
        self._exit_votes = 0          # 마지막 회전교차로 탈출 투표 수 (디버그/로그)
        self._entry_lock_active = False  # 진입측 락온이 걸려 있는 동안 True (one-shot)
        self._exit_lock_t = 0.0          # 탈출 락 남은 시간(초) — RA 탈출 순간 장전
        # DRIVE 소실 폴백용: 최근 조향의 EMA (07-12 run47 v47.6 실증 — 탈출로
        # 노랑->흰 전환부에서 소실 시 '직진' 폴백이 커브를 뚫고 이탈. 소실 중에는
        # 직전 조향 경향을 유지하고, 소실이 길어지면 서서히 중앙으로 감쇠).
        self._steer_hold = None
        self._gate_armed = False      # 가로선이 충분히 꺼진 뒤에만 다음 카운트 무장
        self._gate_off_t = 0.0        # 가로선 연속 OFF 시간 (재무장 판정용)
        self._gate_on_t = 0.0         # 가로선 연속 ON 시간 (진단용)
        self._gate_cluster_on = 0.0   # 현재 군집 누적 ON (gap 병합)
        self._gate_cluster_counted = False
        self._gate_in_cluster = False # 군집 진행 중 플래그 (출생 시점 판정용)
        self._gate_cluster_void = False  # 블랭크 중 태어난 군집 = 통째 무효
        # 브랜치 락온 탈출용 게이트 통과 카운터 (노란 가로선 or 점선 개구부의
        # 상승엣지 = 게이트 1회 통과). roundabout_exit_gates 회 도달 시 출구 브랜치로 락온.
        self._gate_count = 0          # 이번 회전에서 센 게이트 통과 횟수
        self._gate_just_counted = False  # 이번 프레임에 카운트 성립 (직발화용)
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
        if self._merge_bridge_t > 0.0 and lane.lane_found:
            self._merge_bridge_t = max(0.0, self._merge_bridge_t - dt)
        if lane.lane_found:
            # 07-11 방안1: fork 편향(_turn_bias) 을 조향 경로에서 제거. 목표측 가산은
            # 차선 소실 시 증발해 탈출 개구부에서 무력했다 (run20: 우회전 1.5초 사망).
            # 탈출 기동은 탈출 락(post-PID, 아래 DRIVE 블록)이 전담한다.
            # turn_latch 는 텔레메트리 FORK 마커로만 유지.
            target = (lane.offset + self._branch_bias(lane)
                      + self._curve_bias(lane))
        else:
            # 차선 놓침: DRIVE 는 '직전 조향 EMA' 유지 (직진 폴백이 커브를 뚫던
            # run47 전환부 이탈 대응), 3s 시정수로 중앙 수렴. RA 는 ra_blind_bias
            # 전담 (중복 가산 방지).
            if self.state == State.DRIVE and self._merge_bridge_t > 0.0:
                # 병합 브리지 (07-13): 탈출 직후 무차선 구간은 직전 조향 관성이 아니라
                # 기하 사전값(병합은 항상 완만한 좌호)으로 통과한다. run83/84 우측 표류
                # + 분기 점선 오추종 좌이탈의 공통 봉합.
                self._merge_bridge_t = max(0.0, self._merge_bridge_t - dt)
                return float(self.cfg['steer_center']
                             + self.cfg.get('merge_blind_bias', 0.0))
            if self.state == State.DRIVE and self._steer_hold is not None:
                center = self.cfg['steer_center']
                self._steer_hold += (center - self._steer_hold) * min(1.0, dt / 3.0)
                return float(self._steer_hold)
            target = 0.0  # (DRIVE 외 상태) 직진하며 재획득을 기다린다
        correction = self.pid.update(target, dt)
        # steer_scale 은 [-1,1] 보정값을 키트의 조향 범위로 매핑한다.
        # 차가 반대 방향으로 조향하면 이 값의 부호를 뒤집을 것.
        steer = float(self.cfg['steer_center'] + correction * self.cfg['steer_scale'])
        if lane.lane_found:
            # 소실 폴백용 EMA 갱신 (과도 스파이크 완화를 위해 ±0.30 클램프)
            c = self.cfg['steer_center']
            s = max(c - 0.30, min(c + 0.30, steer))
            self._steer_hold = (s if self._steer_hold is None
                                else 0.3 * s + 0.7 * self._steer_hold)
        return steer

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
        0 = 기능 off (red_ratio 를 발행하지 않는 구버전 perception 과도 호환).

        회전교차로를 아직 안 지났으면 무조건 False — 빨간 도로는 RA 이후 구간에만
        존재하므로, 그 전의 빨강은 전부 오검출이다 (out 코스는 RA 가 없어 게이트 없음)."""
        thr = self.cfg.get('red_slow_ratio', 0.0)
        if thr <= 0.0:
            return False
        if self.cfg.get('course', 'in') == 'in' and not self.roundabout_done:
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
        # ROUNDABOUT 을 떠나는 모든 경로(정상 탈출/폴백/타임아웃)에서 한 번만 세운다.
        # 이후부터 빨간 노면 감지가 무장된다 (_in_red_zone).
        if self.state == State.ROUNDABOUT and state != State.ROUNDABOUT:
            self.roundabout_done = True
            # 탈출 락 장전 (07-11 run20, 방안1): 탈출 분기점도 개구부라 nl=0 이 빈발.
            # 시간창 동안 post-PID 우측 편향을 유지한다 (합류부 보조와 같은 원리 —
            # 탈출 후엔 yaw 적분이 멈추므로 위치창 대신 시간창). 모든 탈출 경로 공통.
            self._exit_lock_t = float(self.cfg.get('exit_lock_release_s', 0.0))
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
        # 속도 스케일 확정 (기준 run54 = 20.5s 밴드에서 절대값 3.6/18.5 가 실측 정합).
        # 초록불 없이 진입한 비정상 경로(race_t~0)는 클램프 하한이 흡수.
        ref = max(1e-3, float(self.cfg.get('ra_ref_drive_s', 20.5)))
        self._speed_scale = float(min(1.3, max(0.6, self.race_t / ref)))
        # 게이트(가로선) 카운터: 진입 정지선 재카운트를 3중으로 차단 —
        #  ① 블랭크: gate_blank_s(기본 min_loop 연동) 동안 카운트 금지
        #  ② 재무장(arm): 가로선이 gate_rearm_s 이상 "연속으로 꺼져" 있어야
        #     다음 상승엣지를 셀 수 있음 (한두 프레임 깜빡임으로는 무장 안 됨)
        #  ③ 지속: gate_sustain_s 연속 ON 이어야 카운트 (blip 배제)
        self._gate_count = 0
        self._gate_cluster_on = 0.0
        self._gate_cluster_counted = False
        self._gate_in_cluster = False
        self._gate_cluster_void = False
        # 블랭크 무스케일 (07-13): 임무가 '잔상 출생 시점 덮기'로 축소 — 시작
        # 시점은 속도 무관. 스케일 버전은 빠른 접근에서 오히려 짧아져 위험.
        self._gate_cd = float(self.cfg.get('gate_blank_s', self.cfg['min_loop_time_s']))
        self._gate_armed = False
        self._gate_off_t = 0.0
        self._gate_on_t = 0.0
        # 진입측 락온 (one-shot): RA 켜지는 순간 딱 1회, 링 순환 방향 브랜치로
        # 시드를 밀어 진입 갈림길에서 링을 타게 한다. 해제는 fork 재수렴 또는
        # entry_lock_release_s 중 먼저 오는 쪽 (아래 ROUNDABOUT 블록에서).
        # 이후 링 주행 중 fork 가 오발동해도 latch 를 다시 세우는 코드가 없으므로
        # 재락온은 구조적으로 불가능하다.
        entry_side = self.cfg.get('roundabout_entry_side')
        self.turn_latch = entry_side if entry_side in ('left', 'right') else None
        self.turn_latch_age = 0.0
        self._fork_seen = False
        self._fork_absent_t = 0.0
        self._entry_lock_active = self.turn_latch is not None

    # ---------- 메인 틱 ----------
    def step(self, lane, aruco, dets, dt):
        """(steering, throttle, state_name) 을 반환한다."""
        center = self.cfg['steer_center']
        stop = self.cfg['stop_throttle']
        # 재획득 규칙 무장 여부 (decision -> perception, /decision/merge_zone 토픽).
        # RA+reacq_arm_s 경과 시 True — perception 의 "nl 0->1 재획득 = 우측 경계"
        # 분류 규칙을 무장시킨다. (07-12: yaw 위치창 스코프를 대체. 토픽명은
        # perception 리빌드를 피하려고 merge_zone 그대로 둠.)
        self.reacq_armed = False

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
        # 스로틀 동적 보정 (07-12): 같은 스로틀도 팩 상태로 런마다 속도가 널뜀
        # (run56 부족/run57~58 과속, 전부 0.19~0.21). G->Y래치(분기 노랑 첫 지속
        # 감지) 소요시간으로 이 런의 속도를 측정해 이후 순항 스로틀에 보정을
        # 가산한다. 실측: 적정 4.4~4.7s / 과속 3.2~3.6s — 신호가 킥 과도구간을
        # 포함해 속도차를 ~3배 압축하므로 게인이 큼 (run57 역산 0.04~0.07).
        if (self.state == State.DRIVE and not self.roundabout_done
                and not self._latch_seen):
            if lane.yellow_ratio >= self.cfg.get('y_latch_ratio', 0.02):
                self._y_run += 1
                if self._y_run >= int(self.cfg.get('y_latch_frames', 10)):
                    self._latch_seen = True
                    ref = max(1e-3, float(self.cfg.get('throttle_ref_latch_s', 4.6)))
                    dev = self.race_t / ref - 1.0
                    lim = float(self.cfg.get('throttle_adapt_max', 0.015))
                    self.throttle_adj = float(min(lim, max(-lim,
                        float(self.cfg.get('throttle_adapt_gain', 0.06)) * dev)))
            else:
                self._y_run = 0

        # 갈림길: 표결 확정(confirmed_fork_direction) = 도착 -> 방향 latch, 해제는
        # 기하(fork 켜졌다 꺼짐 = 재수렴). fork_hold_s 는 failsafe 상한으로만.
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
                self._fork_absent_t = 0.0
            else:
                self._fork_absent_t += dt
            # 재수렴 디바운스: fork '0.3s 연속 부재'여야 재수렴, 락온 최소 0.5s 유지
            # (1프레임 blip 재수렴으로 진입 락이 같은 틱에 풀리던 것 실측 대응).
            reconverged = (self._fork_seen
                           and self._fork_absent_t >= 0.3
                           and self.turn_latch_age >= 0.5)
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
        # 07-14: area 분기에도 RA 완료 게이트 — in 코스에서 마커는 항상 RA 뒤 구간에만
        # 존재하므로(위 주석), RA 전 큰 마커 오검출이 OBSTACLE_STOP→클리어→
        # obstacle_done 조기 세팅→빨간불 조기 무장으로 이어지는 체인을 차단한다.
        marker_armed = (self.cfg.get('course', 'in') != 'in' or self.roundabout_done)
        if self.state == State.DRIVE and aruco.detected and marker_armed and (
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
            # 탈출 락: 탈출 직후 시간창 동안 '소실 프레임에만' 우측 호 (차선 잡히면
            # 즉시 PID 인계). 무조건 가산은 추종 기회를 없앰(run25). 양수 = 탈출측(우).
            if self._exit_lock_t > 0.0:
                self._exit_lock_t -= dt
                if not lane.lane_found:
                    steer = max(-1.0, min(1.0, steer + self.cfg['turn_direction']
                                          * self.cfg.get('exit_steer_bias', 0.0)))
            # 회전교차로 진입 (In 코스, 한 번 완료하기 전까지만).
            # 큰 틀(인코스 플로우): 노란 추종으로 링까지 온 뒤, 정지선처럼 보이는
            # 노란 가로선(yellow_crossline)을 "처음" 만나면 ROUNDABOUT on.
            # 오검 방지 장치 둘은 유지 — yellow_ratio 게이트(흰 외곽 루프에서
            # 안 걸리게) + enter_sustain_s 지속 debounce(한 프레임 깜빡임 무시).
            # 탈출은 ROUNDABOUT 블록의 게이트 카운트: 같은 가로선을 "한 번 더"
            # 만나면 출구 브랜치 락온으로 나간다.
            # 07-11: 인코스 '입구'에서 노란 차선이 비스듬히 가로지르며 정지선으로
            # 오인돼 출발 5.9초 만에 오진입했다. 대회는 항상 출발선에서 시작하고
            # 진짜 정지선 도달은 12초+ (auto2/3 실측 12.4~12.9초) 이므로 시간 가드.
            # 중간 배치 테스트 시 ra_min_drive_s=0 으로 끌 것.
            armed_t = self.race_t >= self.cfg.get('ra_min_drive_s', 0.0)
            if self.cfg['course'] == 'in' and not self.roundabout_done and armed_t:
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
            # 노란 구간(DRIVE[Y] = 회전교차로 접근/갈림길)은 검출·조향이 가장 어려운
            # 구간이라 상한을 따로 둔다 (07-11: 노란 접근에서의 요동·이탈이 반복돼
            # 흰 구간과 속도를 분리). yellow_ratio 는 FOLLOW-Y 전환과 같은 문턱을 쓴다.
            ydt = self.cfg.get('yellow_drive_throttle', 0.0)
            if ydt > 0.0 and lane.yellow_ratio >= self.cfg.get('yellow_slow_ratio', 0.03):
                throttle = min(throttle, ydt)
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
            # 진입 피드포워드 (07-11): 진입 갈림길은 코스에서 가장 급한 좌회전인데
            # RA 에선 곡률 피드포워드가 없어 PID 가 뒤쫓기만 하다 3개 런 연속
            # 언더스티어로 링 선을 놓쳤다 (진입 직후 nl=0 -> off -0.94, steer 포화).
            # 진입 락온이 살아있는 첫 ~2초 동안만 회전 방향으로 조향을 미리 얹는다.
            if self._entry_lock_active:
                steer = steer + self.cfg['turn_direction'] * self.cfg.get('entry_steer_bias', 0.0)
            # 재획득 규칙 무장: RA+reacq_arm_s 후 nl 0->1 재획득 선 = 우측(바깥) 경계.
            # 실측(런19~26): 5s 후 0->1 은 합류부에서만 발생(순항 오발 0건), 2->1 은
            # 순항 중 일상이라 제외. 구 yaw 위치창은 속도 의존(창 밀림)으로 폐기 —
            # 시간 하한 5s 는 진입 잔재(<2.6s)와 합류부(~14s+) 사이라 사실상 면역.
            self.reacq_armed = (self.circle_t
                                >= self.cfg.get('reacq_arm_s', 5.0))
            # RA 맹목 폴백 (07-11 run21): 링 위에서 차선 소실 시 기본값 '직진(0.26)'은
            # 항상 오답 — RA+3.5s 소실 후 19초 직진 이탈 실측. 소실 프레임에는 링 유지
            # 호(-0.15 = 링 실측 요구 편차)를 얹는다. 단 진입 락이 이미 활성인
            # 프레임에는 중복 가산하지 않는다 (겹치면 과회전). 정상 추종 프레임 미적용.
            if not lane.lane_found and not self._entry_lock_active:
                steer = steer + self.cfg['turn_direction'] * self.cfg.get('ra_blind_bias', 0.0)
            steer = max(-1.0, min(1.0, steer))

            # (1) 조향 적분 yaw proxy (IMU 대체). 속도가 거의 일정하면 누적 yaw 는
            # 회전 방향 조향 편향량의 합에 비례(∝)한다.
            # 임계값은 실제 트랙에서 캘리브레이션한다 (yaw_lap_threshold).
            # 07-11: 이 차의 조향 규약은 steer>center=좌 (실측 3중 검증). turn_direction
            # (-1=CCW) 가정과 반대라 부호를 뒤집는다 — 안 뒤집으면 CCW 링에서 좌조향이
            # 음수 defl 이 되어 yaw_proxy 가 영영 0 (백업 표결의 yaw 표가 못 나옴).
            defl = -self.cfg['turn_direction'] * (steer - center)
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

            # (2b) 게이트 = 정지선 '군집' 카운터 v2 (07-13): 링 위 가로선 순서는
            # 물리 불변량 — 진입선 잔상 -> 반대편 입구(군집#1) -> 우리 입구(군집#2
            # = 탈출). 속도/배터리/스로틀과 무관 (yaw 문턱 방식의 연쇄 실패
            # run55/64/65/66/68/72 대체).
            #   군집: 목격 간격 < gate_cluster_gap_s 병합 (고속 파편 0.1s x4,
            #         입구 2조각 1.3~1.6s 실측 대응)
            #   카운트: 군집 누적 ON >= gate_cluster_on_s (링 중간 3프레임 약피처
            #           max 0.16s 배제 / 입구 min 0.21s 통과)
            #   잔상: 길이가 링 속도 따라 4~9.3s 로 변해 고정 블랭크로 못 덮는다
            #         (run69 오카운트) -> '블랭크 중 태어난 군집은 통째 무효'.
            #         시작 시점은 속도 무관하게 항상 RA+0 부근이라는 불변량 이용.
            self._gate_just_counted = False
            if self._gate_cd > 0.0:
                self._gate_cd = max(0.0, self._gate_cd - dt)
            gate_now = bool(lane.yellow_crossline)
            if gate_now:
                if not self._gate_in_cluster:
                    self._gate_in_cluster = True
                    self._gate_cluster_void = self._gate_cd > 0.0
                self._gate_off_t = 0.0
                self._gate_on_t += dt
                self._gate_cluster_on += dt
            else:
                self._gate_on_t = 0.0
                self._gate_off_t += dt
                if self._gate_off_t >= self.cfg.get('gate_rearm_s', 0.5):
                    self._gate_armed = True
                if self._gate_off_t >= self.cfg.get('gate_cluster_gap_s', 1.8):
                    self._gate_cluster_on = 0.0        # 군집 종료
                    self._gate_cluster_counted = False
                    self._gate_in_cluster = False
                    self._gate_cluster_void = False
            # yaw 는 위생 하한으로 강등 (1.0 = "조금이라도 돌았다"; 위치 판별은
            # 군집 순서가 담당)
            yaw_ok = (self.yaw_proxy
                      >= self.cfg.get('yaw_gate_min', 0.0) * self._speed_scale)
            if (not self._gate_cluster_counted and not self._gate_cluster_void
                    and self._gate_cluster_on
                    >= self.cfg.get('gate_cluster_on_s', 0.18)
                    and self._gate_cd <= 0.0 and self._gate_armed and yaw_ok):
                self._gate_count += 1
                self._gate_just_counted = True
                self._gate_cluster_counted = True
                self._gate_cd = self.cfg.get('crossline_cooldown_s', 2.0)
                self._gate_armed = False

            # ----- 바퀴 수 판정 -----
            # 절대 하한: min_loop_time 전에는 절대 탈출하지 않는다 (한 바퀴 미달은
            # 미션 실패이고, 진입 쪽 노란 가로선을 출구로 오인하는 것도 막아준다).
            # 주 탈출 = 게이트 카운트. 백업 = 세 가지 추정치(yaw proxy, 시간, 노란
            # 가로선 재등장) 중 >= lap_votes_needed 개 동의.
            # 편향은 늦게 나가는 쪽으로, 절대 일찍 나가지 않는다.
            # 직발화 (07-13 run92/93, 사용자 설계): STOPLINE 청정 스트림에서는
            # 잔상(void)·B(분류기 침묵, ~0.07s)가 걸러져 카운트되는 첫 군집 =
            # A 재도달. 카운트 성립 그 프레임에 즉시 발화한다. min_loop 미적용
            # (run93 후 사용자 결정 — 조기 래치 시 A1차 무랩 탈출 리스크 인지).
            # 고속 밴드 A 체류 0.17s 대응은 gate_cluster_on_s 0.12 (aux.sh).
            # ⚠️ B 누출 스트림(stopline_mode 0)에서는 B 가 #1 로 카운트되므로
            # 이 경로가 B 오발을 만든다 — 반드시 stopline_mode 1 과 함께 켤 것.
            if (int(self.cfg.get('ra_direct_fire', 0))
                    and self._gate_just_counted):
                self.turn_latch = self.cfg['roundabout_exit_side']
                self.turn_latch_age = 0.0
                self._fork_seen = False
                self.roundabout_done = True
                self._merge_bridge_t = float(self.cfg.get('merge_bridge_s', 0.0))
                self._enter(State.DRIVE)
                return steer, self.cfg['slow_throttle'], self.state.value

            if self.circle_t >= self.cfg['min_loop_time_s'] * self._speed_scale:
                # 주(PRIMARY) 탈출: 출구 게이트를 충분히 통과함 -> 출구 브랜치로 락온
                # (표지판 갈림길과 같은 메커니즘). 그러면 차선 추종이 링에 다시 붙는
                # 대신 개구부 밖으로 조향한다. turn_latch 는 fork_dir 로 perception 에
                # publish 되며 위의 fork 재수렴 로직으로 해제된다.
                if self._gate_count >= self.cfg['roundabout_exit_gates']:
                    self.turn_latch = self.cfg['roundabout_exit_side']
                    self.turn_latch_age = 0.0
                    self._fork_seen = False
                    self.roundabout_done = True
                    self._merge_bridge_t = float(self.cfg.get('merge_bridge_s', 0.0))
                    self._enter(State.DRIVE)
                    return steer, self.cfg['slow_throttle'], self.state.value

                # FAILSAFE: 게이트 실패 대비 3중 2 표결 백업 (junction 표는 오검출로
                # 제외). 주 탈출과 동일하게 출구측 락온 — 락온 없인 FOLLOW-Y 가 링을
                # 무한 순환. 링 중간에 걸려도 시드 밀기는 무해.
                yaw_done = (self.yaw_proxy
                            >= self.cfg['yaw_lap_threshold'] * self._speed_scale)
                time_done = (self.circle_t
                             >= self.cfg['nominal_loop_time_s'] * self._speed_scale)
                # 가로선 표도 게이트와 같은 재등장 규율 적용: 재무장(_gate_armed,
                # 0.5s 이상 사라졌다 다시 나타남) 상태에서 보일 때만 인정한다.
                # 진입선이 시야에 계속 남아 있는 것(정차/저속 통과)은 표가 아니다.
                # cross 표는 '반대편 입구를 이미 세었고'(군집 이력) 현재 목격이
                # 무효 군집도, 방금 카운트된 그 군집도 아닐 때만 — 긴 군집 내부
                # 공백(>=0.5s)에 재무장이 일어나면 자기 군집 꼬리가 count>=1 을
                # 자충족해 표결 오발하던 것(run74 파란 꼬리 발화) 봉쇄.
                cross_done = (bool(lane.yellow_crossline) and self._gate_armed
                              and not self._gate_cluster_void
                              and not self._gate_cluster_counted
                              and self._gate_count
                              >= int(self.cfg['roundabout_exit_gates']) - 1
                              and self.yaw_proxy >= self.cfg.get('yaw_gate_min', 0.0)
                              * self._speed_scale)
                votes = int(yaw_done) + int(time_done) + int(cross_done)
                self._exit_votes = votes
                if votes >= self.cfg['lap_votes_needed']:
                    self.turn_latch = self.cfg['roundabout_exit_side']
                    self.turn_latch_age = 0.0
                    self._fork_seen = False
                    self.roundabout_done = True
                    self._merge_bridge_t = float(self.cfg.get('merge_bridge_s', 0.0))
                    self._enter(State.DRIVE)
                    return steer, self.cfg['slow_throttle'], self.state.value

            # 최후 failsafe: 모든 추정치가 실패하면 강제 탈출 (역시 출구측 락온)
            if self.circle_t >= self.cfg['max_loop_time_s']:
                self.turn_latch = self.cfg['roundabout_exit_side']
                self.turn_latch_age = 0.0
                self._fork_seen = False
                self.roundabout_done = True
                self._merge_bridge_t = float(self.cfg.get('merge_bridge_s', 0.0))
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
