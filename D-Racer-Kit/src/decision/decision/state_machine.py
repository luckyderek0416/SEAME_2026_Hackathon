"""RaceStateMachine: the brain that turns perception into driving.

Steering ALWAYS comes from the lane PID (OpenCV offset -> steer).
The state only changes throttle and which mission behaviour is active:

  WAIT_GREEN    -> stopped on the line; YOLO sees green_light -> go
  DRIVE         -> lane-follow at drive speed (straights, S-curve, fork).
                   For the 'in' course it also watches for the roundabout
                   entry (a sustained sharp curve) and switches to ROUNDABOUT.
  ROUNDABOUT    -> circle the ring (lane-follow + a turn bias so we don't
                   leave early) and count laps by JUNCTION DETECTION
                   (the outer ring line opens up at the entry/exit). Exit
                   only after >= target_loops AND min_loop_time -- the rules
                   fail the run for leaving before a full lap.
  OBSTACLE_STOP -> ArUco marker in view -> full stop until it clears
  FINISH        -> past the line, waiting for red_light -> full stop
  DONE          -> stopped for good

Lap counting (NO IMU, NO marker): three independent estimates vote.
  (1) junction  - the outer ring line reopens at the entry/exit point
  (2) yaw proxy - integral of steering deflection (IMU-free heading estimate;
                  at ~constant speed, accumulated yaw is proportional to it)
  (3) time      - circle time reaches the measured one-lap time
We NEVER exit before min_loop_time (an under-shoot fails the mission), then
exit once >= lap_votes_needed of the three agree. max_loop_time is the
failsafe. Bias every threshold so the car exits LATE, never early: the rules
allow >= 1 lap, so a little over-rotation is free but < 1 lap fails.
Calibrate yaw_lap_threshold and nominal_loop_time_s on the real track.
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
        # skip_missions: pure lane-following test mode (no lights/roundabout/obstacle).
        self.state = State.DRIVE if cfg.get('skip_missions') else State.WAIT_GREEN

        self.turn_latch = None        # 'left' / 'right' once a turn sign is confirmed
        self.turn_latch_age = 0.0     # seconds the latch has been held (failsafe timer)
        self._fork_seen = False       # perception flagged the fork geometry during this latch
        # decision_node 가 표결(deque)로 확정해 매 틱 넣어주는 갈림길 방향. None = 미확정.
        self.confirmed_fork_direction = None
        self.green_count = 0
        self.red_count = 0
        self.marker_gone = 0
        self._resume_state = State.DRIVE

        # roundabout (junction-count) state
        self.roundabout_done = False  # already completed the circle (don't re-enter)
        self.enter_acc = 0.0          # sustained-curve accumulator for entry detection
        self.circle_t = 0.0           # time spent in the current circle
        self.cooldown = 0.0           # ignore junctions right after entering / counting
        self.loops = 0
        self.side_present = False     # outer ring line currently visible (not a gap)
        self.yaw_proxy = 0.0          # IMU-free heading estimate (integral of steering)
        self._entry_votes = 0         # last roundabout-entry vote count (debug/log)
        self._exit_votes = 0          # last roundabout-exit vote count (debug/log)
        # gate-passage counter for the branch-lock-on exit (노란 가로선 or 점선 개구부의
        # 상승엣지 = 게이트 1회 통과). roundabout_exit_gates 회 도달 시 출구 브랜치로 락온.
        self._gate_prev = False       # previous-frame gate signal (edge detect)
        self._gate_count = 0          # gate passages counted this circle
        self._gate_cd = 0.0           # debounce cooldown between counts

    # ---------- detection helper ----------
    def _seen(self, dets, label):
        c = self.cfg['conf_threshold']
        for d in dets:
            if d.label == label and d.confidence >= c:
                return d
        return None

    # ---------- steering (always lane PID) ----------
    def _lane_steer(self, lane, dt):
        if lane.lane_found:
            target = lane.offset + self._turn_bias() + self._branch_bias(lane)
        else:
            target = 0.0  # lost the lane: go straight, let it reacquire
        correction = self.pid.update(target, dt)
        # steer_scale maps the [-1,1] correction onto the kit's steering range.
        # Flip its SIGN if the car steers the wrong way.
        return float(self.cfg['steer_center'] + correction * self.cfg['steer_scale'])

    def _turn_bias(self):
        # Fork bias applies only while driving; a stale latch in ROUNDABOUT etc.
        # must not tilt the steering.
        if self.state != State.DRIVE:
            return 0.0
        if self.turn_latch == 'left':
            return -self.cfg['fork_bias']
        if self.turn_latch == 'right':
            return +self.cfg['fork_bias']
        return 0.0

    def _branch_bias(self, lane):
        """In/Out fork selection by COLOUR (direction-agnostic). At the fork the In
        path is yellow and the Out path is white. Only in DRIVE, only when a yellow
        branch is actually in view. Steers toward yellow for 'in', away for 'out' —
        works whichever side the yellow appears, so no left/right setup is needed."""
        if self.state != State.DRIVE:
            return 0.0
        if lane.yellow_ratio < self.cfg['branch_yellow_min']:
            return 0.0
        toward_yellow = 1.0 if lane.yellow_offset >= 0.0 else -1.0
        if self.cfg['course'] == 'in':
            return self.cfg['branch_bias'] * toward_yellow      # take the yellow branch
        return -self.cfg['branch_bias'] * toward_yellow         # stay on the white branch

    def _enter(self, state):
        """Generic transition. Does NOT reset roundabout progress, so resuming
        ROUNDABOUT after an obstacle keeps the lap count."""
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
        self.cooldown = self.cfg['junction_cooldown_s']  # ignore the entry junction
        self.side_present = False
        self.yaw_proxy = 0.0
        self._gate_prev = False
        self._gate_count = 0
        self._gate_cd = self.cfg['junction_cooldown_s']  # ignore the entry-side gate

    # ---------- main tick ----------
    def step(self, lane, aruco, dets, dt):
        """Return (steering, throttle, state_name)."""
        center = self.cfg['steer_center']
        stop = self.cfg['stop_throttle']

        # ----- LANE-ONLY test mode (skip_missions) -----
        # Pure lane following: no green/red light, no roundabout, no obstacle, no fork.
        # Just OpenCV offset -> PID -> steer, with curvature-based speed. Use this to
        # isolate and tune lane keeping (kp/kd/HSV) without any mission logic.
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

        # Fork direction comes from decision_node's MULTI-FRAME VOTE (not a single
        # detection), passed in as confirmed_fork_direction. The sign sits BETWEEN the
        # two branches, so "sign confirmed" == "at the fork": latch the direction, then
        # RELEASE on GEOMETRY — when perception's fork flag turns off again (the two
        # branches have re-merged / the road re-narrowed) after having been seen. This
        # replaces the old timer-only hold (which stacked with decision_node's vote hold
        # to ≈2×fork_hold_s). fork_hold_s is kept only as a FAILSAFE ceiling.
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

        # ----- global obstacle override (mission has priority over driving) -----
        if (self.state != State.OBSTACLE_STOP and aruco.detected
                and aruco.area_ratio >= self.cfg['marker_area_trigger']):
            self._resume_state = self.state
            self._enter(State.OBSTACLE_STOP)

        if self.state == State.OBSTACLE_STOP:
            self.marker_gone = 0 if aruco.detected else self.marker_gone + 1
            if self.marker_gone >= self.cfg['marker_clear_frames']:
                self._enter(self._resume_state)   # resume where we were (lap count kept)
            return center, stop, self.state.value

        steer = self._lane_steer(lane, dt)

        # ----- DRIVE -----
        if self.state == State.DRIVE:
            # roundabout entry (In course, only until we've done it once).
            # 2-of-3 VOTE: yellow_crossline (the yellow horizontal bar at the fork),
            # junction (dashed opening), sharp curvature. No single signal can enter
            # on its own, and the yellow_ratio gate keeps a white outer-loop corner
            # from ever triggering it. The old enter_acc/enter_sustain_s debounce
            # is kept: the vote must HOLD for enter_sustain_s before entering.
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
            # curvature-adaptive speed: slow down on curves (rule recommends it, and it
            # cuts lane-departure risk on the S-curve / corners).
            curve = abs(getattr(lane, 'curvature', 0.0))
            throttle = self.cfg['drive_throttle'] * (1.0 - self.cfg['curve_slow'] * curve)
            throttle = max(self.cfg['slow_throttle'], throttle)
            return steer, throttle, self.state.value

        # ----- ROUNDABOUT (no IMU, no marker: vote across 3 lap estimates) -----
        if self.state == State.ROUNDABOUT:
            self.circle_t += dt
            # hold the ring: bias toward the turn direction so lane following
            # does not take the exit branch before a full lap.
            steer = steer + self.cfg['turn_direction'] * self.cfg['circle_steer_bias']
            steer = max(-1.0, min(1.0, steer))

            # (1) STEERING-INTEGRAL yaw proxy (IMU replacement). At ~constant speed
            # accumulated yaw ∝ sum of steering deflection in the turn direction.
            # The threshold is calibrated on the real track (yaw_lap_threshold).
            defl = self.cfg['turn_direction'] * (steer - center)
            if defl > 0.0:
                self.yaw_proxy += defl * dt

            # (2) JUNCTION reappearance count (vision)
            if not lane.junction:
                self.side_present = True   # outer line visible -> on the ring
            elif (self.cooldown <= 0.0 and self.circle_t >= self.cfg['min_loop_time_s']
                  and self.side_present):
                self.loops += 1
                self.cooldown = self.cfg['junction_cooldown_s']
                self.side_present = False

            # (2b) GATE passage counter for the branch-lock-on exit. Count the RISING
            # EDGE of a gate signal (yellow crossline OR dashed junction opening) with a
            # cooldown so one physical crossing = one count (not one per frame). This is
            # the explicit-exit trigger the fork uses in spirit: pass the exit gate
            # roundabout_exit_gates times, then lock onto the exit branch and leave.
            if self._gate_cd > 0.0:
                self._gate_cd = max(0.0, self._gate_cd - dt)
            gate_now = bool(lane.yellow_crossline or lane.junction)
            if gate_now and not self._gate_prev and self._gate_cd <= 0.0:
                self._gate_count += 1
                self._gate_cd = self.cfg['junction_cooldown_s']
            self._gate_prev = gate_now

            # ----- lap decision -----
            # HARD floor: never exit before min_loop_time (an under-shoot fails the
            # mission; it also stops the entry-side yellow crossline from being
            # mistaken for the exit). After that, exit when >= lap_votes_needed of
            # the FOUR independent estimates agree (junction lap count, yaw proxy,
            # time, yellow crossline back in view). Bias is to exit LATE, never early.
            if self.circle_t >= self.cfg['min_loop_time_s']:
                # PRIMARY exit: passed the exit gate enough times -> lock onto the exit
                # branch (same mechanism as the sign fork) so lane following steers OUT
                # the opening instead of re-hugging the ring. turn_latch is published to
                # perception as fork_dir and released by the fork-reconverge logic above.
                if self._gate_count >= self.cfg['roundabout_exit_gates']:
                    self.turn_latch = self.cfg['roundabout_exit_side']
                    self.turn_latch_age = 0.0
                    self._fork_seen = False
                    self.roundabout_done = True
                    self._enter(State.DRIVE)
                    return steer, self.cfg['slow_throttle'], self.state.value

                # FAILSAFE: gate detection may miss a pass (a no-give-up mission must
                # still exit). Keep the 2-of-4 estimate vote as a backup (no branch
                # lock-on; just release the ring hold).
                junction_done = self.loops >= self.cfg['target_loops']
                yaw_done = self.yaw_proxy >= self.cfg['yaw_lap_threshold']
                time_done = self.circle_t >= self.cfg['nominal_loop_time_s']
                cross_done = bool(lane.yellow_crossline)   # exit fork's yellow bar in view
                votes = int(junction_done) + int(yaw_done) + int(time_done) + int(cross_done)
                self._exit_votes = votes
                if votes >= self.cfg['lap_votes_needed']:
                    self.roundabout_done = True
                    self._enter(State.DRIVE)
                    return steer, self.cfg['slow_throttle'], self.state.value

            # failsafe: force exit if all estimates failed
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
