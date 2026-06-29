"""RaceStateMachine: the brain that turns perception into driving.

Steering ALWAYS comes from the lane PID (OpenCV offset -> steer).
The state only changes throttle and which mission behaviour is active:

  WAIT_GREEN  -> stopped on the line; YOLO sees green_light -> go
  DRIVE       -> lane-follow at drive speed (S-curve, straights, fork)
  ROUNDABOUT  -> lane-follow slowly around the circle, ~1 lap, then exit
  OBSTACLE_STOP -> ArUco marker in view -> full stop until it clears
  FINISH      -> past the line, waiting for red_light -> full stop
  DONE        -> stopped for good

Course selection ('out' vs 'in') decides whether we go DRIVE or ROUNDABOUT
right after the start.

NOTE on the roundabout lap counter: this skeleton uses a simple time-based
proxy (roundabout_seconds). If your board has an IMU, replace
`_roundabout_done()` with yaw-angle integration (exit once accumulated yaw
>= ~330 deg) -- that is far more reliable than time.
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
        self.state = State.WAIT_GREEN

        self.turn_latch = None        # 'left' / 'right' once a turn sign is seen
        self.green_count = 0
        self.red_count = 0
        self.marker_gone = 0
        self.roundabout_t = 0.0
        self._resume_state = State.DRIVE

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
            target = lane.offset + self._turn_bias()
        else:
            target = 0.0  # lost the lane: go straight, let it reacquire
        correction = self.pid.update(target, dt)
        # steer_scale maps the [-1,1] correction onto the kit's steering range.
        # Flip its SIGN if the car steers the wrong way.
        return float(self.cfg['steer_center'] + correction * self.cfg['steer_scale'])

    def _turn_bias(self):
        # Bias the lane target toward the chosen branch at the fork (Out course).
        # Simplification: applies whenever a sign is latched. Ideally geofence
        # this to the fork zone only.
        if self.turn_latch == 'left':
            return -self.cfg['fork_bias']
        if self.turn_latch == 'right':
            return +self.cfg['fork_bias']
        return 0.0

    def _enter(self, state):
        self.state = state
        self.pid.reset()
        self.green_count = 0
        self.red_count = 0
        self.marker_gone = 0
        if state == State.ROUNDABOUT:
            self.roundabout_t = 0.0

    def _roundabout_done(self, dt):
        # TODO: replace with IMU yaw integration if available.
        self.roundabout_t += dt
        return self.roundabout_t >= self.cfg['roundabout_seconds']

    # ---------- main tick ----------
    def step(self, lane, aruco, dets, dt):
        """Return (steering, throttle, state_name)."""
        center = self.cfg['steer_center']
        stop = self.cfg['stop_throttle']

        # latch a turn sign whenever one is clearly seen
        if self._seen(dets, 'left_sign'):
            self.turn_latch = 'left'
        elif self._seen(dets, 'right_sign'):
            self.turn_latch = 'right'

        # ----- WAIT_GREEN -----
        if self.state == State.WAIT_GREEN:
            self.green_count = self.green_count + 1 if self._seen(dets, 'green_light') else 0
            if self.green_count >= self.cfg['green_frames']:
                self._enter(State.ROUNDABOUT if self.cfg['course'] == 'in' else State.DRIVE)
            return center, stop, self.state.value

        # ----- global obstacle override (mission has priority over driving) -----
        if (self.state != State.OBSTACLE_STOP and aruco.detected
                and aruco.area_ratio >= self.cfg['marker_area_trigger']):
            self._resume_state = self.state
            self._enter(State.OBSTACLE_STOP)

        if self.state == State.OBSTACLE_STOP:
            self.marker_gone = 0 if aruco.detected else self.marker_gone + 1
            if self.marker_gone >= self.cfg['marker_clear_frames']:
                self._enter(self._resume_state)   # resume where we were
            return center, stop, self.state.value   # hold still while marker is up

        steer = self._lane_steer(lane, dt)

        # ----- DRIVE -----
        if self.state == State.DRIVE:
            self.red_count = self.red_count + 1 if self._seen(dets, 'red_light') else 0
            if self.red_count >= self.cfg['red_frames']:
                self._enter(State.FINISH)
                return center, stop, self.state.value
            return steer, self.cfg['drive_throttle'], self.state.value

        # ----- ROUNDABOUT (In course) -----
        if self.state == State.ROUNDABOUT:
            if self._roundabout_done(dt):
                self._enter(State.DRIVE)   # exit the circle, carry on to obstacle + finish
            return steer, self.cfg['slow_throttle'], self.state.value

        # ----- FINISH -----
        if self.state == State.FINISH:
            if self._seen(dets, 'red_light'):
                self._enter(State.DONE)
            return center, stop, self.state.value

        # ----- DONE -----
        return center, stop, self.state.value
