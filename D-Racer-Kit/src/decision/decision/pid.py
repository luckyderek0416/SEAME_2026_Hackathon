"""Minimal PID with output clamp and integral anti-windup.

For lane following you will usually run this as a PD controller (ki=0).
Tune in this order:  Kp first, then Kd to kill the wobble, add tiny Ki
only if the car drifts consistently to one side.
"""


class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, out_limit=1.0, i_limit=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        if dt <= 0.0:
            dt = 1e-3
        self.integral += error * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.out_limit, min(self.out_limit, out))
