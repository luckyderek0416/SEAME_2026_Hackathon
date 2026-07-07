"""출력 클램프와 적분 anti-windup 을 갖춘 최소 구현 PID.

차선 추종에서는 보통 PD 제어기(ki=0)로 쓴다.
튜닝 순서: 먼저 Kp, 그다음 Kd 로 흔들림(wobble)을 잡고, 차가 한쪽으로
일관되게 치우칠 때만 아주 작은 Ki 를 추가한다.
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
