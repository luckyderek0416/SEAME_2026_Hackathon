"""PD 제어 — 비전이 준 '목표 각도(오차)'를 부드럽게 조향값으로 변환.

P(비례): 오차에 비례해 꺾음 (오차 크면 많이 꺾음)
D(미분): 오차의 '변화'를 보고 미리 누름 → 출렁임/오버슈트 억제

출력 = 조향값 -1.0(좌) ~ +1.0(우). (D-Racer set_steering_percent 규격)
※ 팀이 별도 PID를 쓴다면 이 노드는 각도만 넘기고 PD를 꺼도 됨(구조상 둘 다 가능).

조향엔 별도 센서 피드백이 없어서, 오차 = 비전 각도(정규화) 기준으로 PD를 건다
(라인추종에서 흔한 방식). Kp/Kd 는 실제 주행으로 튜닝.
"""


class PDController:
    def __init__(self, kp=1.0, kd=0.1):
        self.kp = kp          # 비례 게인 (작으면 둔함, 크면 예민/출렁)
        self.kd = kd          # 미분 게인 (출렁임 억제)
        self.prev_error = 0.0

    def reset(self):
        self.prev_error = 0.0

    def step(self, error, dt=None):
        """error(-1~1) -> steering(-1~1). dt 주면 미분을 시간기준으로."""
        d = error - self.prev_error
        if dt and dt > 0:
            d = d / dt
        self.prev_error = error
        out = self.kp * error + self.kd * d
        return max(-1.0, min(1.0, out))
