# D-Racer 자율주행 스택 — 전체 로직 (SEA:ME 2026)

TOPST D-Racer-Kit 기본 노드(`camera`, `control`, `joystick`, `monitor`, `battery`)
위에 자율주행 노드(`perception`, `inference`, `decision` + msgs)를 얹은 구조.
**2026-07-03 코드 기준.** 미구현/리스크는 루트의 `MISSING_FEATURES.md` 참고.

## 데이터 흐름

![dataflow](docs/dataflow.png)

```
camera_node ─/camera/image/compressed─┬─> lane_node   ─/perception/lane──────┐
                                      ├─> aruco_node  ─/perception/aruco─────┤
                                      └─> yolo_node   ─/inference/detections─┤
                                                                             ▼
                                            decision_node (state_machine + PID)
                                                             │ /control
joystick_node ─/joystick(E-stop)──> control_node ──PWM(I2C/PCA9685)──> 서보+ESC
```

- **조향은 항상 OpenCV 차선 인식 → PID.** 딥러닝(YOLO)은 미션 인식 전용.
- 모니터: `http://<차량IP>:5000` — 원본 카메라 + 차선 디버그 + 배터리 + /control.

## 패키지 구성

| 패키지 | 내용 | 역할 |
|---|---|---|
| `perception` | `lane_node.py`, `lane_detector.py`, `aruco_node.py` | 차선 인식 + 아루코 |
| `inference` | `yolo_ncnn_node.py` (launch가 쓰는 것), `yolo_node.py`(ultralytics 대안) | 4객체 검출 |
| `decision` | `decision_node.py`, `state_machine.py`, `pid.py`, `launch/` | 판단 + /control 발행 |
| `perception_msgs` / `inference_msgs` | `LaneState`, `ArucoState`, `Detections` | 메시지 |
| (킷 기본) `camera`, `control`, `joystick`, `battery`, `monitor` | | 입출력/모니터링 |

## ① 인지 — lane_node (lane_detector.py)

처리 파이프라인 (한 프레임):

```
프레임 → ROI 자르기(아래 45%, roi_top_ratio=0.55)
       → [BEV ON] perspective warp (birdeye_src_ratio → dst_ratio)   ← 기본 ON, src 튜닝 필요
       → HSV 마스크: 흰색 ∪ 노란색 (mask_mode=hsv)
       → morphology 노이즈 제거 (morph_kernel=3)
       → 멀티밴드 분석 (num_bands=4, 아래→위 = 가까운곳→먼곳):
           [GUIDED ON] 각 밴드는 직전 밴드 중심 ± margin 창에서만 탐색
                       (margin=60px + 밴드당 +10px, 점프 clamp 80px)
           밴드별 좌/우 차선 픽셀 평균 → 밴드 중심, lane_width EMA로 한쪽만 보일 때 보정
       → lane_center 계산:
           [LA ON]  0.7×최근접 밴드 + 0.3×최원거리 밴드 (look-ahead 블렌드)
           [LA OFF] 전체 밴드 가중평균
       → offset = (lane_center − 화면중앙) / (화면폭/2)   ∈ [-1, 1]
       → curvature = 밴드 중심의 아래→위 drift (커브 세기)
       → junction = 바깥 차선의 점선/개방 감지 (갈림길·회전교차로 입구)
       → yellow_ratio / yellow_offset = 노란 픽셀 비율·위치 (In/Out 분기, 회전교차로 진입 게이트)
       → offset EMA 스무딩 (smooth_alpha=0.5) → LaneState 발행
```

- **BEV / GUIDED / LA 셋 다 기본 ON** (`use_birdeye`, `use_guided_band`,
  `use_lookahead_control`). 끄면 원래 파이프라인과 동일. 상세: `docs/lane_bev_guided.md`,
  `docs/lookahead_control.md`.
- lane lost 시: `lane_found=false` + 마지막 offset EMA 유지 → decision이 직진 유지.
- 디버그 이미지(`/perception/lane/debug`): BEV ON이면 위 SRC 뷰(원본+노란 소스 사각형)
  + 아래 분석 뷰. 밴드 창(마젠타), 밴드 중심(주황), lane center(빨강), 상태 텍스트.
- 라이브 튜닝 가능: `ros2 param set /lane_node yellow_hsv_lo "[18,45,110]"` 등 즉시 반영.

## ① 인지 — aruco_node / yolo_node

- **aruco_node**: `cv2.aruco`, DICT_4X4_50(+inverted). 마커 검출 여부와
  화면 면적비(`area_ratio`)를 `/perception/aruco`로 발행. 동적 장애물 전용.
- **yolo_ncnn_node**: NCNN 변환 YOLO(320px, 10Hz 타이머 — 매 프레임 아님).
  4클래스: `red_light, green_light, left_sign, right_sign`. 모델 로드 실패 시
  빈 검출 발행 (차선주행 테스트는 모델 없이도 가능).

## ② 판단 — decision_node + state_machine

30Hz 타이머로 `state_machine.step(lane, aruco, dets, dt)` 호출 → (steer, throttle).

### 조향 (모든 상태 공통)

```
target = offset + turn_bias(갈림길) + branch_bias(In/Out 분기)     # lane lost면 0
correction = PID(target)                    # kp=0.6, ki=0, kd=0.15
steer = steer_center(0.2) + correction × steer_scale(−1.0)
```
- `steer_scale`이 **음수인 게 맞다** (트랙 검증: 이 차는 부호 반대).
- decision_node 출력 직전 후처리 (기본 OFF):
  - `max_steering_delta` > 0: 틱당 조향 변화 제한 (튐 방지)
  - `steer_slow` > 0: |조향| 비례 추가 감속

### 스로틀

```
DRIVE:  drive_throttle(0.20) × (1 − curve_slow(0.5) × |curvature|), 바닥 slow_throttle(0.12)
ROUNDABOUT: slow_throttle 고정
정지 상태들: stop_throttle(0.0)
```

### 상태머신

```
WAIT_GREEN ──(green_light 3프레임 연속)──▶ DRIVE
DRIVE:
  · 좌/우 표지판 보이면 latch → fork_bias(0.4) 편향, 마지막 인식 후 fork_hold_s(6s) 지나면 해제
  · 갈림길에서 노란색 보이면 branch_bias: In코스=노란쪽으로 / Out코스=반대쪽으로 (방향 자동)
  · [In코스] |offset|≥0.45 가 0.6s 지속 + 노란 비율 ≥0.06 ──▶ ROUNDABOUT
  · red_light 3프레임 연속 ──▶ FINISH
어느 상태든: 아루코 면적비 ≥0.02 ──▶ OBSTACLE_STOP (정지)
OBSTACLE_STOP ──(마커 5프레임 연속 소실)──▶ 직전 상태 복귀 (랩 카운트 유지)
ROUNDABOUT:
  · 조향에 turn_direction × circle_steer_bias(0.15) 더해 링 유지 (조기 이탈 방지)
  · 랩 판정 = 3-표결 (2표 이상): ①junction 재등장 ②조향 적분(yaw proxy)
    ≥ yaw_lap_threshold ③경과시간 ≥ nominal_loop_time_s
  · 단, min_loop_time_s(3s) 전엔 절대 탈출 안 함 / max_loop_time_s(20s)면 강제 탈출
  · 탈출 ──▶ DRIVE (roundabout_done, 재진입 안 함)
FINISH ──(red_light 재확인)──▶ DONE (완전 정지)
```

- `skip_missions:=true` = 순수 차선추종 모드 (신호등/미션 전부 무시, 즉시 주행).
  차선/PID 튜닝은 항상 이 모드로.
- `race_dir` (left=반시계/right=시계) **하나만** 당일 설정하면 회전교차로 방향,
  junction 탐색 방향이 함께 뒤집힌다.

## ③ 구동 — control_node (킷 + 안정화 패치)

- `/control`(steering, throttle) → PCA9685(I2C-3, 0x40) ch0=조향 서보, ch1=ESC.
  1500µs=중립, ESC 아밍 3초.
- 안정화: 초기화 EBUSY 재시도(10×0.5s), 주행 중 I2C 쓰기 실패는 로그만 남기고
  생존, launch에서 respawn. **그래도 I2C wedge는 하드웨어 문제 — 비상정지는
  ESC 전원 스위치.**
- joystick_node는 E-stop 백업으로 launch에 포함.

## 실행

```bash
cd D-Racer-Kit
source /opt/ros/humble/setup.bash && source install/setup.bash

# 실전 (차가 실제로 움직임!)
ros2 launch decision auto_race.launch.py course:=out race_dir:=left

# 차선추종만 (미션 없이 즉시 주행 — 튜닝용)
ros2 launch decision auto_race.launch.py skip_missions:=true

# In 코스
ros2 launch decision auto_race.launch.py course:=in
```

- launch는 `race_config.yaml`을 **로드하지 않는다.** 시작값 = 소스 기본값.
- 주행 중 라이브 튜닝: `ros2 param set /decision_node drive_throttle 0.16`
  (drive/slow/stop_throttle, curve_slow, kp/ki/kd, steer_center/scale,
  max_steering_delta, steer_slow, fork_bias, fork_hold_s 모두 가능)
- 확정값은 소스 기본값에 반영 후 **반드시 리빌드** (symlink-install 아님):
  `colcon build --packages-select decision perception`

## 튜닝 순서 (트랙에서)

1. **BEV src 캘리브레이션** — 모니터 SRC 뷰 노란 사각형을 트랙 사다리꼴에 맞추기.
   이상하면 `use_birdeye false`로 끄고 진행.
2. **차선 인식** — HSV 범위·roi_top_ratio로 빨간 중심선 안정화.
3. **PID** — `skip_missions:=true`로 kp↑(따라가게) → 떨리면 kd↑. 필요 시
   `max_steering_delta 0.1`, `steer_slow 0.4`.
4. **스로틀** — drive_throttle 낮게 시작, 이탈 않는 선까지.
5. **미션 하나씩** — green → red → 표지판(fork_hold_s 실측) → 아루코 →
   (In 코스면) 회전교차로 랩 캘리브레이션.
