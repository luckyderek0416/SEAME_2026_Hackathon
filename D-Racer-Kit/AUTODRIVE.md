# D-Racer 자율주행 스택 — 전체 로직 (SEA:ME 2026)

TOPST D-Racer-Kit 기본 노드(`camera`, `control`, `joystick`, `monitor`, `battery`)
위에 자율주행 노드(`perception`, `inference`, `decision` + msgs)를 얹은 구조.
**2026-07-05 코드 기준.** 미구현/리스크는 루트의 `MISSING_FEATURES.md` 참고.

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
프레임 → ROI 자르기(아래 65%, roi_top_ratio=0.35)
       → [BEV ON] perspective warp (birdeye_src_ratio → dst_ratio)   ← 기본 ON, src 캘리브레이션 완료(2026-07-07)
       → HSV 마스크: 흰색 / 노란색 각각 + 합산 (mask_mode=hsv)
       → morphology 노이즈 제거 (morph_kernel=3)
       → [In코스] 색상 추종 선택: WHITE 모드=흰 마스크만 / YELLOW 모드=노란 마스크만
           (히스테리시스: 노란 ≥ follow_yellow_ratio(0.03) → YELLOW,
            흰픽셀 > 노란픽셀 → WHITE 복귀, ROUNDABOUT 중엔 YELLOW 강제)
       → 멀티밴드 분석 (num_bands=4, 아래→위 = 가까운곳→먼곳):
           [GUIDED ON] 각 밴드는 직전 밴드 중심 ± margin 창에서만 탐색
                       (margin=60px + 밴드당 +10px, 점프 clamp 80px)
           [fork_dir 수신 시] band 0 시드를 해당 브랜치로 ±fork_seed_px(90) 이동 = 브랜치 락온
           밴드별 좌/우 차선 픽셀 평균 → 밴드 중심, lane_width EMA (초기값 192px=실측 350mm)
       → lane_center 계산:
           [LA ON]  0.7×최근접 밴드 + 0.3×최원거리 밴드 (look-ahead 블렌드)
           [LA OFF] 전체 밴드 가중평균
       → offset = (lane_center − 화면중앙) / (화면폭/2)   ∈ [-1, 1]
       → curvature = 밴드 중심의 아래→위 drift (커브 세기)
       → junction = 바깥 차선의 점선/개방 감지 (회전교차로 개구부; 합산 마스크 기준)
       → fork = 분기 구간 감지 (상단 스캔밴드 라인 군집≥3 or 바깥 span 급증; 락온 해제 신호)
       → yellow_ratio / yellow_offset = 노란 픽셀 비율·위치 (In/Out 분기, 회전교차로 진입 게이트)
       → yellow_crossline = 노란 "가로선" 감지 (yellow-only 마스크 하단창 행-폭 스캔;
                            회전교차로 진입/탈출 트리거. 차선 중심엔 영향 없음)
       → offset EMA 스무딩 (smooth_alpha=0.5) → LaneState 발행
```

- **BEV / GUIDED / LA 셋 다 기본 ON** (`use_birdeye`, `use_guided_band`,
  `use_lookahead_control`). 끄면 원래 파이프라인과 동일. 상세: `docs/lane_bev_guided.md`,
  `docs/lookahead_control.md`.
- **BEV src는 2026-07-07 실측 캘리브레이션됨** (`roi_top_ratio=0.35` 기준
  `birdeye_src_ratio = [0.226,0.342, 0.745,0.342, 0.998,0.965, -0.006,0.965]`,
  직선구간 실프레임 차선 피팅, 워프 후 slope ±0.05 / 0.20w·0.80w 평행).
  **카메라 마운트를 다시 만지면 재캘리브레이션 필요.**
- `LaneState` 필드: `lane_found, offset, curvature, num_lanes, junction,
  yellow_ratio, yellow_offset, yellow_crossline, fork`.
- decision → perception 역방향 채널: `/decision/fork_dir`(String) = 갈림길/회전교차로
  출구의 락온 방향, `/decision/state`(String) = 주행 모드(BEV 디버그 MODE 표시·
  ROUNDABOUT 중 YELLOW 강제에 사용).
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
target = offset + turn_bias(갈림길) + branch_bias(In/Out 분기) + curve_bias(급커브 FF)
correction = PID(target)                    # kp=0.6, ki=0, kd=0.15   (lane lost면 target=0)
steer = steer_center(0.2) + correction × steer_scale(−1.0)   # 좌조향은 steer_left_gain(1.2) 증폭
```
- `steer_scale`이 **음수인 게 맞다** (트랙 검증: 이 차는 부호 반대).
- `curve_steer_bias`(기본 0=off): 곡률 비례 feed-forward — 급좌커브에서 가까운 왼선을
  놓쳐도 미리·더 꺾게. `steer_left_gain`(1.2): 좌조향이 실제로 덜 꺾이는 비대칭 보정.
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
  · 갈림길 표지판: on_dets 표결(최근 5개 중 같은 방향 ≥2 → confirmed_fork_direction 확정)
      → turn_latch = 확정방향 → /decision/fork_dir 로 perception에 전달 →
        lane_node가 guided-band 시드를 그 브랜치로 밀어 "브랜치 락온"(표지판 섬 회피).
        fork_bias(0.2)는 진입 보조. 해제 = lane.fork 재수렴(도로 다시 하나) 또는
        fork_hold_s(8s) failsafe. DRIVE에서만 적용(ROUNDABOUT 자동 억제)
  · 갈림길에서 노란색 보이면 branch_bias: In코스=노란쪽으로 / Out코스=반대쪽으로 (방향 자동)
  · [In코스 색상 추종, perception] WHITE(흰선만 추종) ⇄ YELLOW(노란선만 추종) 히스테리시스:
        노란 비율 ≥ follow_yellow_ratio(0.03) → YELLOW / 흰픽셀 > 노란픽셀 → WHITE 복귀.
        ROUNDABOUT 동안은 YELLOW 강제. course:=out 이면 비활성. 디버그에 FOLLOW-W/Y 표시.
  · [In코스] 진입 = 노란 가로선(yellow_crossline) 첫 감지 + 노란 비율 ≥0.06 게이트,
        enter_sustain_s(0.3s) 지속 ──▶ ROUNDABOUT   (곡률/junction 표결은 폐기)
  · red_light 3프레임 연속 ──▶ FINISH
        단, 출발 후 finish_min_drive_s(15s) 전엔 red 무시 + light_min_area 박스 면적
        게이트(0=off) — 멀리 보이는 도착 신호등/빨간 물체 오인식으로 코스 중간에
        멈추는 사고 방지
어느 상태든: 아루코 면적비 ≥0.02 ──▶ OBSTACLE_STOP (정지)
OBSTACLE_STOP ──(마커 5프레임 연속 소실)──▶ 직전 상태 복귀 (랩 카운트 유지)
ROUNDABOUT:
  · 조향에 turn_direction × circle_steer_bias(0.225) 더해 링 유지 (조기 이탈 방지)
  · 주(PRIMARY) 탈출 = 노란 가로선 상승엣지 카운트: 진입 가로선은 쿨다운으로 무시,
    같은 가로선을 roundabout_exit_gates(1)번 더 만나면 → turn_latch=출구side
    (race_dir 파생; 정방향=right) 로 브랜치 락온 탈출
  · failsafe 랩 판정 = 4-표결 (2표 이상): ①junction 랩 카운트 ②조향 적분(yaw proxy)
    ③경과시간 ≥ nominal_loop_time_s ④yellow_crossline 재등장 — 가로선 검출 실패 대비
  · 단, min_loop_time_s(3s) 전엔 절대 탈출 안 함(진입측 가로선을 출구로 오인 방지)
    / max_loop_time_s(20s)면 강제 탈출
  · 탈출 ──▶ DRIVE (roundabout_done, 재진입 안 함; perception은 노란 추종 유지하다
    흰색>노란색이 되면 흰선 추종 복귀)
FINISH ──(red_light 3프레임 재확인)──▶ DONE (완전 정지)
```

- **갈림길 방향은 한 프레임 오검출이 아니라 표결로 확정** (`decision_node.on_dets`의
  `deque(maxlen=fork_sign_vote_window=5)`, `fork_sign_vote_min=2`,
  `fork_sign_min_conf=0.5`). 확정 방향을 매 틱 state_machine에 넘겨 `turn_latch`로 사용.
  표결-소멸은 `fork_vote_clear_s`(1s) — 갈림길 통과까지의 유지는 turn_latch 한 곳만 담당.
- **회전교차로 진입·탈출의 핵심 신호는 yellow_crossline** (가로선). `enter_curvature` 는
  더 이상 사용하지 않음. 가로선 검출 신뢰도(crossline_* 임계) 실측이 최우선.
- `skip_missions:=true` = 순수 차선추종 모드 (신호등/미션 전부 무시, 즉시 주행).
  차선/PID 튜닝은 항상 이 모드로.
- `race_dir` (left=반시계/right=시계) **하나만** 당일 설정하면 회전교차로 방향,
  junction 탐색 방향이 함께 뒤집힌다.
- decision debug 로그(`--log-level decision_node:=debug`): `xline/junc/yr/entryV/exitV/
  forkDet/forkFix/forkConf/latch` 로 인지 신호와 표결 상태를 실시간 확인.

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
- 주행 중 라이브 튜닝:
  - decision_node: `drive/slow/stop_throttle, curve_slow, kp/ki/kd, steer_center/scale,
    steer_left_gain, curve_steer_bias, max_steering_delta, steer_slow, fork_bias,
    fork_hold_s, fork_sign_min_conf, fork_sign_vote_window(int), fork_sign_vote_min(int),
    fork_vote_clear_s, roundabout_exit_gates, enter_sustain_s, circle_steer_bias,
    finish_min_drive_s, light_min_area, conf_threshold`
  - lane_node: `birdeye_src_ratio, use_birdeye, HSV 범위, crossline_*, follow_yellow,
    follow_yellow_ratio, follow_yellow_exit_white_ratio, lane_width_init, fork_seed_px,
    fork_*` 등 즉시 반영 (예: `ros2 param set /lane_node crossline_min_rows 5`)
- 확정값은 소스 기본값에 반영 후 **반드시 리빌드** (symlink-install 아님):
  `colcon build --packages-select decision perception`
  (LaneState.msg 를 바꾸면 `perception_msgs` 부터: `--packages-select perception_msgs perception decision`)

## 튜닝 순서 (트랙에서)

1. ~~BEV src 캘리브레이션~~ **완료(2026-07-07, roi 0.35 기준)** — 카메라 마운트 바꿨을
   때만 재캘리브레이션. ("src 튜닝해줘" 요청 시 직선구간 차 중앙 정렬 후 자동 재계산.)
2. **차선 인식** — HSV 범위로 빨간 중심선 안정화. In 코스는 노란 HSV 도
   (FOLLOW-Y 전환이 `yellow_hsv_lo` 에 달려 있음).
3. **PID** — `skip_missions:=true`로 kp↑(따라가게) → 떨리면 kd↑. 필요 시
   `max_steering_delta 0.1`, `steer_slow 0.4`. 급좌커브 바깥 튕김은 `curve_steer_bias`
   (0.2~0.5) + `adaptive_lookahead true` 실험.
4. **스로틀** — drive_throttle 낮게 시작, 이탈 않는 선까지.
5. **갈림길(Out)** — 표지판 실거리 인식 확인, `fork_seed_px`(90)·분기감지 임계 실측.
6. **미션 하나씩** — green → red(`finish_min_drive_s` 코스 시간에 맞게) → 표지판 →
   아루코 → (In 코스면) 가로선(`crossline_*`) 신뢰도 + 색상 전환 문턱 + 랩 failsafe
   캘리브레이션(`yaw_lap_threshold`, `nominal_loop_time_s`).
