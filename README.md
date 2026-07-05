# SEAME_2026_Hackathon
TEAM : chita

D-Racer 자율주행 스택. 이 문서는 **미션 구간마다 지금 코드가 어떤 로직·알고리즘으로
동작하는지**를 팀이 빠르게 이해하도록 정리한 것이다.

- 워크스페이스는 **`D-Racer-Kit/` 하나뿐**이다. 소스·빌드·실행 모두 여기서 한다.
- 코스 선택: **In**(회전교차로) / **Out**(S자+갈림길) — `course:=in|out`.
- 트랙 방향: 당일 추첨. `race_dir:=left|right` **한 값**으로 회전 방향·junction side·출구가 한꺼번에 뒤집힌다.

---

## 🗺️ 전체 데이터 흐름

<p align="center">
  <img src="D-Racer-Kit/docs/dataflow.png" alt="D-Racer 자율주행 데이터 흐름" width="480">
</p>

```
카메라 ─┬─► lane_node      (차선: offset/곡률/junction/fork/노란색)  ─┐
        ├─► aruco_node     (동적 장애물 마커)                        ─┤
        └─► yolo_ncnn_node (신호등·좌우표지판)                       ─┤
                                                                     ▼
                                                    decision_node + state_machine
                                                 (미션 상태 판단 + PID 조향 + 곡률감속)
                                                                     │  /control
                                                                     ▼
                                                    control_node → PCA9685 → 서보·모터
```

**핵심 원칙: 조향은 항상 차선 PID에서 나온다.** 상태(state)는 *스로틀*과 *어떤 미션 보정을
켤지*만 바꾼다. ([state_machine.py](D-Racer-Kit/src/decision/decision/state_machine.py) 상단 주석 참고)

### 노드 한눈에 보기

| 노드 | 입력 | 출력 | 하는 일 |
|---|---|---|---|
| `camera_node` | USB 카메라(/dev/video1) | `/camera/image/compressed` (30Hz, JPEG) | 프레임 캡처·JPEG 인코딩. 해상도는 `vehicle_config.yaml`(현재 320×160) |
| `lane_node` | 카메라, `/decision/fork_dir` | `/perception/lane` (LaneState) | OpenCV 차선검출 (HSV·조감도·멀티밴드) |
| `aruco_node` | 카메라 | `/perception/aruco` | ArUco 마커 검출(면적비=근접도). CPU 절감 위해 `aruco_hz`(12Hz)로 저속 검출 |
| `yolo_ncnn_node` | 카메라 | `/inference/detections` (10Hz) | NCNN YOLOv8: red/green_light, left/right_sign |
| `decision_node` | lane·aruco·detections | `/control`, `/decision/fork_dir` | 상태머신 30Hz 구동, 표지판 방향 표결 |
| `control_node` | `/control`, joystick | PCA9685(서보·ESC) | 20Hz 액추에이션(`command_hz`), ESC 아밍 3초, E-STOP |
| `battery_node` | INA219 (I2C) | `/battery_status` (10Hz) | 전압→% (선형 6.4~8.4V) |
| `monitor_node` | 위 토픽들 | 웹 대시보드 :5000 | Flask 모니터(카메라·배터리·디버그) |
| `joystick_node` | 게임패드 | `/joystick` | **E-STOP** 안전 |

---

## 🧠 상태 머신 (미션의 뼈대)

`state_machine.py`의 `RaceStateMachine`이 매 틱(30Hz) 다음 상태 중 하나로 동작한다.

```
WAIT_GREEN ──초록불──► DRIVE ──빨간불──► FINISH ──빨간불──► DONE
                        │  ▲
              (In)회전교차로│  │탈출
                        ▼  │
                    ROUNDABOUT
   (어느 상태든) 마커 등장 → OBSTACLE_STOP → 마커 퇴거 → 직전 상태 복귀
```

- 조향 공식: `steer = steer_center + PID(target) * steer_scale`
  - `target = lane.offset + turn_bias(갈림길) + branch_bias(색기반 In/Out)`
  - `steer_scale = -1.0` (트랙 검증 결과 조향 부호 반대라 음수), `steer_center = 0.2`(드리프트 보정 바이어스)
- `skip_missions:=true` → 미션 전부 건너뛰고 **순수 라인추종만** (kp/kd·HSV 튜닝용).

---

## 🚦 미션별 구현 로직

### 0) 출발 · 신호등 — `WAIT_GREEN`
정지 상태로 대기하다가 **초록불을 연속 `green_frames`(기본 3)프레임** 인식하면 `DRIVE`로.
- 판정: `yolo_ncnn_node`의 `green_light` (conf ≥ `conf_threshold` 0.5)
- 출력: 조향=중립, 스로틀=0
- 코드: [state_machine.py:188-192](D-Racer-Kit/src/decision/decision/state_machine.py#L188-L192)

### 1) 기본 주행 · S자 — `DRIVE`
차선 PID로 조향하며 **곡률 기반 감속**으로 달린다.
- 스로틀: `throttle = drive_throttle * (1 - curve_slow * |curvature|)`, 하한 `slow_throttle`
  - 직선은 `drive_throttle`(0.20), 커브가 심할수록 느려짐 → 이탈 위험↓
- `curvature`는 근거리~원거리 밴드 중심의 가로 드리프트로 추정([lane_detector.py:396-403](D-Racer-Kit/src/perception/perception/lane_detector.py#L396-L403))
- 코드: [state_machine.py:239-244](D-Racer-Kit/src/decision/decision/state_machine.py#L239-L244)

**차선검출 파이프라인** ([lane_detector.py](D-Racer-Kit/src/perception/perception/lane_detector.py)):
1. 하단 ROI 크롭(`roi_top_ratio` 0.55) → 조감도(BEV) 워프(기본 ON, **카메라 높이 바뀌면 재캘리브 필수**)
2. HSV 마스크로 **흰색 + 노란색** 라인 추출 → morph_open으로 대리석 바닥 반사·점선 잡음 제거
3. ROI를 가로 밴드 `num_bands`(4)개로 나눠 근거리(조향)~원거리(커브 예측) 중심 검출
4. **guided band**: 각 밴드는 이전 밴드 중심 ±margin만 탐색 → 먼 분기/출구 라인이 중심을 못 끌어감
5. **look-ahead 블렌드**: 중심 = `near_weight`×근거리 + `lookahead_weight`×원거리 → 커브 선제 조향
6. offset EMA 평활(`smooth_alpha`) → 프레임 간 조향 떨림 감소

> **확정 구성(committed)**: BEV · guided band · look-ahead · HSV(흰+노란) = **기본 ON**,
> `adaptive_lookahead` = OFF. 이 값들이 `lane_node.py`의 `declare_parameter` 기본값이다.
> 각 모드 토글은 **삭제하지 않고 트랙 디버그 스위치로 남겨둔다** — 문제 발생 시
> `ros2 param set /lane_node use_birdeye false`(BEV 격리), `use_guided_band false`(guided 격리)
> 처럼 한 변수씩 꺼서 원인을 좁힌다. 코드 정리(dead-path 삭제)는 레이스(7/16) 후로 미룬다.

### 2) 좌/우 갈림길 (Out 코스) — `DRIVE` 내 `fork` 처리
전제: **표지판이 두 분기 사이(median)에 있다** → "표지판 확정 = 분기 도착".

1. **방향 표결**(decision_node 로컬): `left_sign`/`right_sign`을 최근 `fork_sign_vote_window`(5)개
   메시지에서 누적, `fork_sign_vote_min`(2) 이상인 방향만 확정 → 한 프레임 오검출로 확정 안 함
   ([decision_node.py:240-276](D-Racer-Kit/src/decision/decision/decision_node.py#L240-L276))
2. 확정 방향 → `turn_latch` → `/decision/fork_dir`로 perception에 전달
3. `lane_node`가 그 방향으로 **guided-band 시드를 밀어**(`fork_seed_px`) **선택한 브랜치의 실제
   차선만 추종** → median의 표지판 섬을 배제 ([lane_detector.py:304-334](D-Racer-Kit/src/perception/perception/lane_detector.py#L304-L334))
4. 추가로 `turn_bias`(±`fork_bias`)가 조향을 살짝 밀어 진입 보조
5. **락 해제 = 도로 재수렴**: perception `fork` 플래그가 켜졌다 다시 꺼지면(두 브랜치가 재합류)
   해제. `fork_hold_s`(8s)는 안전 상한(failsafe)일 뿐 ([state_machine.py:172-185](D-Racer-Kit/src/decision/decision/state_machine.py#L172-L185))
- (In/Out 색 기반 보정) `branch_bias`: In=노란 브랜치 쪽, Out=흰 브랜치 쪽으로 편향([state_machine.py:108-120](D-Racer-Kit/src/decision/decision/state_machine.py#L108-L120))

### 3) 동적 장애물 — `OBSTACLE_STOP` (전역 우선)
어느 상태든 ArUco 마커가 충분히 크게(`area_ratio ≥ marker_area_trigger` 0.02) 보이면 **즉시 정지**.
- 마커가 `marker_clear_frames`(5)프레임 동안 안 보이면 **직전 상태로 복귀**(회전교차로 랩카운트 유지)
- 마커 사전/반전(inverted) 지원: `DICT_4X4_50`, `inverted:=true`
- 코드: [state_machine.py:194-204](D-Racer-Kit/src/decision/decision/state_machine.py#L194-L204)

### 4) 회전교차로 (In 코스) — `ROUNDABOUT`
IMU·마커 없이 **비전만으로** 진입·랩카운트·탈출한다.

**진입 판정** (오진입 방지): 아래 2-of-3 표결이 `enter_sustain_s`(0.6s) 지속되고,
`yellow_ratio ≥ yellow_enter_ratio`(노란 링) 게이트를 통과해야 진입 ([state_machine.py:216-233](D-Racer-Kit/src/decision/decision/state_machine.py#L216-L233))
- ① 노란 가로선(`yellow_crossline`) ② junction(점선 개구부) ③ 급커브(`|curvature| ≥ enter_curvature`)
- 흰색 외곽 코너가 실수로 진입시키지 못하도록 노란색 게이트가 막는다

**링 주행**: 차선 PID + `turn_direction × circle_steer_bias`로 안쪽으로 살짝 붙여 **조기 탈출 방지**.

**탈출 (주 경로 = 게이트 락온)**: 링을 돌며 게이트(노란 가로선 or junction 개구부)의 **상승엣지**를
세고, `min_loop_time_s` 이후 **`roundabout_exit_gates`(기본 2)회** 도달하면 출구 브랜치(`roundabout_exit_side`)로
`turn_latch` 락온 → 갈림길과 같은 방식으로 **명시적으로 빠져나간다** ([state_machine.py:270-300](D-Racer-Kit/src/decision/decision/state_machine.py#L270-L300)).

**탈출 failsafe (2-of-4 표결)**: 게이트 감지를 놓쳐도 나가도록 백업.
- ① junction 랩카운트 ② 조향각 적분(yaw proxy, IMU 대체) ③ 경과시간 ④ 노란 가로선 재출현
- `min_loop_time_s`(3s) 전엔 절대 탈출 안 함(1바퀴 미만=실패). `max_loop_time_s`(20s)는 강제탈출.
- ⚠️ `yaw_lap_threshold`·`nominal_loop_time_s`·`roundabout_exit_gates`는 **트랙 실측 캘리브 필요**.

### 5) 도착 · 신호등 — `FINISH` → `DONE`
`DRIVE` 중 **빨간불을 연속 `red_frames`(3)프레임** 인식하면 정지(`FINISH`),
다시 빨간불이 보이면 완전 종료(`DONE`). ([state_machine.py:322-329](D-Racer-Kit/src/decision/decision/state_machine.py#L322-L329))

---

## ▶️ 실행

```bash
cd D-Racer-Kit
source /opt/ros/humble/setup.bash
source install/setup.bash          # ← 반드시 D-Racer-Kit/install
colcon build --packages-select <수정한_pkg>   # src 수정 시 반드시 리빌드 (symlink 아님)

# 전체 자율주행 (차 실제 구동! 안전 확인 후)
ros2 launch decision auto_race.launch.py course:=in race_dir:=left
# course:=out 이면 lane_node 의 use_yellow 가 자동 off (흰색만 검출 → 노란기 오염 차단)

# 순수 라인추종만 테스트
ros2 launch decision auto_race.launch.py skip_missions:=true
```

> ⚠️ `auto_race.launch.py`는 실제로 차를 구동한다. 진단은 개별 노드(`ros2 run camera camera_node`)로.

### 주행 중 라이브 튜닝 (`ros2 param set`, 리빌드 불필요)
```bash
ros2 param set /decision_node drive_throttle 0.16     # 속도
ros2 param set /decision_node curve_slow 0.6          # 커브 감속 세기
ros2 param set /decision_node kp 0.7                  # 라인추종 반응
ros2 param set /decision_node roundabout_exit_gates 2 # 회전교차로 탈출 게이트 수
ros2 param set /lane_node yellow_hsv_lo "[18,45,110]" # 노란색 검출(대시보드 보며)
ros2 param set /lane_node use_birdeye false           # BEV 끄고 검증
```
> `auto_race.launch.py`는 `race_config.yaml`을 **로드하지 않는다.** 시작값은
> `decision_node.py`의 `declare_parameter` 기본값이다. 마음에 드는 값은 여기에 확정 후 리빌드.

---

## 🛠️ 트랙 운영 노트 (2026-07-05 주행에서 배운 것)

| 증상 | 원인 | 대응 |
|---|---|---|
| SSH 끊김 · 대시보드 렉 · 카메라 로그 멈춤 | **핫스팟 대역폭 포화 + 보드 CPU 포화** (노드 통신 문제 아님, DDS는 로컬) | 집 **공유기를 로컬 LAN(AP)로** 사용 / 실주행 중 웹탭 닫기 / `debug_log:=false` / `OPENCV_DEBUG_MODE` 주행 땐 off |
| 출발 순간 배터리 % 급락 | LiPo **전압 sag** (%는 순간전압 선형매핑, 평활 없음) — **정상** | 무부하 전압으로 판단. 부하 시 6.6V↓면 충전 |
| 카메라 높이 바꾼 뒤 차선 못 따라감 | **BEV 캘리브(`birdeye_src_ratio`)가 깨짐** (높이/각도 종속) | `use_birdeye:=false`로 원인 격리 → 재캘리브 / `drive_throttle`↓ |

- **캘리브레이션 종속성**: `birdeye_src_ratio`는 특정 카메라 높이/각도로 실측한 값이라
  카메라를 옮기면 반드시 재캘리브해야 한다 ([lane_node.py:54-57](D-Racer-Kit/src/perception/perception/lane_node.py#L54-L57)).
- 미구현/튜닝 남은 항목은 [`MISSING_FEATURES.md`](MISSING_FEATURES.md), 문제해결은
  [`D-Racer-Kit/docs/TROUBLESHOOTING.md`](D-Racer-Kit/docs/TROUBLESHOOTING.md) 참고.
