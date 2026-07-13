# SEAME_2026_Hackathon
TEAM : chita

D-Racer 자율주행 스택. 이 문서는 **미션 구간마다 지금 코드가 어떤 로직·알고리즘으로
동작하는지**를 팀이 빠르게 이해하도록 정리한 것이다. (2026-07-13 기준,
런별 상세 이력은 [`RUN_LOG_2026-07-12.md`](RUN_LOG_2026-07-12.md))

- 워크스페이스는 **`D-Racer-Kit/` 하나뿐**이다. 소스·빌드·실행 모두 여기서 한다.
- 코스 선택: **In**(회전교차로) / **Out**(S자+갈림길) — `course:=in|out` 별도 런.
- ★ **In 코스는 run80 에서 풀 미션 무개입 완주** (출발→링 게이트→탈출→병합→마커
  정지/재출발→빨간불 정지). Out 코스는 구현 완료 / 실주행 미검증.

---

## 🗺️ 전체 데이터 흐름

<p align="center">
  <img src="D-Racer-Kit/docs/dataflow.png" alt="D-Racer 자율주행 데이터 흐름" width="480">
</p>

```
카메라 ─┬─► lane_node      (차선: offset/곡률/junction/fork/노랑/가로선/빨간노면) ─┐
        ├─► aruco_node     (동적 장애물 마커, DICT_6X6_50)                        ─┤
        └─► yolo_ncnn_node (신호등·좌우표지판)                                    ─┤
                                                                                  ▼
                                                       decision_node + state_machine
                                                    (미션 상태 판단 + PID 조향 + 곡률감속)
                                                                     │  /control
                                                                     ▼
                                                    control_node → PCA9685 → 서보·모터
```

**핵심 원칙: 조향은 항상 차선 PID에서 나온다.** 상태(state)는 *스로틀*과 *어떤 미션 보정을
켤지*만 바꾼다.

| 노드 | 하는 일 |
|---|---|
| `camera_node` | USB 캡처(`vehicle_config.yaml`, 320×160 JPEG 30Hz) |
| `lane_node` | OpenCV 차선검출: HSV → BEV → 멀티밴드 + guided band + **SW 코리도** |
| `aruco_node` | 마커 검출(면적비=근접도, `aruco_hz` 12Hz). **사전은 실물과 일치 필수** (현재 6×6 id3) |
| `yolo_ncnn_node` | red/green_light, left/right_sign (NCNN) |
| `decision_node` | 상태머신 구동, 표지판 표결, 라이브 파라미터 |
| `control_node` | PCA9685 액추에이션(20Hz), ESC 아밍, E-STOP. **1틱=스로틀 0.0098** (0.16≡0.165) |
| `battery_node` / `monitor_node` / `joystick_node` | 전압 / 웹 대시보드 :5000 / E-STOP |

---

## 🚦 In 코스 미션 로직 (검증: run73·76~80)

경로: 체커 출발 → 하단 분기(노랑 래치) → B 개구부 통과 → 링 우측면 등판 → **A 정지선**
→ 반시계 1랩(B는 통과) → **A 재도달에서 발화 → 우측 탈출** → 흰 병합 → 왼쪽
직선(ArUco) → 체커 빨간불 정지.

### 회전교차로 게이트 — 군집 카운트 (위치 불변량)
yaw/시간 임계(속도 의존, run66 까지의 방식)를 폐기하고 **"몇 번째 가로선 군집인가"** 로 판정:
- 가로선 목격을 간격 1.8s 로 군집화, 누적 ON ≥ `gate_cluster_on_s` 면 카운트
- **블랭크(`gate_blank_s`) 중 태어난 군집은 통째 무효** (진입 잔상 — 길이 무관, 출생시점 불변량)
- `roundabout_exit_gates`(2) 도달 = 탈출. **표결 백업**: count≥1 상태에서 새 군집 첫
  프레임 + 시간표 → 즉시 앵커 발화 (`gate=1 exitV=2` 가 정상 주경로 로그)
- 같은군집 재발화 가드(run74), yaw 백업은 7.0(사실상 봉인, run86), `max_loop_time_s` 75 (자가복구 여유)

### 정지선 오검출 방어 3층 (07-13 재설계)
1. **곡률 부호 가드** `sw_curv_max_a`: 링 순환 중 우곡률 코리도 피팅 기각 — B 개구부
   가지 오물림 방지 (run87 실측: 링 −0.013~+0.0015 vs B 가지 +0.006)
2. **SW 교차 게이트** `crossline_sw_gate`: 코리도를 안 걸치는 선 기각
3. **STOPLINE 분류기** `stopline_mode`: 코리도 좌우 경계 사이를 **관통(cov≥0.25)하면서
   정면(직교각≤15°)**인 프레임만 정지선 인정 — B 스침은 원리적으로 군집 형성 불가.
   페어/단선 코리도 모두 지원. 시퀀스가 [A 1차=#1 → A 재도달=#2 발화]로 고정

### 탈출 → 흰 병합 — 병합 브리지 (07-13 재설계)
노랑 소멸~흰 획득 사이 무차선 구간(1~4s)을 관성이 아니라 **기하 사전값**으로 통과:
- 결정층: `merge_bridge_s`(6s) 창 안 무검출 프레임 = 고정 완만 좌호(`merge_blind_bias`)
- 지각층: w_align 창에서 **점선(분기 표식) 추종 금지**(`w_align_dash_fallback 0`) —
  실선 부족이면 무검출로 두고 브리지에 위임. 노랑만 소실돼도 해제(blind release)

### 동적 장애물(ArUco) / 도착
- `DRIVE` + 마커 + (빨간존 내 **or** 면적 ≥ `marker_area_trigger` 0.02) → `OBSTACLE_STOP`,
  마커 소실 `marker_clear_frames`(8, ≈0.8s) 연속 → 재출발. **사전 불일치 = 원리적 무검출**
  (run76~78 교훈) — 새 마커 인쇄 시 반드시 사전/ID 확인
- 빨간불 `red_frames`(3) → `FINISH` → `DONE`. 무장 = obstacle_done or 60s 경과

---

## 🚦 Out 코스 미션 로직 (구현 완료 / 실주행 미검증)

경로: 체커 출발 → 오른쪽 S커브 북상 → 갈림길(표지판) → 상단 서진 → 왼쪽 직선(ArUco)
→ 체커 빨간불 정지. 도로는 **양쪽 실선**(중앙선 없음).

- **상시 SW 코리도** (`sw_out_always`): DRIVE 전 구간, 흰 마스크 입력, dir=0
  (곡률가드·탈출규칙·STOPLINE 자동 비활성). S자에서 좌/우 경계 교대 소실을 관성 추적
- **갈림길 = 시야 마스킹** (`fork_blind_frac` 0.40): YOLO 표결(최근 5건 중 2표, conf 0.5)로
  방향 확정 → **반대쪽 컬럼 40%를 track+코리도 입력에서 제거** (nl==2 복원 시 중지,
  60프레임 상한). 조향 편향(소실 시 증발) 대신 지각 차원에서 가지를 선택
- ArUco/빨간불은 In 과 공용 (out 은 빨간존 무장에 RA 게이트 없음 = 즉시)

---

## ▶️ 실행 · 운영

```bash
cd D-Racer-Kit
source /opt/ros/humble/setup.bash
source install/setup.bash          # ← 반드시 D-Racer-Kit/install
colcon build --packages-select <수정한_pkg>   # symlink 아님 — src 수정 시 반드시 리빌드

ros2 launch decision auto_race.launch.py course:=in  race_dir:=left   # 차 실제 구동!
ros2 launch decision auto_race.launch.py course:=out race_dir:=left
ros2 launch decision auto_race.launch.py skip_missions:=true          # 순수 라인추종
```

- 시작값은 `decision_node.py`/`lane_node.py` 의 `declare_parameter` 기본값
  (`race_config.yaml` 은 **로드되지 않음**). 라이브 변경: `ros2 param set /decision_node drive_throttle 0.16`
- 스로틀 현재 기본: 흰 0.19 / 노랑·링 0.165(틱 324; 라이브로 0.17 운용 중) / 킥 0.23 / 동적보정 off
- **발화 로그 읽는 법**: `gate=1 exitV=2` = 표결 앵커 발화(정상 주경로), `exitV=0` = 카운트
  경로, `circle_t=max_loop` = 강제 타임아웃(비상)
- **정지 프로토콜**: stop.sh(또는 노드 kill) → PCA9685 0x40/0x42 중립 확인 → 프로세스 0
  확인. ssh 불가 시 **ESC 물리 컷이 유일한 정지 수단**
- 게이트/지각 로직 변경 시 **오프라인 검증 필수**: scratchpad 의 리플레이(sw_replay,
  validate_*)와 텔레메트리 시뮬로 과거 런 회귀 확인 후 배포
- 롤백 지점: `e8b0df2`(run60) / `4456538`(run80 풀체인) / HEAD

## ⚠️ 하드웨어 주의

- **배터리 팩 노화**: "완충"이 무부하 7.9V(정상 8.4V)까지 내려옴. 부하 새그 ~1V.
  무부하 7.7V 미만 주행은 등판 정지·보드 브라운아웃(이력 2회) 위험. 랩타임이
  15→30s 로 늘며 시간 기반 가정을 전부 왜곡 — **교체 팩이 정답**
- 보드 Wi-Fi 동글 불안정(주행 중 사망 이력 다수). IP 유동(핫스팟 서브넷 스캔).
  재부팅 시 /tmp 스크립트 재배포 필요. **아이폰 핫스팟은 "호환성 최대화" 필수**(2.4GHz)
- 카메라 마운트 변경 시 `birdeye_src_ratio` **재캘리브 필수**
- 문제해결: [`D-Racer-Kit/docs/TROUBLESHOOTING.md`](D-Racer-Kit/docs/TROUBLESHOOTING.md),
  남은 작업: [`MISSING_FEATURES.md`](MISSING_FEATURES.md)
