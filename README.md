# SEAME_2026_Hackathon
TEAM : chita

D-Racer 자율주행 스택. 이 문서는 **미션 구간마다 코드가 어떤 로직·알고리즘으로
동작하는지**를 팀이 빠르게 이해하도록 정리한 것이다.

- 워크스페이스는 **`D-Racer-Kit/` 하나뿐**이다. 소스·빌드·실행 모두 여기서 한다.
- 코스 선택: **In**(회전교차로) / **Out**(S자+갈림길) — `course:=in|out` 별도 런.
- ★ **In 코스 풀 미션 무개입 완주 검증** (출발→노랑 래치→링 1랩→탈출→병합→마커
  정지/재출발→빨간불 정지, 69.5s). Out 코스는 구현 완료 / 실주행 미검증.

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
켤지*만 바꾼다. (예외: RA 탈출 직후의 짧은 고정 시퀀스 — 아래 참조)

| 노드 | 하는 일 |
|---|---|
| `camera_node` | USB 캡처(`vehicle_config.yaml`, 320×160 JPEG 30Hz) |
| `lane_node` | OpenCV 차선검출: HSV → BEV → 멀티밴드 + guided band + **SW 코리도** |
| `aruco_node` | 마커 검출(면적비=근접도). **사전은 실물과 일치 필수** (6×6) |
| `yolo_ncnn_node` | red/green_light, left/right_sign (NCNN) |
| `decision_node` | 상태머신 구동, 표지판 표결, 라이브 파라미터 |
| `control_node` | PCA9685 액추에이션(20Hz), ESC 아밍, E-STOP, 명령 스테일 워치독(0.5s). **1틱=스로틀 0.0098** |
| `battery_node` / `monitor_node` / `joystick_node` | 전압 / 웹 대시보드 :5000 / E-STOP |

---

## 🚦 In 코스 미션 로직

경로: 체커 출발 → 하단 분기(노랑 래치) → B 개구부 통과 → 링 등판 → **A 정지선**
→ 반시계 1랩(B는 통과) → **A 재도달에서 발화 → 우측 탈출** → 흰 병합 → 왼쪽
직선(ArUco) → 체커 빨간불 정지.

### 회전교차로 게이트 — 가로선 군집 카운트 (위치 불변량)
yaw/시간 임계(속도 의존) 대신 **"몇 번째 가로선 군집인가"** 로 판정:
- 가로선 목격을 간격(`gate_cluster_gap_s`)으로 군집화, 누적 ON ≥ `gate_cluster_on_s` 면 카운트
- **블랭크(`gate_blank_s`) 중 태어난 군집은 통째 무효** (진입 잔상 — 길이 무관, 출생시점 불변량)
- STOPLINE 분류기(`stopline_mode`): 코리도를 **관통(cov≥0.25)하면서 정면(직교각≤15°)**인
  프레임만 정지선 인정 — B 개구부 스침은 원리적으로 군집 형성 불가
- 직발화(`ra_direct_fire`) + 링 체류 하한(`ra_fire_min_frac`): 카운트 성립 프레임에 즉시 발화
- 유일 failsafe = `ra_failsafe_exit_s`(링 체류 절대 상한). min_loop 전에는 절대 탈출하지
  않는다 — **1바퀴 미달 = 실격**

### RA 탈출 → 개구부 통과 (발화 후 시퀀스)
발화 순간부터 시간 순서로:

1. **침묵 0.3s** (`sw_mouth_delay_f` 6f): 지각은 탈출 락을 보고하지 않고, 조향은
   **직진 0.1s**(`exit_straight_s`) → **탈출측 호 0.3**(`exit_steer_bias`)으로 회전을 개시.
   발화 직후 시야에 남은 반대편(우측) 선을 물지 않기 위한 대기다.
2. **mouth 창 5s** (`sw_exit_mouth_frames` 100f): 개구부 전용 추종 규칙 —
   - 보이는 모든 선(점선·실선)은 **탈출 차선의 좌측 경계(+1)** 로 취급. 스템의 점선과
     쐐기 실선은 꼭짓점에서 이어지는 하나의 경계이기 때문.
   - 선택은 **직전 락 최근접**(연속성) — 실선↔점선 인계에서 궤적이 끊기지 않는다.
   - 점프 가드는 방향별: **좌향 점프는 관대**(자연 인계 방향, 2.5×반차폭),
     **우향 점프는 엄격**(0.5×반차폭, 반대편 선 스냅 차단). 계속 실패하면 소실로 보고.
   - 발행 중심에 **하한 150px**: 지나쳐 왼쪽 뒤에 남은 선을 물어도 좌향 명령이 나가지 않는다.
3. **소실 폴백**: 발화 후 `exit_lock_release_s`(3.5s) 안의 무검출 프레임은 위 1의
   직진/호 규칙, 이후는 마지막 조향 유지(병합 완료까지). 스로틀은 `exit_cap_throttle`(0.185) 캡.
4. **노랑 강제 5s** (`exit_yw_min_f` 100f): 이 시간 전에는 어떤 경로로도 흰 전환 금지 —
   탈출로 중간에 전방 흰 구간이 보여 조기 전환하면 노란 경계를 잃는다.
5. **흰 전환**: 노란 픽셀 완전 소실(`exit_y_zero_ratio`) + 흰 실제 가시(`exit_w_min_px`)
   연속 6f → 흰 전용 모드로 원웨이 전환 (Y 재래치 금지).
6. **병합 정렬 창 3s** (`w_align_frames`): 전환 직후 보이는 **흰 단선 = 내 차선의 우측
   경계**로 강제 분류 (중심 = 실선 − 반차폭 = 실선의 왼쪽) — 병합 각도에서 첫 흰 락을
   좌측 경계로 오인해 실선을 넘는 이탈을 차단. 양선 복원되면 일반 흰 추종.

### 동적 장애물(ArUco) / 도착
- `DRIVE` + 마커 + (빨간존 내 **or** 면적 ≥ `marker_area_trigger`) → `OBSTACLE_STOP`,
  마커 소실 `marker_clear_frames` 연속 → 재출발. **사전 불일치 = 원리적 무검출** —
  새 마커 인쇄 시 반드시 사전/ID 확인
- 빨간불 `red_frames`(**20**, ≈0.7s 연속) + bbox 면적 ≥ `light_min_area`(0.002) → `FINISH`
  → `DONE`. 짧은 문턱은 **빈 신호등 하우징 오인식으로 조기 정지**한다(실측).
  무장 = obstacle_done or `finish_min_drive_s`(60s) 경과

---

## 🚦 Out 코스 미션 로직 (구현 완료 / 실주행 미검증)

경로: 체커 출발 → 오른쪽 S커브 북상 → 갈림길(표지판) → 상단 서진 → 왼쪽 직선(ArUco)
→ 체커 빨간불 정지. 도로는 **양쪽 실선**(중앙선 없음).

- **상시 SW 코리도** (`sw_out_always`): DRIVE 전 구간, 흰 마스크 입력, dir=0.
  S자에서 좌/우 경계 교대 소실을 관성 추적
- **갈림길 = 시야 마스킹** (`fork_blind_frac`): YOLO 표결로 방향 확정 → **반대쪽 컬럼을
  track+코리도 입력에서 제거**. 조향 편향(소실 시 증발) 대신 지각 차원에서 가지를 선택
- ArUco/빨간불은 In 과 공용 (out 은 빨간존 무장에 RA 게이트 없음 = 즉시)

---

## ▶️ 실행 · 운영

```bash
cd D-Racer-Kit
export ROS_DOMAIN_ID=14            # 원격 ros2 CLI/모니터링에 필수
export PYTHONUNBUFFERED=1          # 강제 종료 시 로그 꼬리 유실 방지
source /opt/ros/humble/setup.bash
source install/setup.bash          # ← 반드시 D-Racer-Kit/install
colcon build --packages-select <수정한_pkg>   # symlink 아님 — src 수정 시 반드시 리빌드

ros2 launch decision auto_race.launch.py course:=in  race_dir:=left   # 차 실제 구동!
ros2 launch decision auto_race.launch.py course:=out race_dir:=left
ros2 launch decision auto_race.launch.py skip_missions:=true          # 순수 라인추종
```

- 시작값은 `decision_node.py`/`lane_node.py` 의 `declare_parameter` 기본값
  (`race_config.yaml` 은 **로드되지 않음**). 라이브 변경: `ros2 param set /decision_node exit_steer_bias 0.25`
- 스로틀 기본: 흰 0.20 / slow 0.17 / 노랑 0.19 / 킥 0.23 / 탈출~병합 캡 0.185
  (0.17은 운동유지 임계 0.175 아래 — 탈출 캡을 내리면 병합 중 물리 스톨)
- **정지 프로토콜**: 노드 kill → PCA9685 중립 확인 → 프로세스 0 확인.
  ssh 불가 시 **ESC 물리 컷이 유일한 정지 수단**
- 지각/게이트 로직 변경 시 **오프라인 검증 필수**: 손굴림 캡처 리플레이(A/B)와
  상태머신 시뮬로 회귀 확인 후 배포. 보드 배포는 scp + 리빌드 + **src=install md5 검증**
  (전원 차단으로 install 사본이 0바이트가 된 사례 있음 — 빌드 후 `sync` 권장)
- 롤백 지점: `85c9eae`(풀코스 완주 스택) / HEAD(주석·데드코드 정리판, 동작 동일 검증됨)

## ⚠️ 하드웨어 주의

- **배터리 팩 노화**: "완충"이 무부하 7.9V(정상 8.4V)까지 내려옴. 부하 새그 ~1V.
  무부하 7.7V 미만 주행은 등판 정지·보드 브라운아웃 위험. 랩타임 변동이 시간 기반
  가정을 왜곡 — **교체 팩이 정답**
- 보드 Wi-Fi 동글 불안정(주행 중 사망 이력 다수). IP 유동(핫스팟 서브넷 스캔).
  **아이폰 핫스팟은 "호환성 최대화" 필수**(2.4GHz). 노드 간 통신은 로컬이라
  **주행 자체는 Wi-Fi 없이도 정상** — 단, ssh 세션에서 런치했다면 동글 분리 = 노드 사망
  (보드 콘솔 런치 또는 nohup/tmux 사용)
- 카메라 마운트 변경 시 `birdeye_src_ratio` **재캘리브 필수** (BEV 30×30mm 점선 실측 기반)
- 문제해결: [`D-Racer-Kit/docs/TROUBLESHOOTING.md`](D-Racer-Kit/docs/TROUBLESHOOTING.md)
