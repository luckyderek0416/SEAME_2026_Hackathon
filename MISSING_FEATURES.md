# 레이스 준비 현황 — 보완·튜닝·주행중 파라미터 총정리

기준: **OT 자료 10~19p** vs 현재 코드 (**2026-07-06**, 커밋 `6650084`).
레이스: **7/15 (1차), 7/16 (2차)**. 순위는 1·2차 중 최고 기록 반영. **★ = 레이스 전 필수.**

- 코스: Out(Base) = 출발→S자→**좌/우 갈림길**→동적 장애물→도착
       / In(Option) = 출발→**회전 교차로**→동적 장애물→도착. `course:=in|out` 선택.
  - `course:=out`이면 `use_yellow` **자동 off** (흰색만 검출; 2026-07-06 launch 반영).
- 트랙 방향: 당일 정/역방향 추첨 → `race_dir:=left|right` 하나로 flip.
- 미션 실패 = 수동 복귀(팀장), 포기 = +2분 (회전교차로는 포기 불가). 자가 복귀는 설계상 없음.

---

## 미션별 구현 현황

| 미션 | 규정 요약 | 코드 | 상태 |
|---|---|---|---|
| 출발·신호등 | 초록불 후 출발. 미인식=미션실패 | `WAIT_GREEN` (green 3프레임) | ✅ 구현 / YOLO 실거리 검증(B-5) ★ |
| S자 | 미션 없음, 이탈 +30s | `DRIVE` + `curve_slow` 곡률 감속 | ✅ 구현 / kp·throttle 튜닝(B-3) |
| 좌/우 갈림길 | 표지판 방향대로. 어기면 미션실패 | 표결→`turn_latch`→브랜치 시드→재수렴 해제 | ✅ 재설계 완료 / 분기감지·시드 튜닝(B-9) ★ |
| 동적 장애물 | 등장 시 정지·퇴거 시 출발 | `OBSTACLE_STOP` (면적비 트리거, 12Hz 검출) | ✅ 구현 / 사전·임계 튜닝(B-6) |
| 도착·신호등 | 빨간불 정지. 미인식 +30s | `DRIVE→FINISH→DONE` (red 3프레임) | ✅ 구현 / 실거리 검증(B-5) |
| 회전 교차로 | 1회전 후 탈출. **포기 불가** | 진입 2-of-3 표결 / 탈출=게이트 2회→브랜치 락온(+2-of-4 failsafe) | ⚠️ 로직 완료 / 캘리브레이션 필요(B-2) ★ |

6개 미션 모두 상태머신에 반영, 구조 건전. 남은 건 **보완 2건(A)** + **트랙 실측·튜닝(B)**.

---

## A. 보완이 필요한 것 (코드/캘리브 수정)

### A-1. ★★ BEV 재캘리브레이션 — **07-05 카메라 높이 변경으로 기존 캘리브 무효**

07-05 트랙에서 카메라 높이를 수정한 뒤 차선을 못 따라간 주원인.
`birdeye_src_ratio = [0.262,0.05, 0.811,0.05, 1.017,0.95, 0.034,0.95]` 는 **이전
높이/각도 기준 실측값**이라 현재 마운트에선 워프가 틀어져 offset 자체가 왜곡된다.
**fork/회전교차로 감지도 전부 BEV 위에서 돌므로, 이거 안 하면 B의 모든 튜닝이 헛수고.**

절차:
1. `ros2 param set /lane_node use_birdeye false` 로 끄고 기본 추종 되는지 먼저 확인(원인 격리)
2. 직선 구간에서 모니터 debug 상단(SRC 노란 사각형) 보며 `birdeye_src_ratio` 라이브 조정
   — 워프 후 좌우 차선이 **수직 평행**이 되면 OK
3. 확정값을 `lane_node.py` 기본값에 반영 + 리빌드. **카메라 마운트 다시 만지면 재실행.**

### A-2. ★ In 코스 전환구간 조향 애매 (흰+노란 공존) — 미구현

트랙 도면 확인 결과 **링은 안팎 모두 노란선**, 흰선은 바깥 메인트랙. 문제는 둘이
만나는 **전환 구간**: 흰+노란 합산 마스크로 중심을 잡아 갈라지는 두 선을 섞고,
`_branch_bias` 는 노란 픽셀 전체 centroid 하나를 쓰므로 노란색이 양옆에 걸치면
편향 부호가 프레임마다 뒤집혀 **애매한 방향으로 감** (07-05 실주행에서 관측).

대응(택1, 미구현):
- **방안 A(권장): 노란색 커밋 모드** — In 코스에서 `yellow_ratio` 가 임계 이상이면
  그 구간은 **노란 마스크만으로 차선 중심** 계산 → 링에 확실히 커밋
- 방안 B: 진입 감지 시 fork 처럼 노란 브랜치 쪽으로 guided-band 시드를 밀어 락온

Out 코스만 뛰면 무시 가능. **In 코스 갈 거면 레이스 전 구현 필요.**

### A-3. 문서 정합성 (낮음)

`D-Racer-Kit/AUTODRIVE.md` 갈림길 서술이 옛 "6초 홀드" 방식 → 재설계(브랜치 락온)에
맞춰 갱신 대상. 주행에는 영향 없음.

---

## B. 트랙에서 실측·튜닝해야 하는 것 (코드 결함 아님)

> ⚠️ 공통 전제: **A-1(BEV 재캘리브) 먼저.** 순서: BEV → skip_missions 기본 추종(B-3)
> → YOLO 검증(B-5) → 미션별 튜닝(B-2/B-6/B-7/B-9).

### B-2. ★ 회전교차로 캘리브레이션 (In 코스 시 필수)

진입 = crossline/junction/급커브 **2-of-3 표결** + 노란비율 게이트 + 0.6s 지속.
탈출 = 게이트 상승엣지 `roundabout_exit_gates`(2)회 → 출구 브랜치 락온 (+ 2-of-4 failsafe).

보정 안 된 값 (debug 로그 `entryV/exitV/gate/curv` 로 실측):
- `roundabout_exit_gates: 2` — 한 바퀴에 게이트 몇 번 지나는지 확인 (라이브 튜닝 가능)
- `roundabout_exit_side` — 정방향 출구가 오른쪽 브랜치인지 1회 확인
- `enter_curvature: 0.45` — curvature 스케일(≈0.1~0.3)에 비해 과대 → **0.15~0.25 재튜닝**
  (안 하면 curvature 표가 죽어 2표에만 의존)
- `yaw_lap_threshold: 6.0` / `nominal_loop_time_s: 8.0` — 한 바퀴 실측 (failsafe 표용)

### B-3. S자·기본 주행 튜닝

`skip_missions:=true` 로 kp/kd, `steer_center/scale`, `drive_throttle`, `curve_slow` 튜닝.
튐/이탈 시 `max_steering_delta`(틱당 조향 제한), `steer_slow`(조향 비례 감속) 활용.
07-05 이탈은 속도보다 **BEV 문제(A-1)가 먼저** — 캘리브 후 재평가.

### B-5. ★ YOLO 신호등/표지판 실거리 인식 검증

NCNN 4클래스 배선 완료, 실거리·조명 인식률 미검증. 치명도: 초록불 미인식=**미션 실패** >
빨간불 +30s > 표지판(늦으면 B-9 표결창 재조정).
⚠️ yolo `conf_threshold`(0.5)는 **launch 시점 고정 — 라이브 변경 불가.** 약하면 재실행 필요.

### B-6. 아루코 사전·임계

`DICT_4X4_50`+inverted 는 사진 추정 → 당일 `tools/identify_aruco.py` 로 확정.
`marker_area_trigger`(0.02) 민감하면 "미리 멈춤=미션실패" → 등장 거리 기준 튜닝.
⚠️ 이 값도 **라이브 변경 불가**(decision live_tunable 목록에 없음) → 값 바꾸면 재실행.
검출 주기는 `aruco_hz`(12Hz, 라이브 가능)로 저속화됨 — 접근 마커 놓침 없는지 1회 확인.

### B-7. yellow_crossline 임계 실측

`crossline_min_width_ratio`(0.30)·`crossline_min_rows`(4)·스캔창(0.55~0.90)은 합성 테스트만
통과. 실제 노란 가로선 앞에서 debug `xline=1` 뜨는지 확인, `yellow_hsv_lo` 와 함께 조정.

### B-8. 역방향(시계방향) 실주행 검증

`race_dir:=right` flip 로직은 있으나 **역방향 실주행 이력 없음.** 배정 확률 50% → 1회 검증.

### B-9. ★ 갈림길 분기감지 + 브랜치 선택 튜닝 (Out 코스)

증상 → 손잡이 매핑 (0→4 순서로):
0. **방향을 아예 안 정함** → YOLO 검출 확인: `ros2 topic echo /inference/detections`
1. **브랜치 사이에서 헤맴 / 과하게 물어 코너 컷** → `fork_seed_px`(90) ↑/↓ (+`guide_margin_px` 60)
2. **지났는데 계속 쏠림 / 너무 일찍 풀림** → fork 지오메트리: `fork_min_groups`(3)·
   `fork_span_ratio`(0.65)·`fork_scan_top/bottom_ratio`(0.0~0.5). debug `fork=1` 이 분기에서만 뜨게
3. **반대 방향 확정 / 확정 늦음** → `fork_sign_vote_min`(2)·`fork_sign_min_conf`(0.5)·`fork_vote_clear_s`(1s)
4. 진입 보조: `fork_bias`(0.2). 좌/우 라벨↔실제 브랜치 매핑 1회 확인.
관찰: `ros2 topic echo /decision/fork_dir` + decision debug 로그(`forkDet/forkFix/latch`).

---

## 주행 중 라이브 조정 파라미터 총정리 (`ros2 param set`)

리빌드 없이 즉시 반영. **확정한 값은 해당 노드 `declare_parameter` 기본값에 박고 리빌드.**

### `/decision_node` (live_tunable 목록에 있는 것만 적용됨)

| 파라미터 | 기본 | 용도 / 언제 만지나 |
|---|---|---|
| `drive_throttle` | 0.20 | 직선 속도. 이탈하면 ↓, 기록 단축 ↑ |
| `slow_throttle` | 0.12 | 감속 하한 (ESC 데드밴드 위로) |
| `curve_slow` | 0.5 | 커브 감속 세기. 커브 이탈 시 ↑ |
| `kp` / `ki` / `kd` | 0.6 / 0 / 0.15 | 라인추종 PID. 반응 둔함 kp↑, 떨림 kd↑ or kp↓ |
| `steer_center` | 0.2 | 직진 드리프트 보정 바이어스 |
| `steer_scale` | -1.0 | 조향 부호/스케일 (부호 반대면 뒤집기) |
| `max_steering_delta` | 0 (off) | 틱당 조향 변화 상한 — 조향 튐 억제 |
| `steer_slow` | 0 (off) | 조향각 비례 감속 게인 |
| `fork_bias` | 0.2 | 갈림길 진입 보조 편향 |
| `fork_hold_s` | 8.0 | 락온 failsafe 상한 |
| `fork_sign_min_conf` | 0.5 | 표지판 표 최소 confidence |
| `fork_sign_vote_min` / `vote_window` | 2 / 5 | 방향 확정 표수 / 표결 창 |
| `fork_vote_clear_s` | 1.0 | 표지판 끊김 후 표결 리셋 |
| `roundabout_exit_gates` | 2 | 회전교차로 탈출 게이트 수 (트랙 실측) |

⚠️ **라이브 불가** (재실행 필요): `conf_threshold`, `green/red_frames`, `marker_area_trigger`,
`branch_bias`, `enter_curvature`, `yaw_lap_threshold`, `nominal_loop_time_s`, `race_dir`, `course`

### `/lane_node` (detector 속성은 사실상 전부 라이브)

| 파라미터 | 기본 | 용도 |
|---|---|---|
| `use_birdeye` / `birdeye_src_ratio` | true / (A-1 참고) | BEV on/off·캘리브. **A-1 최우선** |
| `roi_top_ratio` | 0.55 | ROI 시작(하단 45% 사용). 카메라 높이 바꾸면 재확인 |
| `white_hsv_lo/hi` | [0,0,180]/[179,60,255] | 흰선 검출. `tools/hsv_sampler.py` 로 실측 |
| `yellow_hsv_lo/hi` | [18,45,110]/[40,255,255] | 노란선 검출 (In 코스) |
| `use_yellow` | true (out이면 자동 false) | 노란 마스크 on/off |
| `smooth_alpha` | 0.5 | offset 평활. 떨림 ↓쪽, 반응 굼뜸 ↑쪽 |
| `guide_margin_px` / `guide_max_jump_px` | 60 / 80 | guided 탐색창 폭 / 튐 클램프 |
| `fork_seed_px` | 90 | **갈림길 주 튜닝 손잡이** (B-9) |
| `fork_min_groups` / `fork_span_ratio` / `fork_col_min_ratio` | 3 / 0.65 / 0.15 | 분기 구간 판정 |
| `fork_scan_top/bottom_ratio` | 0.0 / 0.5 | 분기 스캔밴드 |
| `crossline_min_width_ratio` / `crossline_min_rows` | 0.30 / 4 | 노란 가로선 판정 (B-7) |
| `lookahead_weight` / `adaptive_lookahead` | 0.3 / false | 커브 선제 조향. adaptive 는 트랙에서 실험 후 확정 |
| `debug_hz` | 10 | 디버그 이미지 발행 주기 (부하 시 ↓) |

### `/aruco_node`

| 파라미터 | 기본 | 용도 |
|---|---|---|
| `aruco_hz` | 12 | 검출 주기. 0 이하 = 매 프레임(구동작) |

### 관찰용 명령

```bash
ros2 topic echo /inference/detections     # YOLO 검출 (라벨·conf)
ros2 topic echo /decision/fork_dir        # 갈림길/출구 락온 방향
ros2 topic hz /perception/lane /camera/image/compressed   # 파이프라인 생존 확인
ros2 run decision decision_node --ros-args --log-level decision_node:=debug  # 상태/표/게이트 로그
```

---

## C. 완료 / 리스크 수용

- ✅ **부하 경감 (2026-07-06, 커밋 6650084)** — 07-05 SSH 끊김·대시보드 렉 대응:
  camera `debug_log` off / lane 디버그 이미지 30→10Hz(`debug_hz`) / aruco 검출 30→12Hz
  (`aruco_hz`) / control `command_hz` 10→20 (지연 절반) / joystick 디버그 로그 off (E-STOP 유지).
- ✅ **Out 코스 `use_yellow` 자동 off** — launch OpaqueFunction 으로 course 분기.
- ✅ **갈림길 재설계** — 홀드 이중적용 제거 + 브랜치 시드 락온 + fork 재수렴 해제.
- 운영 지식 (README 트랙 운영 노트 참고): 핫스팟 대신 **공유기 로컬 LAN** 지참 /
  실주행 중 웹 대시보드 탭 닫기 / 출발 순간 배터리 % 급락은 LiPo 전압 sag = **정상**.
- 차선 이탈 자가 복구 없음(lane lost 시 직진·재획득 대기) — 이탈 방지 튜닝(B-3)에 투자(수용).
- I2C wedge 는 소프트웨어로 못 막음(재시도+respawn까지). **진짜 비상정지 = ESC 전원 / 조이스틱 X버튼 E-STOP.**
- 확정 구성: BEV·guided band·look-ahead·HSV = 기본 ON / `adaptive_lookahead` = OFF.
  모드 토글은 디버그 스위치로 유지, dead-code 정리는 레이스 후.

---

## 레이스 전 체크리스트 (우선순위순)

1. [ ] **BEV 재캘리브레이션** (A-1) ★★ — 현재 높이 기준. 이거 없인 아래 전부 무의미
2. [ ] 리빌드 반영 확인 (`colcon build --packages-select camera control perception decision`)
3. [ ] kp/steer/throttle 튜닝으로 S자 완주 (B-3, `skip_missions:=true`)
4. [ ] **초록불/빨간불 실거리 인식 검증** (B-5) ★
5. [ ] **갈림길: 표지판 인식 + 분기감지·시드 튜닝** (B-9) ★ (Out 코스)
6. [ ] In 코스 갈 거면: 전환구간 노란색 커밋 모드 구현 (A-2) + 랩 캘리브 (B-2) ★
7. [ ] yellow_crossline 임계 실측 (B-7)
8. [ ] 아루코 사전/임계 확정 (B-6) — 12Hz 검출로 놓침 없는지 포함
9. [ ] 역방향 1회 검증 (B-8)
10. [ ] 장비: **공유기(로컬 LAN)**, 게임패드(E-STOP), 충전된 LiPo 예비
