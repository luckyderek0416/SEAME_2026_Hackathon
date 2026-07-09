# 레이스 준비 현황 — 보완·튜닝·주행중 파라미터 총정리

기준: **OT 자료 10~19p** vs 현재 코드 (**2026-07-07**).
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
| 동적 장애물 | 등장 시 정지·퇴거 시 출발 | `OBSTACLE_STOP` (면적비 트리거, 12Hz 검출, **DRIVE 전용** — 코스상 RA 뒤 마지막 미션이라 RA 중 마커 오반응 차단) | ✅ 구현 / 사전·임계 튜닝(B-6) |
| 도착·신호등 | 빨간불 정지. 미인식 +30s | `DRIVE→FINISH→DONE` (red 3프레임×2 + 오인식 가드) | ✅ 구현 / 실거리 검증(B-5) |
| 회전 교차로 | 1회전 후 탈출. **포기 불가** | 노란 추종→가로선 1회=진입 / 가로선 +1회=락온 탈출(+2-of-4 failsafe) | ⚠️ 로직 완료 / 가로선·색전환 캘리(B-2) ★ |

6개 미션 모두 상태머신에 반영, 구조 건전. 남은 건 **보완 2건(A)** + **트랙 실측·튜닝(B)**.

---

## A. 보완이 필요한 것 (코드/캘리브 수정)

### A-1. ✅ BEV 재캘리브레이션 — **완료 (2026-07-08, 카메라 높이 28cm 재실측)**

직선 구간 실프레임 캡처 → 좌/우 흰선 직선 피팅 → y=0.342h/0.965h 의 차선 x 를 src
꼭짓점으로 사용(차선이 워프 후 정확히 0.20w/0.80w). 검증: slope L-0.021/R+0.002,
차선폭 top 191 / bottom 193px (목표 192px=실차선 350mm).
- 확정값: `roi_top_ratio=0.35` (하단 65% 사용) /
  `birdeye_src_ratio=[0.288,0.342, 0.713,0.342, 0.857,0.965, 0.150,0.965]`
  ⚠️ **카메라 마운트 다시 만지면 재실행.** (07-07 낮은 마운트 값은 lane_node 주석에 보존)

### A-2. ✅ In 코스 색상 추종 상태머신 — **구현 완료 (2026-07-07, 방안 A)**

흰+노랑 합산 마스크의 전환구간 애매함 해결: WHITE(흰 전용 추종) ↔ YELLOW(노란 전용
추종) 히스테리시스. 노란색이 조금이라도 보이면(`follow_yellow_ratio`=0.03) YELLOW 진입,
흰 픽셀 > 노란 픽셀이면 WHITE 복귀. **ROUNDABOUT 동안은 YELLOW 강제 유지.**
`course:=out` 이면 자동 비활성(launch 전달). BEV 디버그에 `FOLLOW-W/Y` 표시.
남은 것: 전환 문턱 실측(B-2).

### A-3. 문서 정합성

`D-Racer-Kit/AUTODRIVE.md` 의 갈림길("6초 홀드")·회전교차로(2-of-3 표결 진입) 서술이
07-07 재설계(브랜치 락온·가로선 트리거·색상 추종)와 불일치 → 갱신 대상. 주행 영향 없음.

---

## B. 트랙에서 실측·튜닝해야 하는 것 (코드 결함 아님)

> 순서: skip_missions 기본 추종(B-3) → YOLO 검증(B-5) → 미션별 튜닝(B-2/B-6/B-7/B-9).
> (BEV 재캘리브는 07-07 완료 — 카메라 마운트 안 만지는 한 재실행 불필요)

### B-2. ★ 회전교차로 캘리브레이션 (In 코스 시 필수)

**인코스 플로우(2026-07-08 확장):** 초록불→흰선 추종 DRIVE → 노란색이 보이면
(≥`follow_yellow_ratio`) **노란 실선만 추종**(FOLLOW-Y; 점선/정지선은 track 에서 성분
필터로 제외, raw 신호는 유지) → 노란 가로선(정지선) **처음** 보이면 ROUNDABOUT on
**+ 진입측 one-shot 락온**(링 순환 방향; `entry_lock_release_s`=2s 후 자동 해제, 링 위
fork 오발동으로 재락온 불가) → 같은 가로선을 **한 번 더**(`roundabout_exit_gates`=1)
만나면 출구 브랜치 락온으로 탈출(정방향=우회전; 진입선 재카운트는
블랭크 `gate_blank_s`=6s + 재무장 `gate_rearm_s`=0.5s 연속 OFF 로 차단) → 노란 추종
유지 → **흰색>노란색** 검출되면 흰선 추종 복귀. ROUNDABOUT 동안은 강제 노란 추종.
기존 안전장치 유지: `min_loop_time_s`(3s) HARD floor, `crossline_cooldown_s` 디바운스,
**3-표결 failsafe**(yaw/time/crossline; junction 표는 오검출로 제거, 랩카운트 코드 삭제),
`max_loop_time_s`. 백업·강제 탈출도 주 탈출과 동일하게 **출구측 락온**을 건다
(락온 없이 DRIVE 복귀 시 FOLLOW-Y 가 링을 계속 따라 무한 순환 위험). 단선(실선 1개) 추종 안전장치: 차선폭 EMA 하한 가드
(측정폭 < `lane_width_init`×0.5 는 학습 스킵).

보정 안 된 값 (debug 로그 `entryV/exitV/gate` + BEV `FOLLOW-Y/W` 표시로 실측):
- **`yellow_crossline` 검출 신뢰도(B-7)가 이제 진입·탈출 둘 다의 핵심** — 가로선 앞에서
  `xline=1` 안정적으로 뜨는지 최우선 확인
- `roundabout_exit_gates: 1` — 진입 후 가로선 추가 조우 횟수(1=한 바퀴). 라이브 튜닝 가능
- `roundabout_exit_side` — 정방향 출구가 오른쪽 브랜치인지 1회 확인
- `follow_yellow_ratio: 0.03` / `follow_yellow_exit_white_ratio: 1.0` — 색상 전환 문턱 실측
- `enter_sustain_s: 0.3` — 가로선 지속 debounce (라이브 튜닝 가능)
- `yaw_lap_threshold: 6.0` / `nominal_loop_time_s: 8.0` — 한 바퀴 실측 (failsafe 표용)
- ~~`enter_curvature`~~ — 진입이 가로선 단독 트리거로 바뀌어 **미사용**

### B-3. S자·기본 주행 튜닝

`skip_missions:=true` 로 kp/kd, `steer_center/scale`, `drive_throttle`, `curve_slow` 튜닝.
튐/이탈 시 `max_steering_delta`(틱당 조향 제한), `steer_slow`(조향 비례 감속) 활용.
07-05 이탈은 속도보다 **BEV 문제(A-1)가 먼저** — 캘리브 후 재평가.

### B-5. ★ YOLO 신호등/표지판 실거리 인식 검증

NCNN 4클래스 배선 완료, 실거리·조명 인식률 미검증. 치명도: 초록불 미인식=**미션 실패** >
빨간불 +30s > 표지판(늦으면 B-9 표결창 재조정).

**빨간불 오인식 가드 (07-07 추가, 07-08 구조화)** — 주행 중 red_light 검출(멀리 보이는
진짜 도착 신호등 or 오분류)로 코스 중간에 FINISH/DONE 되던 사고 대응:
- **주 무장 = 장애물 미션 완료(`obstacle_done`)**: 코스 순서 고정(… → 동적 장애물 →
  도착)이므로 장애물 정지→퇴거 1회를 마친 뒤에만 빨간불 대기 시작. 그 전엔 완전 무시.
- `finish_min_drive_s`(60s): **예비 무장** — 아루코를 통째로 놓친 비상 주행에서도 이
  시간 경과 시 빨간불 인식을 켠다. **실측 코스 소요시간보다 길게** 재설정. 라이브 가능.
- `light_min_area`(0=off): 신호등 bbox 최소 면적(정규화 w×h) — 먼 오검출 필터.
  `/inference/detections` 로 실물/오검출 박스 크기 재고 설정. 라이브 가능.
- FINISH→DONE 도 red 3프레임 연속 필요 (기존 1프레임).
- decision `conf_threshold` 는 이제 **라이브 가능** — 실물 인식 conf 를 재고, 오인식과의
  사이에 문턱. **측정 없이 올리면 초록불 미인식(=미션실패) 리스크** — blind 상향 금지.

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
| `roundabout_exit_gates` | 1 | 진입 후 가로선 추가 조우 횟수 (트랙 실측) |
| `enter_sustain_s` | 0.3 | 진입 가로선 지속 debounce |
| `entry_lock_release_s` | 2.0 | 진입측 락온 강제 해제 (one-shot) |
| `gate_blank_s` / `gate_rearm_s` | 6.0 / 0.5 | 게이트 블랭크 / 재무장(연속 OFF) — 진입선 재카운트 차단. 블랭크는 실측 한바퀴 시간보다 짧게 |
| `crossline_cooldown_s` | 2.0 | 게이트 카운트 간 최소 간격 |
| `circle_steer_bias` | 0.225 | 회전교차로 링 유지 편향 (이탈 시 ↑, 출구 과회전 시 ↓) |
| `curve_steer_bias` | 0 (off) | DRIVE 급커브 feed-forward — 급좌커브 바깥 튕김 억제 (0.2~0.5 실험) |
| `steer_left_gain` | 1.0 (off) | 좌조향 비대칭 보정 (07-07 원 테스트: 좌우 대칭 확인, 1.0 확정) |
| `branch_bias` | 0 (off) | In/Out 색상 편향 — FOLLOW-Y 로 대체돼 기본 OFF, 현장 재활성용 |
| `finish_min_drive_s` | 60.0 | 빨간불 예비 무장 시간 — 주 무장은 장애물 완료(B-5 가드) |
| `light_min_area` | 0 (off) | 신호등 bbox 최소 면적 (B-5 가드) |
| `conf_threshold` | 0.5 | YOLO confidence 문턱 (측정 후에만 상향 — B-5) |

⚠️ **라이브 불가** (재실행 필요): `green/red_frames`, `marker_area_trigger`,
`yaw_lap_threshold`, `nominal_loop_time_s`, `race_dir`, `course`

### `/lane_node` (detector 속성은 사실상 전부 라이브)

| 파라미터 | 기본 | 용도 |
|---|---|---|
| `use_birdeye` / `birdeye_src_ratio` | true / (A-1 확정값) | BEV on/off·캘리브. **07-07 실측 완료** |
| `roi_top_ratio` | 0.35 | ROI 시작(하단 65% 사용). 카메라 높이 바꾸면 재확인 |
| `follow_yellow` / `follow_yellow_ratio` | true / 0.03 | In 코스 색상 추종 (A-2). 노란 점선이라 문턱 낮음 |
| `follow_yellow_exit_white_ratio` | 1.0 | 흰>노랑×배율이면 WHITE 복귀 |
| `filter_yellow_dashes` / `yellow_solid_min_h_ratio` | true / 0.30 | Y추종 중 점선·정지선 track 제외(실선만) |
| `yellow_dash_fallback_px` | 120 | 실선 픽셀 이 미만이면 그 프레임은 점선 포함 폴백 — 급좌회전에서 안쪽 실선 시야 이탈 대응 |
| `lane_width_init` | 192 | 차선폭 프리셋(px) = 실측 350mm 의 BEV 환산. 재캘리 시 재확인 |
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
- ✅ **BEV 재캘리브 (07-07)** — roi 0.35 + 실측 프레임 피팅 src (A-1).
- ✅ **In 코스 색상 추종 상태머신 (07-07)** — WHITE↔YELLOW 히스테리시스, RA 중 YELLOW
  강제, `FOLLOW-W/Y` 디버그 표시 (A-2).
- ✅ **회전교차로 가로선 트리거 재설계 (07-07)** — 진입=가로선 첫 감지(0.3s 지속),
  탈출=가로선 +1회 → 출구 브랜치 락온. 2-of-4 failsafe·시간 하한 유지 (B-2).
- ✅ **빨간불 오인식 가드 (07-07)** — `finish_min_drive_s`/`light_min_area`/DONE 3프레임 (B-5).
- ✅ **급커브·조향 보조 (07-07)** — `curve_steer_bias`(feed-forward)·`steer_left_gain`
  (좌조향 비대칭)·`lane_width_init`(350mm→192px 프리셋)·`circle_steer_bias` 0.225.
- 운영 지식 (README 트랙 운영 노트 참고): 핫스팟 대신 **공유기 로컬 LAN** 지참 /
  실주행 중 웹 대시보드 탭 닫기 / 출발 순간 배터리 % 급락은 LiPo 전압 sag = **정상**.
- 차선 이탈 자가 복구 없음(lane lost 시 직진·재획득 대기) — 이탈 방지 튜닝(B-3)에 투자(수용).
- I2C wedge 는 소프트웨어로 못 막음(재시도+respawn까지). **진짜 비상정지 = ESC 전원 / 조이스틱 X버튼 E-STOP.**
- 확정 구성: BEV·guided band·look-ahead·HSV = 기본 ON / `adaptive_lookahead` = OFF.
  모드 토글은 디버그 스위치로 유지, dead-code 정리는 레이스 후.

---

## 레이스 전 체크리스트 (우선순위순)

1. [x] ~~BEV 재캘리브레이션~~ (A-1) 07-07 완료 — 카메라 마운트 만지면 재실행
2. [ ] 리빌드 반영 확인 (`colcon build --packages-select camera control perception decision`)
3. [ ] kp/steer/throttle 튜닝으로 S자 완주 (B-3, `skip_missions:=true`)
4. [ ] **초록불/빨간불 실거리 인식 검증 + `finish_min_drive_s` 실측** (B-5) ★
5. [ ] **갈림길: 표지판 인식 + 분기감지·시드 튜닝** (B-9) ★ (Out 코스)
6. [ ] **In 코스: 색상 전환(FOLLOW-W/Y) 문턱 + 가로선 진입·탈출 검증** (B-2) ★
7. [ ] yellow_crossline 임계 실측 (B-7) — 이제 회전교차로 진입·탈출의 핵심 신호
8. [ ] 아루코 사전/임계 확정 (B-6) — 12Hz 검출로 놓침 없는지 포함
9. [ ] 역방향 1회 검증 (B-8)
10. [ ] 장비: **공유기(로컬 LAN)**, 게임패드(E-STOP), 충전된 LiPo 예비
