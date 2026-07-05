# 미구현 / 미완성 / 수정필요 정리 (트랙·미션 기준)

기준: **OT 자료 10~19p (트랙·미션 소개)** vs 현재 코드 (**2026-07-05**).
레이스: 7/15 (1차), 7/16 (2차). 순위는 1·2차 중 **최고 기록** 반영.
**굵은 항목이 레이스 전 필수.**

- 코스: Out(Base) = 출발→S자→**좌/우 갈림길**→동적 장애물→도착 (11p)
       / In(Option) = 출발→**회전 교차로**→동적 장애물→도착. In/Out은 팀 선택(`course:=`).
- 트랙 방향: 당일 정/역방향 추첨 배정, 변경 불가 (13p) → `race_dir:=left|right` 하나로 flip.
- 미션 실패 = 미션 시작지점으로 **수동 복귀**(팀장), 시간 페널티 없음 / 미션 포기 = +2분
  (회전교차로는 포기 불가) (13p). **차량은 자가 복귀를 하지 않는다 — 설계상 수동.**

---

## 미션별 구현 현황 (10~19p 대조)

| 미션 (PDF) | 규정 요약 | 코드 | 상태 |
|---|---|---|---|
| 출발·신호등 (14p) | 초록불 점등 후 출발. 초록불 미인식=미션실패, 준비 5분초과=주행실패 | `WAIT_GREEN` (green 3프레임 연속) | ✅ 구현 / YOLO 실거리 검증 필요(B-5) |
| S자 (15p) | 미션 없음, 곡률 저속 권장. 이탈 +30s | `DRIVE` + `curve_slow` 곡률 감속 | ✅ 구현 / kp·throttle 튜닝(B-3) |
| 좌/우 갈림길 (16p) | 랜덤 표지판 방향대로. 방향 어기면 미션실패. 이탈 +30s | 표결 확정→`turn_latch`→**브랜치 선택**(perception guide 시드)→`fork` 재수렴 시 해제 | ✅ 로직 재설계 완료(A-1 버그 제거·median 회피) / **트랙에서 분기감지·시드 튜닝 필요**(B-9) |
| 동적 장애물 (17p) | 등장 시 정지·퇴거 시 출발. 미리 멈춤/일찍 출발=미션실패. 이탈 +30s | `OBSTACLE_STOP` (면적비 트리거) | ✅ 구현 / 아루코 사전·임계 튜닝(B-6) |
| 도착·신호등 (18p) | 완주 후 빨간불 점등 시 정지. 빨간불 미인식 +30s | `DRIVE→FINISH→DONE` (red 3프레임) | ✅ 구현 / 빨간불 실거리 검증(B-5) |
| 회전 교차로 (19p) | 1회 이상 회전 후 탈출. 이탈=미션실패+이탈 동시, **포기 불가** | `ROUNDABOUT` 진입 2-of-3 표결 / **탈출=게이트 2회→출구 브랜치 락온**(+ 4-표결 failsafe) | ⚠️ 탈출 로직 재설계 완료 / **게이트수·출구side·랩 캘리브레이션 필요**(B-2) |

코드는 6개 미션을 모두 상태머신에 반영했고 구조는 건전하다. 남은 건 **버그 1건(A)**과
**트랙 실측·튜닝(B)**이다.

---

## A. 지금 코드에서 고쳐야 할 것 (실측 없이 판단 가능)

### A-1 / A-2. ✅ 갈림길 로직 재설계 완료 (홀드 이중적용 제거 + 브랜치 선택)

전제: **표지판이 두 분기 사이(median)에 있음** → "표지판 확정 = 분기 도착". 상수 편향을
길게 유지하던 옛 방식을 버리고, **선택한 브랜치의 실제 차선만 추종**하도록 재설계했다.

바뀐 동작:
- **홀드 이중적용(옛 A-1) 제거.** `decision_node`의 표결-소멸은 `fork_hold_s`가 아니라
  짧은 `fork_vote_clear_s`(1s, 떨림방지창)만 담당. 갈림길 통과까지의 유지는 `state_machine`
  `turn_latch` **한 곳**이 담당하고, 해제 기준은 타이머가 아니라 **형상(도로 재수렴)**이다.
  ([state_machine.py](D-Racer-Kit/src/decision/decision/state_machine.py) — `_fork_seen` 후 `lane.fork` 하강엣지에 해제, `fork_hold_s`=8s는 failsafe 상한만.)
- **브랜치 선택(median 회피).** `perception`이 분기를 감지(`lane.fork`)하고, `decision`이
  확정 방향을 `/decision/fork_dir`로 전달하면 `lane_detector`가 guided-band 시드를 그
  브랜치 쪽으로 밀어(`fork_seed_px`) **한쪽 브랜치만 추종** → 두 브랜치 평균으로 표지판
  섬에 돌진하던 실패모드 제거. `_turn_bias` 상수는 진입 보조로 축소(`fork_bias` 0.4→0.2).
- **분기 "구간" 인식 추가(옛 A-2 해소).** `lane_detector._detect_fork`가 BEV 상단 스캔밴드의
  세로 라인 군집 수/바깥 span 으로 분기 구간을 판정. 이 신호로 진입·해제 타이밍을 형상에 맞춤.

남은 일 → **B-9**(트랙에서 분기감지 임계·시드 실측 튜닝). 분기 후 "올바른 쪽으로 들어갔는지"
자가 확인은 규정상 수동이라 미구현.

### A-3. ✅ 문서 정합성

AUTODRIVE.md 갈림길 서술은 위 재설계에 맞춰 갱신 대상(옛 "6초 홀드" 서술 폐기).

---

## B. 트랙에서 실측·튜닝해야 하는 것 (코드 결함 아님)

### B-2. ★ 회전교차로 캘리브레이션 (In 코스 선택 시 필수)

진입 = `yellow_crossline`/`junction`/`|curvature|≥enter_curvature` **2-of-3 표결** +
노란비율 게이트 + `enter_sustain_s`(0.6s).

**탈출(2026-07-05 재설계) = 갈림길과 동일한 브랜치 락온.** 링을 돌며 게이트(노란 가로선
`yellow_crossline` **or** 점선 개구부 `junction`)의 **상승엣지**를 세고(쿨다운 디바운스),
`roundabout_exit_gates`(기본 2)회 도달 + `min_loop_time_s`(3s) HARD floor 지나면 **출구
브랜치로 `turn_latch` 락온**해 명시적으로 빠져나간다(옛 "바이어스만 해제" 암묵 탈출 대체).
출구 side 는 `roundabout_exit_side`('' 이면 `race_dir` 파생: 정방향=오른쪽/역방향=왼쪽).
락온 해제는 갈림길과 동일(`fork` 재수렴 or `fork_hold_s` failsafe). **게이트 검출이
한 번 빠져도 못 나가는 최악을 막으려** 기존 yaw/time/junction **2-of-4 표결을 failsafe로
유지**(이땐 락온 없이 링 해제만).

보정 안 된 값:
- `roundabout_exit_gates: 2` — **한 바퀴에 게이트를 몇 번 지나는지 트랙에서 확인**해 맞춤
  (진입1+한바퀴1=2 가정). debug 로그 `gate=` 로 실측. 라이브 튜닝 가능.
- `roundabout_exit_side` — 정방향 출구가 카메라 기준 오른쪽 브랜치인지 1회 확인(아니면 지정).
- `enter_curvature: 0.45` — 원래 `|offset|` 기준인데 지금 `|curvature|`에 적용됨.
  curvature 스케일은 더 작아(≈0.1~0.3) 현 값은 거의 안 걸림 → **0.15~0.25로 재튜닝.**
  (안 하면 curvature 표가 죽어 crossline+junction 2표에만 의존)
- `yaw_lap_threshold: 6.0` — 한 바퀴 조향 적분값 실측(failsafe 표용).
- `nominal_loop_time_s: 8.0` — 레이스 속도 한 바퀴 시간 실측(failsafe 표용).
한 바퀴 돌려 decision debug 로그(`entryV/exitV/gate/curv`)로 읽고 기본값 반영.
규정상 회전교차로는 미션 포기 불가라 In 코스면 필수. (Out만 뛰면 무시 가능)

### B-3. S자·기본 주행 튜닝

`skip_missions:=true`(순수 차선추종)로 kp/kd, `steer_center/scale`, `drive_throttle`,
`curve_slow` 튜닝해 S자 완주. 필요 시 `max_steering_delta`(틱당 조향 제한),
`steer_slow`(조향 비례 감속)로 튐/이탈 억제.

### B-5. ★ YOLO 신호등/표지판 실거리 인식 검증

NCNN 4클래스(`red/green_light`, `left/right_sign`) 모델은 배선됐으나 실제 트랙
거리·조명에서 인식률 미검증. 치명도 순:
- 초록불 미인식 = **미션 실패**(가장 치명적, 14p)
- 빨간불 미인식 = +30s (18p)
- 표지판 인식거리 짧으면 A-1 홀드·표결창(`fork_sign_vote_window/min`) 재조정 필요
`conf_threshold`(0.5)·`fork_sign_min_conf`(0.5)·추론 주기도 현장 조정.

### B-6. 아루코(동적 장애물) 사전·임계

`DICT_4X4_50`+inverted는 사진 추정값 → 당일 실물로 `tools/identify_aruco.py` 확정.
`marker_area_trigger`(0.02)가 너무 민감하면 규정상 "미리 멈춤=미션 실패"(17p)이므로
등장 거리 기준으로 튜닝.

### B-7. yellow_crossline 임계 실측

`crossline_min_width_ratio`(0.30)·`crossline_min_rows`(4)·스캔창(0.55~0.90)은 합성
테스트만 통과. 실제 노란 가로선 앞에서 `xline=1` 뜨는지 모니터/디버그로 확인,
필요 시 `yellow_hsv_lo`와 함께 조정.

### B-8. 역방향(시계방향) 트랙 실주행 검증

`race_dir:=right` 하나로 회전교차로 방향·junction 탐색이 함께 뒤집히게 돼 있으나
**역방향 실주행 테스트 이력 없음.** 당일 역방향 배정 확률 50% → 연습 때 1회 검증.

### B-9. ★ 갈림길 분기감지 + 브랜치 선택 튜닝 (재설계 로직, Out 코스)

로직은 구현·빌드 완료(A-1/A-2). **실제 분기 모습으로 임계를 맞춰야** 동작한다:
- `fork_min_groups`(3)·`fork_span_ratio`(0.65)·`fork_col_min_ratio`(0.15)·스캔밴드
  (`fork_scan_top/bottom_ratio` 0.0~0.5): 분기 앞에서 debug 로그 `fork=1`이 **분기에서만**
  뜨는지 확인(직선/코너 오검출 없게). 모니터 debug 영상으로 BEV 상단 라인 군집 보며 조정.
- `fork_seed_px`(90): 방향 확정 후 차가 **표지판 섬을 피해 올바른 브랜치로** 붙는지.
  너무 작으면 median으로, 너무 크면 바깥선 밖으로. `guide_margin_px`(60)와 함께 튜닝.
- `fork_bias`(0.2, 라이브)·`fork_vote_clear_s`(1s)·`fork_hold_s`(8s failsafe).
- 좌/우 매핑: 표지판이 정면 median에 있어 **카메라 left 표지판 = 카메라 왼쪽 브랜치**로
  직결. YOLO 라벨(`left_sign`/`right_sign`)이 실제 방향과 맞는지 1회 확인.
- 신규 토픽 `/decision/fork_dir`가 흐르는지: `ros2 topic echo /decision/fork_dir`.

---

## C. 이미 완료 / 리스크 수용

- **BEV src 캘리브레이션 완료 (2026-07-05)**: `birdeye_src_ratio =
  [0.262,0.05, 0.811,0.05, 1.017,0.95, 0.034,0.95]` (워프 후 잔차 ≤1.5px).
  lane_node.py 기본값 반영·리빌드 완료. ⚠️ **카메라 마운트 다시 만지면 재캘리브레이션.**
- **차선 이탈 복구 로직 없음**: lane lost 시 offset 0 직진·재획득 대기가 전부. 완전 이탈
  시 자가 복귀 불가(규정상 팀장 수동 복귀 +30s). 적극 탐색(조향 스윕)은 미구현 —
  **이탈을 막는 튜닝(B-3)에 시간 쓰는 게 낫다**(리스크 수용).
- **하드웨어 리스크**: I2C 버스 wedge는 소프트웨어로 못 막음(재시도+respawn까지).
  **진짜 비상정지는 ESC 전원 스위치 / 조이스틱 E-stop.**

---

## 레이스 전 체크리스트 (우선순위순)

1. [x] ~~BEV src 캘리브레이션~~ 완료 (C; 카메라 마운트 바꾸면 재실행)
2. [x] ~~A-1 갈림길 홀드 이중적용~~ 재설계로 제거 (A-1/A-2, 코드·빌드 완료 / 트랙 튜닝=B-9)
3. [ ] **초록불/빨간불 실거리 인식 검증** (B-5) ★
4. [ ] kp/steer 튜닝으로 S자 완주 (B-3)
5. [ ] **갈림길: 표지판 실거리 인식 + 분기감지·시드 튜닝** (B-9) ★
6. [ ] yellow_crossline 임계 실측 (B-7)
7. [ ] 아루코 사전/임계 확정 (B-6)
8. [ ] 역방향 1회 검증 (B-8)
9. [ ] In 코스 갈 거면 `enter_curvature` 재튜닝 + 랩 캘리브레이션 (B-2) ★
