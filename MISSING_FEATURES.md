# 남은 작업 — 레이스 준비 현황 (2026-07-13 갱신)

레이스: **7/15 (1차), 7/16 (2차)**. 순위는 1·2차 중 최고 기록. **★ = 레이스 전 필수.**

미션 6종 전부 구현 완료. In 코스는 **run80 풀 미션 무개입 완주**로 실증됐고,
07-13 재설계(STOPLINE 분류기·병합 브리지·곡률 가드·Out 상시 코리도·갈림길 시야
마스킹)는 오프라인 전수 검증(프레임 6세트 + 텔레메트리 10런 시뮬)을 통과했다.
남은 것은 **검증·캘리·하드웨어**다. (이력·근거: RUN_LOG_2026-07-12.md)

---

## A. 실차 검증 (우선순위 순)

### A-1. ★ In 코스 — 07-13 재설계 스택 검증 런
run80 완주는 이전 스택 기준. 이후 들어간 [STOPLINE(stopline_mode 1 + blank 3.5) +
병합 브리지 + 곡률 가드]는 오프라인 검증만 됨 → **충전 후 풀 체인 1~2런**.
- 성공 기준: A 재도달 앵커 발화(`gate=1 exitV=2`) → 우측 탈출 → 병합(표류 없음)
  → 마커 정지/재출발 → 빨간불 정지
- 실패 시 킬스위치(전부 라이브): `stopline_mode 0`+`gate_blank_s 7.0` /
  `merge_bridge_s 0`+`w_align_dash_fallback 1` / `sw_curv_max_a 0`

### A-2. ★ Out 코스 — 첫 실주행 (전 구간 실차 이력 0회)
상시 코리도(`sw_out_always`) + 갈림길 시야 마스킹(`fork_blind_frac`) + 표지판 표결.
- 선행: **표지판 벤치** (배터리 거의 안 씀) — `/tmp/launch_out.sh` 로 올려 WAIT_GREEN
  정지 상태에서 left/right 인쇄물 → `confirmed_fork_direction` 로그 확인
- 실주행은 단계적으로: S자 추종만 → 갈림길 → 풀 체인
- 갈림길 증상→손잡이 맵 (현장용):
  0. 방향 미확정 → `ros2 topic echo /inference/detections` 로 YOLO 검출부터
  1. 마스킹 안 걸림 → `/decision/fork_dir` 발행 확인, `fork_blind_frac`(0.40)·`fork_blind_frames`(60)
  2. 브랜치 사이 헤맴 → `fork_seed_px`(90), `guide_margin_px`(60)
  3. 지나고도 쏠림/조기 해제 → `fork_min_groups`(3)·`fork_span_ratio`(0.65) — debug `fork=1` 확인
  4. 반대 방향/늦은 확정 → `fork_sign_vote_min`(2)·`fork_sign_min_conf`(0.5)

### A-3. In 코스 재현성 — 배터리 밴드별
게이트는 랩 15~30s 전 대역 시뮬 통과했으나 실차 재현은 특정 밴드에 몰림.
새 팩(완충) 고속 밴드에서 1런 확인.

### A-4. 역방향 (`race_dir:=right`) — 실주행 이력 없음, 배정 확률 50%
flip 로직은 있음. ⚠️ **곡률 가드(`sw_curv_max_a`)는 left 전제 구현** — right 추첨 시
라이브로 0(off) 하고 주행할 것.

---

## B. 대회 당일 체크리스트 ★

1. **빨간 노면**: 연습 트랙(수제)엔 없음 — 현장 정적 확인: 차를 빨간 구간에 놓고
   `rr` ≥ `red_slow_ratio`(0.05) 뜨는지. 안 뜨면 `red_hsv_*` 라이브 조정
2. **ArUco**: 대회 지급 마커 **사전/ID 스캔** (전 사전 detectMarkers 스윕 or
   `tools/identify_aruco.py`). 사전 불일치 = 원리적 무검출 (run76~78, 3런 소모한 교훈).
   현재 기본 `DICT_6X6_50`(연습 마커 id3 기준)
3. **신호등**: 실거리 green/red 인식률 + `light_min_area` 캘리 (현재 0=off —
   run69/72 오FINISH 이력). `finish_min_drive_s`(60s)는 실측 코스 시간보다 길게.
   `conf_threshold` blind 상향 금지(초록불 미인식=미션실패 리스크)
4. 카메라 마운트 점검(BEV `birdeye_src_ratio` 종속 — 만졌으면 재캘리),
   /tmp 스크립트 재배포, 핫스팟은 2.4GHz + 기기 격리 없는 것(아이폰 주의)

---

## C. 하드웨어

- ★ **배터리 팩 교체/보강**: 현 팩 완충 무부하 7.9V(정상 8.4V), 부하 새그 ~1V,
  무부하 7.7V 미만 주행 시 등판 정지·브라운아웃 위험. 7/12~13 실패 런 다수의 근본 원인
- Wi-Fi 동글 불안정(주행 중 사망 이력 다수) — 여분 동글/이더넷 케이블 지참.
  UART 복구: 콘솔(115200, topst/topst) → `nmcli device wifi connect "SSID" password "PW"`

---

## D. 선택 (여유 있으면)

- V-피드포워드(전압→스로틀): 캘리 산점 축적됨(250cm 출발 + 링 9129.72mm). 새 팩이면 후순위
- FINISH 이중화: light_min_area + 위치 게이트
- `D-Racer-Kit/AUTODRIVE.md` 옛 설계 서술 정합 (주행 영향 없음)
- 코드 정리(dead-path)는 레이스 후

---

## 참고: 자주 쓰는 라이브 파라미터 (`ros2 param set`, 리빌드 불필요)

| 파라미터 (노드) | 현재값 | 용도 |
|---|---|---|
| `drive_throttle` (decision) | 0.19 | 흰 구간 속도 (틱 327) |
| `yellow_drive_throttle`/`slow_throttle` | 0.165 기본, 0.17 운용 | 노랑/링 속도. **틱 사다리 주의: 0.16≡0.165(324), 0.17(325)** |
| `curve_slow` / `curve_steer_bias` | 0.5 / — | 커브 감속 / 급커브 선행 조향 |
| `kp`/`kd`, `steer_center`(0.26) | 0.6/0.15 | 라인추종 PID / 직진 보정 |
| `gate_blank_s` / `gate_cluster_on_s` | 3.5 / 0.20 | 게이트 블랭크 / 군집 카운트 문턱 |
| `roundabout_exit_gates` / `max_loop_time_s` | 2 / 75 | 발화 군집 수 / 강제 탈출 상한 |
| `yaw_lap_threshold` | 7.0 | 백업 표결 yaw (사실상 봉인값) |
| `merge_bridge_s` / `merge_blind_bias` | 6.0 / 0.10 | 병합 브리지 창 / 좌호 바이어스 |
| `marker_area_trigger` / `marker_clear_frames` | 0.02 / 8 | 마커 정지 트리거 / 재출발 판정 |
| `stopline_mode`·`sw_curv_max_a`·`crossline_sw_gate` (lane) | 1 / 0.003 / 1 | 정지선 3층 방어 킬스위치 |
| `sw_out_always`·`fork_blind_frac` (lane) | 1 / 0.40 | Out 상시 코리도 / 갈림길 마스킹 |

확정한 값은 해당 노드 `declare_parameter` 기본값에 박고 리빌드.
