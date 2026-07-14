# 인코스 완주율 감사 — 2026-07-14 (초기 진단)

목적: 차선이탈·미션실패 없는 완주를 위한 최적화 디버깅의 출발점.
대상: 현행 스택 (git 1c05f1a, run108 이후). 방법: 5개 차원 병렬 코드 감사
(상태머신 / 차선 파이프라인 / 인지 보조 / 제어·구동 / 실패이력 대조) 후
발견별 적대적 검증 — **검증 통과 15/15건**, 아래는 전부 file:line 재확인 완료.

---

## 0. 결론 — 즉시 조치 Top 5

| # | 조치 | 근거 |
|---|---|---|
| 1 | **aux.sh 운용값 15개를 소스 기본값으로 박제** (재부팅 시 /tmp 램디스크 소실!) | §2-A1 |
| 2 | **ra_direct_fire ↔ stopline_mode 인터록** (반쪽 주입 = B 오발 구조적) | §2-A2 |
| 3 | **스테일 페일세이프 3종** (control 타임아웃 / lane 나이 검사 / decision respawn) | §2-A3 |
| 4 | **FINISH 오발 봉쇄** (finish_min_drive_s 60s는 run73 완주 70.4s보다 짧음 + red_frames 3틱은 오추론 1건에 뚫림 + light_min_area 0) | §2-B1 |
| 5 | **물리 스톨 재킥 + PWM 틱 정렬** (run63/71 결함 현존, 0.19=틱327은 출발임계 틱328의 -1틱) | §2-A4, §2-C1 |

---

## 1. 인코스 상태 전환 조건표 (전이 / 유지 / 해제)

> 표기: 소스 기본값, 괄호(→운용값)=aux.sh 주입값. launch 는 course/race_dir/skip_missions 만
> 전달하므로 **나머지는 전부 declare_parameter 기본값이 실효값**이다 (aux.sh 미주입 시).

### WAIT_GREEN → DRIVE (출발)
- **전이**: YOLO `green_light` conf ≥ `conf_threshold`(0.5) AND 면적 ≥ `light_min_area`(**0.0=off**)
  가 `green_frames`(10)틱 연속. 30Hz 틱 기준이므로 **YOLO 6Hz 실질 2추론** — 미달 시 0 리셋.
- **출발 순간**: 킥 `start_kick_throttle`(0.23)+adj 를 `start_kick_s`(0.4s),
  `max_throttle_delta`(0.02/틱) 램프 → 킥 레벨 체류 실질 ~0.1s (§2-C2).
- **타임아웃 없음** — 초록 미인식 시 무한 대기 (규정: 준비 5분 초과 = 주행 실패).

### DRIVE 공통 (조향/스로틀)
- 조향 = PID(kp 0.6 / ki 0 / kd 0.15) 대상 = `lane.offset`(+옵션 편향들);
  `steer = steer_center(0.26) + corr × steer_scale(-1.0)`.
- 소실 시: `_steer_hold` EMA 유지(±0.30 클램프), 3s 시정수로 center 수렴.
  병합 브리지 창에서는 EMA 대신 `steer_center + merge_blind_bias`(0.10 → **운용 0**).
- 스로틀 = `drive_throttle`(0.19) × (1 − `curve_slow`(0.5)×|curv|), 하한 `slow_throttle`(0.165);
  `yellow_ratio ≥ yellow_slow_ratio`(0.03) 이면 상한 `yellow_drive_throttle`(0.165);
  병합 브리지/빨간노면 중 상한 slow.

### Y래치 (perception, DRIVE[W]→DRIVE[Y])
- **진입**: `yr ≥ follow_yellow_ratio`(0.03 → 운용 0.02) 1프레임.
- **유지**: RA 중 강제 유지. 해제 카운터는 조건 미충족 시 0 리셋.
- **해제**: [yr < ratio × `follow_yellow_exit_yellow_frac`(0.5 → 운용 3.0, 단 **RA 전 구간은
  0.75 캡 박제됨** — 4b1ceed) AND 흰픽셀 > 노랑픽셀] 연속 `follow_yellow_exit_frames`(10f,
  SW 탈출창 내 4f), **또는** 노랑 소실 단독 `follow_yellow_blind_release_frames`(12f).
- 해제 순간: `w_align` 창(60f) 무장 — 흰 점선 제거 + 노란 꼬리 유지 + 헤딩 정렬(gain 0.4).
- decision 측 별도 Y래치(런당 1회): `y_latch_ratio`(0.02) × `y_latch_frames`(10틱) →
  `throttle_adj` = clamp(0.06 × (race_t/4.6s − 1), ±0.015) **원샷, 해제 없음** (§2-C3).

### DRIVE → ROUNDABOUT (A 정지선 = RA 진입)
- **전이** (AND): course=in / NOT roundabout_done / `race_t ≥ ra_min_drive_s`(7.5s) /
  `yr ≥ yellow_enter_ratio`(0.06) / `yellow_crossline` 지속 `enter_sustain_s`(**0.2 → 운용 0.12**;
  기본값이면 run89 고속밴드 미진입 재발) / 곡률게이트 `enter_max_curvature`(0=off).
- **진입 순간**: `_speed_scale` = clamp(race_t/`ra_ref_drive_s`(20.5s), 0.6~1.3) 확정;
  진입 락온 `entry_lock_release_s`(3.0s) + `entry_steer_bias`(-0.15);
  블랭크 `gate_blank_s`(**7.0 → 운용 3.5**, 무스케일) 장전; 게이트 카운터 전체 리셋.
- ⚠️ 래치 위치는 속도 의존 (정상속: A 직전 / 최저속 run79: B 분기 페인트) — B 크로싱은
  직교 전폭 실선이라 형태 구분 불가 확정. 군집 카운트가 흡수하는 게 현행 설계.

### ROUNDABOUT 유지 (군집 카운트 v2)
- 링 유지: `circle_steer_bias`(0=off), 소실 프레임 `ra_blind_bias`(-0.15), yaw_proxy 적분.
- **군집 규율**: crossline ON 간격 < `gate_cluster_gap_s`(1.8s) 병합; 카운트 성립 =
  [누적 ON ≥ `gate_cluster_on_s`(**0.25 → 운용 0.12**; 기본값이면 run93 카운트 실패 재발)]
  AND [블랭크 밖 출생 (void 아님)] AND [OFF ≥ `gate_rearm_s`(0.5s) 재무장]
  AND [yaw ≥ `yaw_gate_min`(1.0)×scale] → 카운트 + 쿨다운 `crossline_cooldown_s`(2.0s).
- STOPLINE 분류기(`stopline_mode` **0 → 운용 1**): RA+코리도 락 프레임에서 "관통(cov)+정면(각도)"
  만 인정, B 스침은 원리 탈락. 분류기 전제 미성립 프레임은 crossline 자체 억제 (run102 방어).

### ROUNDABOUT → DRIVE 탈출 (4경로, 전부 출구측 락온 + merge_bridge_s 6.0s 무장)
1. **직발화** `ra_direct_fire`(**0 → 운용 1**): 카운트 성립 프레임 즉시. min_loop 미적용.
   ⚠️ stopline_mode=1 필수 전제 — **코드 인터록 없음, 주석 경고뿐** (§2-A2).
2. **주경로**: `circle_t ≥ min_loop_time_s`(17s)×scale AND count ≥ `roundabout_exit_gates`(2).
3. **백업 표결**: {yaw ≥ `yaw_lap_threshold`(**5.0 — 봉인값 7.0 미박제**, §2-B4)×scale,
   circle_t ≥ `nominal_loop_time_s`(15s)×scale, cross 재등장} 중 `lap_votes_needed`(2)표.
4. **강제**: circle_t ≥ `max_loop_time_s`(**30 → 운용 75**; 기본값이면 2랩 자가복구 붕괴).
- 탈출 공통: `exit_lock_release_s`(2.5s) 동안 소실 프레임 `exit_steer_bias`(+0.18);
  브리지 창 스로틀 slow 캡; perception `sw_exit_frames`(**0 → 운용 40**) 코리도.

### 병합 → 흰 순항 → 빨간노면 → OBSTACLE_STOP
- 흰 복귀는 Y래치 해제 조건(위) — 병합 처방(exit_frac 3.0 + sw_exit 40f)은 **실차 미검증**.
- `in_red_zone` = `red_ratio ≥ red_slow_ratio`(0.05) AND (in 코스면 roundabout_done 필수)
  → 스로틀 slow 캡.
- **OBSTACLE_STOP 진입** (DRIVE 전용, 1메시지 즉발): `aruco.detected` AND
  (in_red_zone **OR** `area_ratio ≥ marker_area_trigger`(0.02) — ⚠️ area 분기는
  roundabout_done 게이트 없음, §2-B2).
- **해제**: detected=False 연속 `marker_clear_frames`(8틱 = **0.27s**; 주석 의도 0.8s와 3배
  불일치, §2-B3) → `obstacle_done=True` + DRIVE 복귀. 재검출 1틱이면 카운터 리셋.

### DRIVE → FINISH → DONE
- **무장**: `obstacle_done` OR `race_t ≥ finish_min_drive_s`(**60.0s** — run73 완주 70.4s보다
  짧아 코스 중반에 예비 무장이 열림, §2-B1).
- **전이**: `red_light` 연속 `red_frames`(**3틱** = 0.1s < YOLO 1추론 지속 5틱 —
  **오추론 1건으로 발화**) → FINISH(즉시 중립, **복귀 전이 없음**) → 재차 3틱 → DONE.

---

## 2. 확정 위험점 및 방어로직 제안 (적대 검증 완료)

### A급 — 구조적, 완주 실패 직결

#### A1. 운용 방어 스택 전체가 aux.sh 주입 의존 — 소스/런치에 미박제
- 위치: [auto_race.launch.py:119-120](D-Racer-Kit/src/decision/launch/auto_race.launch.py#L119)
- aux.sh 는 보드 /tmp 램디스크 (재부팅 소실). launch 전달값은 course/race_dir/skip_missions 뿐.
- 미주입 시 조용히 기본값 복귀 → **run89(0.2s 미진입)·92/93(미발화)·94(병합 이탈)·74/75(표결 오발)
  기전이 그대로 부활**하고 기동 로그에 실효값이 없어 무증상.
- **부분 주입이 무주입보다 위험**: ra_direct_fire=1 + stopline_mode=0 조합은 B 오발 (시뮬 실증).
- **방어**:
  1. 운용 확정값을 `declare_parameter` 기본값으로 박제 (CLAUDE.md 의 기존 원칙 그대로):
     decision — `ra_direct_fire 1`, `gate_cluster_on_s 0.12`, `enter_sustain_s 0.12`,
     `gate_blank_s 3.5`, `merge_blind_bias 0.0`, `yaw_lap_threshold 7.0`, `max_loop_time_s 75.0`;
     lane — `stopline_mode 1`, `follow_yellow_exit_yellow_frac 3.0`, `sw_exit_frames 40`,
     `follow_yellow_ratio 0.02`.
  2. 기동 시 **실효값 요약 1줄 로그** (decision/lane 각각) — 당일 검증 가능하게.
  3. aux.sh 자체도 리포에 커밋 (보드 스크립트 유실 대비).

#### A2. ra_direct_fire ↔ stopline_mode 인터록 부재
- 위치: [state_machine.py:539](D-Racer-Kit/src/decision/decision/state_machine.py#L539) (주석 경고만)
- 직발화는 "카운트되는 첫 군집 = A 재도달" 전제 — B 누출 스트림(stopline_mode 0)에선 B 가
  첫 카운트가 되어 **B 로 탈출 = 링 이탈 + 미션 실패**.
- **방어**: 두 노드에 걸친 값이라 완전 잠금은 어려움 → ① decision 에 `stopline_mode` 를
  미러 파라미터로 선언해 launch 가 두 노드에 같은 값 주입 ② decision 기동 시
  `ra_direct_fire=1 && stopline_mode!=1` 이면 **직발화 자동 비활성 + ERROR 로그**
  ③ lane_node 가 stopline_mode 를 `/perception/lane` 헤더나 별도 토픽으로 1회 공표 →
  decision 이 대조 (완전판).

#### A3. 스테일 페일세이프 전무 (decision/lane 사망 = 폭주)
- 위치: [control_node.py:126-140](D-Racer-Kit/src/control/control/control_node.py#L126),
  [decision_node.py:434-441](D-Racer-Kit/src/decision/decision/decision_node.py#L434)
- decision 죽으면 control 이 **마지막 조향·스로틀을 20Hz 로 영구 재적용**. lane 죽으면
  decision 이 동결 offset 으로 계속 조향. decision/lane 은 respawn 대상도 아님
  (respawn 은 battery/camera/control 만).
- **방어** (검증자 부작용 확인 완료 — 정상 주행 중 30Hz 발행이라 0.5s 갭 불가, 발화 체인 무영향):
  1. control_node: `/control` 마지막 수신 후 `control_timeout`(0.5s) 초과 시 스로틀 중립
     (**use_joystick_control=False 일 때만** — 수동 모드 오동작 방지).
  2. decision_node: `lane.header.stamp` 나이 > 0.5s 이면 정지 명령 (조향은 center 수렴).
  3. launch: decision/lane 에도 `respawn=True` 추가.

#### A4. 물리 스톨 재출발 불가 (run63/71 결함 현존)
- 위치: [decision_node.py:564](D-Racer-Kit/src/decision/decision/decision_node.py#L564)
- 킥 트리거 = `_prev_throttle ≤ 1e-3` (명령 0→상승 에지 전용). 명령 0.165 유지 중 물리
  정지(운동임계 미달·노면 저항)는 감지 불가 → **링/커브 한복판 영구 정지**. DRIVE 하한이
  slow(0.165)라 출력이 0에 안 닿아 킥 영구 굶음.
- **방어** (IMU/엔코더 없음 전제):
  1. **비전 스톨 감지**: lane_node 에서 연속 프레임 마스크 차분(mean|Δ|)이 ε 미만 N프레임
     (예: 1.5s) AND decision 스로틀 명령 > 0 → `/perception/stall` 발행 → decision 재킥
     (킥 1회 + 쿨다운 3s, 최대 3회 후 포기 로그).
  2. 재킥 시 `_kick_left` 재장전만으로는 부족 — **감산을 출력이 킥값 90% 도달 후 시작**
     (현행은 램프가 킥 창을 잠식, §2-C2와 동일 처방).
  3. 배터리 전압 회복 곡선(무부하 복귀) 감지는 battery_node 상시 가동 전제라 보류 (§2-B6).

#### A5. ESC 소프트 LVC(무부하 7.5V) 대응 부재
- 위치: [decision_node.py:96-101](D-Racer-Kit/src/decision/decision/decision_node.py#L96)
- `undervolt_slow_v/stop_v` 기본 0=off, launch `use_battery` 기본 false → battery_node 자체
  미기동 → 가드 이중 비활성. run96/97 (스로틀 무반응) 재발 가능.
- ⚠️ 검증자 경고: battery_node 재기동은 **07-10 실증된 i2c-3 경합(INA219 10Hz vs PCA9685
  20Hz) 리스크 재도입** — 코드 방어보다 절차 방어가 안전.
- **방어**: ① 런 전 무부하 전압 측정 절차 고정 (하한 7.5V, 안전선 7.7V — RUN_LOG 확정치,
  런 사이 충전/교체 판단) ② WAIT_GREEN 중 1회성 전압 로그 (battery_node 를 출발 전에만
  띄웠다 내리는 프리플라이트 스크립트) ③ 주행 중 상시 가드는 버스 경합 재검증 후.

### B급 — 미션 실패 경로

#### B1. FINISH 오발 3중 결합 (예비 무장 60s + red_frames 3틱 + light_min_area 0)
- 위치: [decision_node.py:109,115,116](D-Racer-Kit/src/decision/decision/decision_node.py#L109)
- run73 완주 70.4s → **60s 시점 = 병합~흰 순항 구간에서 예비 무장이 열림**. red_frames 3틱은
  YOLO 1추론(5틱 지속)으로 관통 — 스테일 dets 래치라 추론 정지 시에도 유지. 빨간 노면
  구간·빨간 옷·트랙 띠의 작은 오검출 박스(light_min_area 0 = 무필터) 1건이면
  **코스 중반 FINISH 영구 정지** (복귀 전이 없음).
- **방어**: ① `finish_min_drive_s` 120s (규정 6분 제한 내 폴백 여전히 유효)
  ② `red_frames` 를 "독립 추론 메시지 기준"으로 — dets 메시지 갱신 시에만 카운트 증가
  (구현: Detections 수신 시각 비교), obstacle_done 전엔 2배 요구
  ③ `light_min_area` 당일 실측 박제 (§4) — 단 red 는 원거리 인식 지연 부작용 있으니
  초록/빨강 거리 both 실측 ④ FINISH 에서 red 소실 5s 시 DRIVE 복귀 (완전판, 선택).

#### B2. 마커 area 분기가 roundabout_done 미게이트 + obstacle_done 조기 세팅 체인
- 위치: [state_machine.py:347-349](D-Racer-Kit/src/decision/decision/state_machine.py#L347)
- RA 전 DRIVE 에서 큰 마커(면적 ≥0.02)가 보이면 OBSTACLE_STOP 진입 → 클리어 →
  `obstacle_done=True` → **빨간불 무장 조기 개방** (B1 과 결합 시 치명).
- **방어**: in 코스에서 area 분기에도 `roundabout_done` 요구 (in_red_zone 분기와 동일 게이트).
  코드 주석("장애물은 항상 RA 뒤")이 이미 근거를 문서화하고 있음 — 게이트만 누락.

#### B3. marker_clear_frames=8 단위 착오 (의도 0.8s vs 실제 0.27s)
- 위치: [decision_node.py:123](D-Racer-Kit/src/decision/decision/decision_node.py#L123)
- 30Hz 틱 기준 8틱=0.27s = aruco 12Hz 의 3프레임. run80 검증 흡수창의 1/3. 잔존 위험은
  반복 깜빡임 크리프 + 발진 후 근접 마커가 화각 아래로 빠져 재검출 불가로 지나침.
- **방어**: `marker_clear_s`(0.8) dt 누적으로 전환 (enter_acc 패턴 재사용, state_machine.py:392).

#### B4. 백업 표결 봉인값(yaw 7.0)·강제탈출(75s) 미박제 — run83/84/86 재발 경로
- 위치: [decision_node.py:183,191](D-Racer-Kit/src/decision/decision/decision_node.py#L183)
- 기본 yaw_lap_threshold 5.0 → 무앵커 백업 발화(count=0 병합 블라인드 이탈, run83/84) 부활;
  기본 max_loop 30s → 2랩 자가복구 전제 붕괴. → **A1 박제 목록에 포함** (yaw 7.0 / max_loop 75).

#### B5. YOLO/카메라 무증상 고장 (빈 검출 영구 발행 / GStreamer 웨지)
- 위치: [yolo_ncnn_node.py:154-159](D-Racer-Kit/src/inference/inference/yolo_ncnn_node.py#L154),
  camera_node 캡처 루프
- 모델 로드 실패 → 빈 Detections 영구 발행 → WAIT_GREEN 무한 대기 (준비 5분 초과 = 주행
  실패). `ex.extract` 반환값 미확인 → 추론 고장도 빈 발행으로 은폐. 카메라 USB 단선/드라이버
  행은 노드 생존+발행 중단이라 respawn 무력.
- **방어**: ① 모델 로드 실패 시 `raise`(노드 사망 → 눈에 보임; yolo 에 respawn 추가하면
  자동 재시도) ② camera_node: 캡처 실패/프레임 나이 > 1s 연속 N회 → 노드 자살 → respawn
  ③ decision: WAIT_GREEN 에서 dets 메시지 5s 무수신 시 WARN 반복 (조작자 가시성).

#### B6. NCNN 클래스 순서 하드코딩 — red/green 스왑 무증상 위험
- 위치: [yolo_ncnn_node.py:104](D-Racer-Kit/src/inference/inference/yolo_ncnn_node.py#L104)
- `['red_light','green_light','left_sign','right_sign']` 이 실제 export 순서와 다르면
  **빨간불에 출발**. 모델은 보드 측이라 코드로 검증 불가.
- **방어**: 당일 절차 — 초록 실물 제시 → `/inference/detections` 라벨 육안 확인 (§4).

#### B7. ArUco 소스 기본값 불일치 (4X4/false vs 실물 6X6/inverted)
- 위치: [aruco_node.py:25-26](D-Racer-Kit/src/perception/perception/aruco_node.py#L25)
- launch 는 6X6+inverted 로 맞지만 (run78~80 실물 검증 완료), **`ros2 run` 개별 기동 시
  기본값 함정** — run77/78 에서 실제 재현된 이력. 미검출 → obstacle_done=False →
  빨간불 무장은 60s 백업뿐.
- **방어**: 소스 기본값을 `DICT_6X6_50` + `inverted=True` 로 박제 (launch 와 일치).

### C급 — 한계 조건 / 품질

#### C1. PWM 1틱=0.0098 양자화 — 명목/실효 괴리 (재검산 완료)
- 0.19→틱327(실효 0.1934, **출발임계 틱328 의 -1틱**), 0.165→틱324, 0.23→틱331.
  0.19+0.005=0.195 도 같은 틱327 = 하드웨어 무변화. 낡은 주석(0.19 를 '1602us'로 표기)이
  함정을 가림.
- **방어**: ① 기동 시 주요 스로틀 파라미터의 `명목값→us→틱→실효값` 환산표 로그
  ② 주석 갱신 ③ 튜닝은 틱 경계 인지 하에 (1틱 = ±0.0098 단위로만 의미).

#### C2. 출발 킥이 rate limit 램프에 잠식 (킥 레벨 체류 ~0.1s)
- 30Hz 에서 램프(0.02/틱)가 킥값 0.23 도달에 12프레임 = 킥 창(0.4s=12프레임) 전부 소모 →
  0.23 체류 1프레임, ≥0.20 체류 3프레임. 배터리 열화로 임계 상승 시 미출발.
- **방어**: `_kick_left` 감산을 **출력이 킥값 90% 도달한 프레임부터 시작** (킥 체류를
  설계값 0.4s 로 복원). 부작용 없음 — 출발 지점은 직선.

#### C3. throttle_adj 음보정이 slow 경로를 틱323으로 (run63/71 스톨 구간)
- 과속 밴드(Y래치 ≤4.1s)에서 adj ≤ -0.0063 → slow 0.165(틱324) → 틱323. run63/71 이
  실제 스톨한 구간. 음보정은 과속 팩(실제 임계 낮음)에서만 발화해 상쇄되지만 보장 없음.
- **방어**: adj 가산을 drive_throttle 경로에만 적용, slow/yellow 캡 경로는 제외
  (또는 slow 경로 하한을 틱324 실효값 0.16406 으로 클램프).

#### C4. I2C 웨지 무방비 (예외 삼킴 → respawn 도달 불가, e-stop 도 동일 경로)
- 위치: [control_node.py:142-155](D-Racer-Kit/src/control/control/control_node.py#L142)
- PCA9685 는 마지막 레지스터값으로 PWM 을 계속 출력 → 버스 웨지 시 **마지막 성공 PWM 으로
  폭주**, e-stop 도 I2C 쓰기라 무력 (run90/91 웨지 이력).
- **방어**: `_io_err_streak ≥ 20`(≈1s) 시 ① SMBus close→재오픈 1회 시도 ② 그래도 실패면
  노드 종료 (respawn=True 가 이미 있어 자동 재시작 + ESC 아밍 재수행). 성공 시 streak 리셋이
  이미 있어(148행) 순간 오류 오발 없음.

#### C5. control respawn 후 스텝 인가 (아밍 해제 순간 0.19 즉시)
- 위치: [control_node.py:131-140](D-Racer-Kit/src/control/control/control_node.py#L131)
- respawn 시 3s 아밍(중립) 후 decision 현재 명령이 **스텝 인가** — 정지 차량이면 킥 없이
  임계 근처 값이라 미출발 가능 (decision 은 control 재시작을 모름).
- **방어**: 아밍 해제 후 자체 램프(0→목표, 0.02/50ms)로 재시작. A4 재킥이 있으면 이중 안전.

#### C6. stopline_cov_min 기본값 불일치 (lane_node 0.35 vs detector·매뉴얼 0.25)
- 위치: [lane_node.py:47](D-Racer-Kit/src/perception/perception/lane_node.py#L47)
- 캘리 근거(f59 실측 0.28 포용)는 0.25 — lane_node 0.35 가 실효라 **진짜 A 재도달(cov
  0.28대)을 기각할 수 있음**. → 0.25 로 통일 박제.

#### C7. 직발화 무랩 탈출 리스크 (B 조기 래치 시) — 사용자 인지 결정 사항
- min_loop 미적용은 run93 후 사용자 결정. 단 **최저속 밴드에서 래치가 B 페인트로 밀리면**
  (run79 실측) 첫 카운트 = A "1차" 도달 → 무랩 탈출 = 미션 실패.
- **제안** (결정 존중, 옵션): `ra_direct_fire_yaw_min`(기본 0=off) — 직발화에만 yaw 바닥.
  B래치→A1차 yaw 와 A래치→A재도달 yaw(실측 3.10~3.37) 사이 값으로 텔레메트리 캘리 후 결정.

### 미검증 방어 (실차 확인 대기 — 코드는 존재)
1. 병합 처방: `exit_yellow_frac 3.0` + `sw_exit_frames 40` (run95 대응) — **실차 0회**
2. `yw_premix` (run103/105/108 리플레이 3회 보정) — 실차 A/B 없음
3. RA 개구부 유출 차단 (상시-on 하드코딩, 킬스위치 없음) — 실차 1회
4. 탈출 락 slow 캡 (run61 이후 정상 발화 런에서 미확인)

---

## 3. 구현 우선순위 로드맵

**즉시 (다음 리빌드에 — 회귀 위험 최소, 전부 파라미터/게이트 수준)**
1. A1 박제 + 실효값 기동 로그 (§2-A1 목록 그대로)
2. A2 인터록 (직발화 자동 비활성 + ERROR)
3. B2 area 분기 roundabout_done 게이트 (1줄)
4. B7 ArUco 기본값 6X6/inverted (2줄)
5. C6 stopline_cov_min 0.25 통일 (1줄)
6. B1 ①② (finish_min_drive_s 120 + red_frames 독립추론화)

**레이스 전 (실차 1회 검증 필요)**
7. A3 스테일 페일세이프 3종
8. C2 킥 램프 보정 + C3 adj slow 제외
9. B3 marker_clear_s 시간 기반
10. B5 YOLO raise + 카메라 워치독
11. C4 I2C 재오픈/자살

**여유 시 (신규 로직 — 리플레이 검증 먼저)**
12. A4 비전 스톨 감지 + 재킥
13. C7 직발화 yaw 바닥 (텔레메트리 캘리 후)
14. B1 ④ FINISH 복귀 전이

---

## 4. 대회 당일 체크리스트 (재캘리 필수)

| 항목 | 방법 | 관련 |
|---|---|---|
| 배터리 무부하 전압 ≥7.7V | 런 전 매회 측정, 7.5V 미만 교체/충전 | A5, run96/97 |
| `light_min_area` 실측 | 실제 신호등 거리에서 초록/빨강 bbox 면적 로그 → 값 박제 | B1, run91 |
| YOLO 라벨 스왑 확인 | 초록 실물 제시 → detections 라벨 육안 확인 | B6 |
| ArUco 실물 확인 | 대회 마커로 detected/id 확인 (사전/inverted) | B7, run78 |
| 빨간 노면 `rr ≥ 0.05` 정적 확인 | 빨간 구간 앞에 차 두고 red_ratio 로그 | 코스정의 |
| BEV 기둥 검증 | 직선 흰선 2개 기둥 slope ~0, 폭 192±5px | 마운트 변경 시 |
| race_dir 확인 | 배정 트랙 방향 → launch 인자 (좌우 파생 전부 자동) | 규정 |
| 실효 파라미터 로그 대조 | 기동 로그의 A1 요약줄 = 운용 확정값인지 | A1 |
| 준비 5분 시간 관리 | WAIT_GREEN 진입 후 초록 미인식 시 수동 개입 판단 기준 사전 합의 | B5 |

---

## 5. 차선 파이프라인 (lane_detector 심층) — 별도 감사 진행 중

lane_detector.py 1603줄 전체(SW 코리도 무장/락/해제 경계, HSV·BEV 하드코딩의 대회장 노면
가정, 프레임수 창의 FPS 의존, 런 간 상태 잔존)에 대한 심층 감사가 진행 중이며,
완료 시 이 섹션에 병합한다.
