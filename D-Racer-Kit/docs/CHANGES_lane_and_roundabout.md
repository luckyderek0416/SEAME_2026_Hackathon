# 라인추종 · 회전교차로 — 원본 vs 변경 상세 메모

> 공식 클론(main) 코드에서 우리가 어떻게 바꿨는지 기록. 트랙 튜닝·인수인계용.
> 관련 파일: `perception/lane_detector.py`, `perception/lane_node.py`,
> `decision/state_machine.py`, `decision/decision_node.py`, `perception_msgs/LaneState.msg`

---

## 1. 라인추종 (lane_detector.py)

### 원본 방식
```
① ROI 자르기 (화면 아래쪽)
② 흑백 변환 → 밝기 임계값(bright_thresh=160)으로 이진화   ← 흰 선만
③ ROI 전체를 좌/우 절반으로 나눔
④ 각 절반의 흰 픽셀 "평균 x" = 왼선/오른선 위치
⑤ 두 선 중점 = 차선중심 → offset
   - 한 선만 보이면: 고정 비율(single_line_offset) 만큼 떨어진 곳
```
- **단일 밴드**(코앞 한 구간만) + **평균 x** 방식
- 문제점:
  - 커브를 **미리 못 봄** → 늦게 꺾여 이탈 위험 (S자·회전교차로·코너)
  - 평균 x는 **노이즈 한 점에도 흔들림** (대리석 글레어, 점선)
  - 흑백 밝기라 **흰선/노란선 구분 불가**, 노란 회전교차로 차선에 약함
  - 한 선만 보일 때 고정 비율 추정 → 커브에서 부정확

### 변경 방식 (우리)
| # | 보강 | 내용 | 효과 |
|---|---|---|---|
| 1 | **HSV 색 검출** | 흑백 대신 HSV로 **흰색+노란색** 마스크 결합 | 노란 회전교차로 차선 검출, 색 구분 |
| 2 | **멀티밴드 룩어헤드** | ROI를 가로 N밴드(가까움~멈)로 나눠 각 밴드 중심 계산, 가까운 밴드 가중 | **커브 미리 봄** → 일찍 꺾음 |
| 3 | **곡률 추정** | (먼 밴드 중심 − 가까운 밴드 중심) 정규화 → `curvature` | decision이 **곡률 감속** |
| 4 | **차선폭 기억** | 두 선 보일 때 폭을 EMA로 기억 → 한 선만 보이면 그 폭 절반만큼 | 커브에서 중심 정확 |
| 5 | **노이즈 정리** | MORPH_OPEN(erode→dilate) | 글레어·점선 조각 제거 |
| 6 | **offset 평활** | EMA(smooth_alpha) | 프레임간 떨림 ↓ |

### 새 파라미터 (lane_node)
`num_bands`(4), `morph_kernel`(3), `width_ema`(0.1), `smooth_alpha`(0.5),
`mask_mode`(hsv), `use_white/use_yellow`, `white_hsv_lo/hi`, `yellow_hsv_lo/hi`
+ decision: `curve_slow`(0.5) — DRIVE에서 `throttle = drive_throttle × (1 − curve_slow×|curvature|)`

---

## 2. 회전교차로 (state_machine.py)

### 원본 방식
```
course=='in' 이면 출발 직후 바로 ROUNDABOUT 상태로 진입
ROUNDABOUT: 저속으로 lane-follow 하며 roundabout_seconds(기본 8초) 경과하면 탈출
```
- **한 바퀴 = 시간(8초)** 로 판정 (`_roundabout_done` = 시간 적산)
- 코드 주석도 "IMU 있으면 yaw 적산으로 바꾸라"고 인정
- 문제점:
  - **시간 기반이라 속도 변동(배터리)에 취약** → 한 바퀴 못 채우거나 과회전
  - 출발 즉시 ROUNDABOUT 진입 → **접근 직선구간까지 8초에 포함** = 튜닝 불가
  - **원을 강제로 유지하는 로직 없음** → 분기점에서 1바퀴 전 탈출 가능
  - **좌/우 출발 방향(정/역방향 트랙) 대응 없음**

### 변경 방식 (우리) — 마커·IMU 없이 3중 투표
```
DRIVE ─(노란색 + 급커브 지속)→ ROUNDABOUT ─(한 바퀴 확정)→ DRIVE(완료)
```
**진입 감지**: 곡률 크고 + 노란색 보임(둘 다) → 진입 (흰 외곽 코너 오진입 차단)

**회전 중**: lane PID + turn_direction 바이어스로 원에 붙잡음(조기 탈출 방지)

**한 바퀴 판정 = 3개 독립 신호 투표**:
| 신호 | 방식 |
|---|---|
| ① junction | 분기점 **점선 패턴**(세로 on/off 전환) 재출현 카운트 |
| ② 조향각 적분 | `Σ 회전방향 조향편차 × dt` (IMU 대체 yaw 추정) |
| ③ 시간 | 실측 한 바퀴 시간(nominal_loop_time) 도달 |

**탈출 규칙**:
```
if 시간 ≥ min_loop_time (하드 플로어, 그 전엔 절대 안 나감):
    votes = ①+②+③
    if votes ≥ lap_votes_needed(2):  → 탈출
if 시간 ≥ max_loop_time:  → 강제 탈출(안전망)
```
- 설계 철학: 룰이 "1회 **이상**"이라 **과회전은 합법, 미달은 실패** → "늦게 나가도 일찍은 안 나간다"로 편향
- **race_dir**(left/right): turn_direction + junction_side 일괄 설정 (정/역방향 트랙 대응)
- **In/Out 분기**: 노란색 쪽으로 자동 편향(방향 무관)
- **skip_missions**: 라인추종만 테스트하는 모드

### 새 파라미터 (decision)
`turn_direction`/`race_dir`, `circle_steer_bias`, `target_loops`,
`min_loop_time_s`/`max_loop_time_s`, `nominal_loop_time_s`, `yaw_lap_threshold`,
`lap_votes_needed`, `junction_cooldown_s`, `enter_curvature`/`enter_sustain_s`,
`use_yellow_entry`/`yellow_enter_ratio`, `branch_bias`/`branch_yellow_min`

### junction 점선 검출 (lane_detector)
원본: 없음 → 우리: 이 트랙은 **회전교차로 입출구만 노란 점선**이라, 세로 방향
on/off 전환 횟수(`junction_dash_transitions`)로 점선=junction을 검출.
(`junction_side`, `junction_min_row_pixels`, `junction_gap_rows`)

---

## 3. 트랙에서 실측·튜닝해야 할 값
| 값 | 잡는 법 |
|---|---|
| `yellow_hsv_lo/hi`, `white_hsv_lo/hi` | `tools/hsv_sampler.py`로 실측 |
| `kp`/`kd` | 라인추종 튜닝 (Kp↑ 따라갈 때까지→Kd로 출렁임 잡기) |
| `drive_throttle`/`slow_throttle` | ESC 데드밴드 넘는 최소값 이상 |
| `nominal_loop_time_s` | 한 바퀴 시간 측정 |
| `yaw_lap_threshold` | 한 바퀴 끝 `yaw_proxy` 로그값 |
| `junction_side`/`junction_dash_transitions` | 디버그 영상으로 점선 검출 확인 |
| `race_dir` | 배정된 트랙 방향(left/right) |

> 실측값보다 살짝 크게(×1.1) 잡으면 "확실히 1바퀴" 안전 편향.
