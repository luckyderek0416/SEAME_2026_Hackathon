# Look-Ahead Steering (부분 Pure Pursuit)

기존 offset 기반 PID 조향은 유지하고, **PID에 들어가는 offset 자체를**
가까운 밴드 + 앞쪽(look-ahead) 밴드의 블렌드로 바꿔 커브를 미리 보고 꺾게 한다.
완전한 Pure Pursuit 교체가 아니다. **look-ahead 블렌딩(`use_lookahead_control`)은
현재 기본 ON**, decision 출력 성형(`max_steering_delta`/`steer_slow`)은 기본 OFF(0)다.
끄려면 `ros2 param set /lane_node use_lookahead_control false`. LaneState 메시지·토픽 변경 없음.

## 구조

```
[lane_detector]  bands(near..far)
   ├─ 기존(OFF): lane_center = 전체 밴드 가중평균  ──► offset ─► LaneState
   └─ LA ON:     lane_center = near_weight×최근접밴드 + lookahead_weight×원거리밴드
                 (adaptive: |curvature| ≥ thresh 이면 lookahead 비중 ↑)

[decision_node]  steer = steer_center + PID(offset+bias)×steer_scale  (기존 그대로)
   ├─ max_steering_delta > 0: 틱당 조향 변화 제한 (튐 방지)
   └─ steer_slow > 0:        |조향| 비례 감속 (slow_throttle 바닥)
```

- lane lost 시 fallback 은 기존 그대로: offset EMA 유지 + PID target 0 → 조향 급변 없음.
- roundabout 상태머신·circle_steer_bias·랩 카운트 로직은 손대지 않음.
  (rate limit 은 최종 출력에만 적용)

## 파라미터

### /lane_node (블렌딩)
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `use_lookahead_control` | `true` | look-ahead 블렌딩 on/off (기본 ON) |
| `near_weight` | `0.7` | 최근접 밴드 비중 |
| `lookahead_weight` | `0.3` | 원거리 밴드 비중 |
| `lookahead_band_index` | `-1` | 사용할 밴드 (-1=검출된 가장 먼 밴드) |
| `adaptive_lookahead` | `false` | 급커브에서 lookahead 비중 자동 증가 |
| `curve_lookahead_weight` | `0.4` | 급커브 시 lookahead 비중 |
| `curve_lookahead_thresh` | `0.25` | 급커브 판정 \|curvature\| 문턱 |

### /decision_node (출력 성형)
| 파라미터 | 기본 | 설명 |
|---|---|---|
| `max_steering_delta` | `0.0` (off) | 틱(1/30s)당 조향 변화 상한. 시작값 0.08~0.15 권장 |
| `steer_slow` | `0.0` (off) | \|steer−center\| 비례 감속 게인. 시작값 0.3~0.5 권장 |

## 실행 예 (라이브 튜닝 — 주행 중 다른 터미널에서)

```bash
# 기본 블렌딩 켜기
ros2 param set /lane_node use_lookahead_control true

# 커브 보정 실험 (더 미리 꺾기)
ros2 param set /lane_node near_weight 0.6
ros2 param set /lane_node lookahead_weight 0.4
ros2 param set /lane_node adaptive_lookahead true

# 조향 튐 방지 + 코너 감속
ros2 param set /decision_node max_steering_delta 0.1
ros2 param set /decision_node steer_slow 0.4
```

노드 단독 실행 예:
```bash
ros2 run perception lane_node --ros-args -p use_lookahead_control:=true \
  -p near_weight:=0.7 -p lookahead_weight:=0.3
ros2 run decision decision_node --ros-args -p skip_missions:=true \
  -p max_steering_delta:=0.1
```

## 디버그 확인 (모니터 HSV Lane 패널)

- 상태 텍스트: `BEV … GUIDED … LA ON/OFF`
- LA ON 시: **파란 세로선(하단 1/3)** = near 밴드 중심, **주황 세로선(상단 1/3)** = look-ahead 중심,
  빨간 선 = 최종 블렌드 중심. 셋의 간격으로 블렌딩 정도를 눈으로 확인.
- decision 디버그 로그(`--log-level debug`)에 `steer/thr/off/curv` 출력.

## 추천 초기 튜닝값 & 순서

1. `use_lookahead_control=true`, 0.7/0.3 으로 직선 주행 → 진동 없으면 통과
2. S커브에서 `0.6/0.4` 또는 `adaptive_lookahead=true` 비교
3. 조향이 프레임 간 튀면 `max_steering_delta 0.1` (너무 작으면 조향 반응이 느려져 커브 이탈 — 0.05 미만 금지)
4. 코너 과속 이탈 시 `steer_slow 0.4`
5. 확정값은 lane_node/decision_node 기본값에 반영 후 리빌드

## 주의

- `lookahead_weight` 를 0.5 이상으로 올리면 직선에서 원거리 노이즈에 민감해져 진동 위험.
- BEV·guided 와 독립적으로 동작하지만, guided ON 이면 원거리 밴드가 더 안정적이라 궁합이 좋다.
- 리빌드: `colcon build --packages-select perception decision` (symlink-install 아님)
