# Lane Detector: Bird-Eye View + Guided Band Search

`lane_node`(perception)에 옵션 2개가 추가됐다. **현재 둘 다 기본 ON** (운영 기본값).
끄려면 `ros2 param set /lane_node use_birdeye false` 처럼 라이브로 내리면 기존
파이프라인과 완전히 동일하게 동작한다.

> ⚠️ **BEV 주의:** `birdeye_src_ratio` 기본값은 실제 트랙/카메라에 맞춘 값이 아니라
> 일반적인 추정값이다. 모니터 HSV Lane 패널의 위쪽 **SRC 뷰**(노란 사각형)를 보며
> src 좌표를 트랙에 맞게 조정해라. 조정 전엔 오히려 인식이 나빠질 수 있으니,
> 이상하면 `use_birdeye false` 로 내려라.

## 1. Bird-Eye View (BEV)

ROI를 자른 뒤 ROI BGR 이미지에 perspective warp를 적용한다(결과 크기 동일).
이후 단계(mask/band/junction/yellow)는 변경 없음. warp 실패 시 원본 ROI로 자동 fallback.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `use_birdeye` | `true` | BEV on/off (기본 ON — src 튜닝 필요) |
| `birdeye_src_ratio` | `[0.25,0.05, 0.75,0.05, 0.95,0.95, 0.05,0.95]` | 원본 4점 (TL,TR,BR,BL), ROI 비율 0..1, flat 8개 |
| `birdeye_dst_ratio` | `[0.20,0.00, 0.80,0.00, 0.80,1.00, 0.20,1.00]` | 목적 4점, 같은 형식 |

## 2. Guided Band Search

기존 multi-band는 매 band를 전체 폭에서 고정 중앙(w//2) 기준 좌/우 분리로 탐색한다.
guided 모드는 **직전 band에서 찾은 중심 ± margin 창만** 탐색하고, 좌/우 분리도
guide 중심 기준으로 한다. lane_width EMA·weighted average·curvature 계산은 기존과 동일.
목적: 회전교차로/갈림길에서 중심이 먼 출구선으로 튀는 것 억제.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `use_guided_band` | `true` | guided on/off (기본 ON) |
| `guide_margin_px` | `60` | 직전 중심 ± 탐색 폭 (320px 폭 기준) |
| `guide_margin_growth_px` | `10` | 위 band로 갈수록 margin += i×growth |
| `guide_min_pixels` | `20` | guided 창 내 최소 픽셀 (좁아서 min_pixels보다 낮게) |
| `guide_use_previous_frame` | `true` | 최하단 band 실패 시 이전 프레임 중심으로 재시도 |
| `guide_max_jump_px` | `80` | band 간 중심 점프 clamp |

## 실행 예

```bash
cd D-Racer-Kit
source /opt/ros/humble/setup.bash && source install/setup.bash

# 카메라 (config 명시)
ros2 run camera camera_node --ros-args \
  -p vehicle_config_file:=$PWD/src/config/vehicle_config.yaml

# lane: BEV + guided 켜기
ros2 run perception lane_node --ros-args \
  -p use_birdeye:=true -p use_guided_band:=true \
  -p guide_margin_px:=60 -p guide_margin_growth_px:=10

# 라인추종 로직만 (액추에이터 없음)
ros2 run decision decision_node --ros-args -p skip_missions:=true
```

## 라이브 튜닝 (재시작 불필요)

```bash
ros2 param set /lane_node use_birdeye true
ros2 param set /lane_node use_guided_band true
ros2 param set /lane_node guide_margin_px 50
ros2 param set /lane_node birdeye_src_ratio "[0.3,0.1, 0.7,0.1, 0.95,0.95, 0.05,0.95]"
```

## Debug 이미지 (`/perception/lane/debug`)

- 좌상단 텍스트: `BEV ON/OFF  GUIDED ON/OFF`
- 하단 텍스트: `off`(offset) / `yr`(yellow_ratio) / `cv`(curvature)
- guided 모드: band별 탐색 창(마젠타 사각형) + 찾은 중심(주황 점)
- 기존 표시 유지: 차선 픽셀(시안), lane center(빨강 세로선), 이미지 중앙(초록 세로선)

## 주의

- 수정 후 반드시 리빌드: `colcon build --packages-select perception` (symlink-install 아님)
- BEV를 켜면 junction/yellow 통계도 warped 이미지 기준으로 계산된다.
- 튜닝 우선순위: `roi_top_ratio` → HSV 범위 → `use_birdeye`+src/dst → guided margin → decision의 kp/kd
