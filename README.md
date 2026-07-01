# SEAME_2026_Hackathon
TEAM : chita

---

## 🔧 `taehulim` 브랜치 변경점 (자율주행 스택 보강)

공식 클론(main) 기준으로 **라인추종·회전교차로·YOLO**를 보강한 브랜치입니다.
자세한 원본 vs 변경 비교 → [`D-Racer-Kit/docs/CHANGES_lane_and_roundabout.md`](D-Racer-Kit/docs/CHANGES_lane_and_roundabout.md)

### 라인추종 (perception/lane_detector.py)
| | 원본(main) | 변경(taehulim) |
|---|---|---|
| 색 검출 | 흑백 밝기 임계값 (흰선만) | **HSV 흰색+노란색** (회전교차로 노란차선 대응) |
| 시야 | 단일 밴드(코앞만) | **멀티밴드 룩어헤드** (커브 미리 봄) |
| 강건성 | 평균 x (노이즈에 약함) | **노이즈정리(morph) + 차선폭 기억 + offset 평활** |
| 속도 | 일정 | **곡률 기반 감속** |

### 회전교차로 (decision/state_machine.py)
| | 원본(main) | 변경(taehulim) |
|---|---|---|
| 한 바퀴 판정 | **시간 8초** 만 | **junction(점선)+조향각적분+시간 3중 투표** (마커·IMU 없이) |
| 진입 감지 | 출발 즉시 진입 | **노란색+급커브 게이팅** (흰 코너 오진입 차단) |
| 조기 탈출 방지 | 없음 | **min_time 하드플로어+조향 바이어스** (1바퀴 미만 탈출=실패 방지) |
| 방향 대응 | 없음 | **race_dir** (정/역방향 트랙 일괄 설정) |

### 그 외
- **YOLO**: ultralytics → **ncnn 기반 `yolo_ncnn_node`** (ARM 보드 경량) + 학습모델(`models/`) 포함
- **ArUco**: 반전(inverted) 마커 지원 추가
- **테스트 모드**: `skip_missions:=true` → 라인추종만 단독 테스트
- **도구**: `tools/hsv_sampler.py`(색 측정), `tools/identify_aruco.py`(마커 식별)
- **삭제**: makedb, opencv(패키지), image_raw.jpg

### 실행 (라인추종만 테스트)
```bash
cd D-Racer-Kit
colcon build && source install/setup.bash
# 4개 노드: camera / lane / decision(skip_missions) / control
ros2 run decision decision_node --ros-args -p skip_missions:=true -p drive_throttle:=0.3
```

### 트랙에서 튜닝할 값
`kp/kd`(라인추종), `drive_throttle/slow_throttle`(ESC 데드밴드),
`yellow/white_hsv_*`(색, hsv_sampler로 실측), `nominal_loop_time_s`·`yaw_lap_threshold`(회전교차로),
`race_dir`(배정 방향) — 자세한 건 위 CHANGES 문서 참고.
