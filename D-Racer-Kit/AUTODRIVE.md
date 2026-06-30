# D-Racer 자율주행 ROS2 스택 (SEA:ME 2026)

기존 TOPST D-Racer-Kit 위에 **자율주행에 필요한 노드만 추가**한 워크스페이스입니다.
킷이 기본 제공하는 노드(`camera`, `control`, `joystick`, `monitor`, `battery`)는
그대로 두고, 아래 5개 패키지를 추가했습니다.

## 데이터 흐름

```
camera_node ──/camera/image/compressed──┬──> lane_node   ──/perception/lane──────┐
                                         ├──> aruco_node  ──/perception/aruco─────┤
                                         └──> yolo_node   ──/inference/detections─┤
                                                                                  │
                                                       decision_node (상태머신+PID) <┘
                                                              │
                                                          /control (control_msgs/Control)
                                                              │
joystick_node ──/joystick(E-stop)──> control_node ──> 서보/ESC (차량 구동)
```

- **운전(조향)** = `lane_node`(OpenCV 차선) → `decision_node`의 PID. 딥러닝 아님.
- **미션 인식** = `yolo_node`(YOLO, 4객체) + `aruco_node`(ArUco 마커).
- **YOLO는 오직 `inference/yolo_node` 한 곳에만** 있습니다.

## 추가한 패키지

| 패키지 | 타입 | 들어있는 것 | 역할 |
|---|---|---|---|
| `perception_msgs` | msgs | `LaneState.msg`, `ArucoState.msg` | 차선/마커 결과 메시지 |
| `inference_msgs` | msgs | `Detection.msg`, `Detections.msg` | YOLO 검출 메시지 |
| `perception` | python | `lane_node.py`, `aruco_node.py`, `lane_detector.py` | **HSV** 차선추종(흰/노란선) + ArUco |
| `inference` | python | `yolo_node.py` | **YOLO 객체검출(NCNN, 유일한 딥러닝 노드)** |
| `decision` | python | `decision_node.py`, `state_machine.py`, `pid.py`, `launch/`, `config/` | 상태머신 + 차선 PID, /control 발행 |

## 상태머신 (decision/state_machine.py)

```
WAIT_GREEN ──(초록불)──▶ DRIVE ──(In코스면 먼저 ROUNDABOUT)
   DRIVE ──(좌/우 표지판 latch → 갈림길에서 분기)──▶ 계속 DRIVE
   any   ──(아루코 등장)──▶ OBSTACLE_STOP ──(사라짐)──▶ 직전 상태로 복귀
   DRIVE ──(빨간불)──▶ FINISH ──(정지)──▶ DONE
```

- 조향은 항상 차선 PID. 상태는 throttle과 미션 행동만 바꿈.
- **회전교차로 한 바퀴 판정**은 지금 시간 기반(`roundabout_seconds`) 임시 구현.
  보드에 IMU가 있으면 `_roundabout_done()`을 **yaw 각도 적분(누적 ~330° 시 탈출)**으로 교체하세요. 그게 훨씬 안정적입니다.

## 빌드 & 실행

```bash
cd ~/SEAME_2026_Hackathon/D-Racer-Kit
pip install ultralytics ncnn       # yolo_node(NCNN 추론) 용
colcon build
source install/setup.bash

# Out 코스 (S자 + 갈림길)  — model_path 기본값 = training/best_ncnn_model
ros2 launch decision auto_race.launch.py course:=out

# In 코스 (회전교차로)
ros2 launch decision auto_race.launch.py course:=in

# 노란 중심선 트랙 / 조향 반대일 때
ros2 launch decision auto_race.launch.py course:=out lane_mode:=yellow steer_scale:=-1.0
```

> 모델 경로 기본값은 `~/SEAME_2026_Hackathon/training/best_ncnn_model`(NCNN export 폴더)입니다.
> 다른 모델을 쓰려면 `model_path:=/경로/best_ncnn_model` 또는 `.pt` 파일을 넘기세요.
> ⚠️ 보드 OpenCV는 GStreamer 빌드(시스템 4.5.x)가 필요하고 numpy<2 여야 합니다
> (pip `opencv-python`이 깔려 있으면 제거 → 시스템 cv2 사용).

## 튜닝 순서 (중요)

1. **먼저 차선 인식부터.** `lane_node`의 debug 토픽(`/perception/lane/debug`)을 모니터로 보면서
   차선 중심선(빨간 선)이 안정적으로 잡히게 맞춥니다. 트랙 선 색에 따라 `mode`(white/yellow),
   `roi_top_ratio`, `lookahead`를 조정. HSV 범위는 `opencv/hsv_filter.py`로 영상 보며 튜닝.
2. 그 다음 **PID**: `ki=0`으로 두고 `kp`를 올려 따라가게 → 떨리면 `kd`로 잡기.
   차가 반대로 꺾이면 `steer_scale`을 음수로.
3. **throttle**: `drive_throttle`을 낮게 시작해서 안 벗어나는 선까지.
4. 미션은 마지막에 하나씩: green → red → 표지판 → 아루코 → (시간 남으면) 회전교차로.

## 먼저 할 일

미션 다 빼고 **Out 코스 기본 주행으로 한 바퀴**부터 완성하세요.
`yolo_node`는 모델이 없어도 빈 검출을 발행하므로, 학습 전이라도 차선주행 테스트가 됩니다.
