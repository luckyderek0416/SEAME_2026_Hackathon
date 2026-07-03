# SEAME 2026 Hackathon — 자율주행 프로젝트

## 스캔하지 말 것 (토큰 절약)
아래는 **생성물/대용량 데이터**라 탐색·검색 대상에서 제외한다. 전체 폴더를 `find`/`glob`로
훑지 말고, 소스는 항상 `D-Racer-Kit/src/` 아래에서만 찾는다.
- `build/`, `install/`, `log/` (루트 및 `D-Racer-Kit/` 둘 다)
- `**/__pycache__/`, `recordings/`, `bagfile/`
- 모델 가중치: `*.pt`, `*.onnx`, `*.engine`, `*.ncnn.bin`, `models/`

## 워크스페이스 (중요)
정식 워크스페이스는 **`D-Racer-Kit/`** 하나뿐이다. 소스·빌드·실행 모두 여기서 한다.
```bash
cd D-Racer-Kit
source /opt/ros/humble/setup.bash
source install/setup.bash          # ← 반드시 D-Racer-Kit/install (루트 install 아님)
```
- 루트(`/home/topst/SEAME_2026_Hackathon-clone/`)의 `build/ install/ log/` 는 **잘못된 위치에서
  빌드된 stale 사본**이다. 루트엔 `src/` 가 없어 리빌드도 불가능하니 **사용/소스 금지**. 정리해도 됨.
- src 를 수정하면 symlink-install 이 아니므로 **반드시 리빌드**해야 반영된다:
  `colcon build --packages-select <pkg>`

## 카메라 노드 주의점
`camera_node` 는 `src/config/vehicle_config.yaml`(USB vs MIPI, 디바이스, 해상도)을 읽는다.
이 파일을 못 찾으면 조용히 **MIPI 640x480 기본값**으로 떨어지고, 실제 카메라는 USB(/dev/video1)라
GStreamer 파이프라인 열기에 실패해 노드가 죽는다. `auto_race.launch.py` 는 이 경로를
`vehicle_config` 인자로 명시 전달하므로 어느 워크스페이스에서 실행해도 안전하다.

## 주행 튜닝 (throttle 등)
- `auto_race.launch.py` 는 `race_config.yaml` 을 **로드하지 않는다** — 이 파일을 고쳐도 반영 안 됨.
  시작값은 `decision_node.py` 의 `declare_parameter` 기본값이다.
- `drive_throttle, slow_throttle, stop_throttle, curve_slow` 는 **주행 중 라이브 변경 가능**:
  ```bash
  ros2 param set /decision_node drive_throttle 0.16
  ```
  마음에 드는 값을 찾으면 `decision_node.py` 기본값에 확정 후 리빌드.

## 실행 시 안전
`ros2 launch decision auto_race.launch.py` 는 실제로 차를 구동한다. 진단은 개별 노드
(예: `ros2 run camera camera_node`)로 하고, 전체 launch 는 사용자 확인 없이 실행하지 않는다.
