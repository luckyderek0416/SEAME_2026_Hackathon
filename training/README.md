# 딥러닝 파트 — YOLO 객체탐지 (SEA:ME 2026 Hackathon)

신호등/표지판 **4객체**를 YOLO26n으로 탐지하는 **학습·추론 코드**.
(차량 연결 / 제어 / OpenCV 차선주행은 팀의 다른 파트가 담당)

## 4 클래스 (data.yaml)
| id | 클래스 | 미션 |
|----|--------|------|
| 0 | red_light | 도착 시 정지 |
| 1 | green_light | 출발 |
| 2 | left_sign | 갈림길 좌회전 |
| 3 | right_sign | 갈림길 우회전 |

## 파이프라인
```
1. 프레임 추출   python extract_frames.py --video <영상> --out frames/
2. 라벨링        (Roboflow / labelImg 등으로 4클래스 박스 라벨) → dataset/
3. 학습          python train.py --model yolo26n.pt --epochs 100 --imgsz 320
4. 추론 테스트   python detect.py --weights runs/detect/train/weights/best.pt --source <img/video>
5. 차량 배포     best.pt + 추론 노드를 팀 ROS 패키지로 전달
```

## 데이터셋 구조 (라벨링 후)
```
dataset/
  images/train/*.jpg   labels/train/*.txt
  images/val/*.jpg     labels/val/*.txt
```

## 설치
```bash
pip install ultralytics opencv-python
```

## 차량 노드와의 토픽 약속(인터페이스)
탐지 결과를 아래 토픽으로 발행한다 (팀과 합의 후 확정):
- `/perception/traffic_light` — red / green
- `/perception/turn_sign` — left / right

## 메모
- `yolo26n.pt` 가 없으면 `yolo11n.pt` 또는 `yolov8n.pt` 로 대체 가능 (모두 nano).
- D3-G(엣지)에서 가볍게: `imgsz` 작게, 낮은 fps, ROI, NPU(ONNX/TFLite) 변환.
- 데이터셋/가중치(*.pt)는 용량이 커서 git 에 올리지 않음(.gitignore).
