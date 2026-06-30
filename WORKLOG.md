# 작업 일지 (loun) — 자율주행 인지·주행 파트

> 이 브랜치(`loun`)에는 **내가 담당한 파트만** 올라갑니다:
> - `training/` — 딥러닝(YOLO 객체 인지)
> - `opencv/` — OpenCV 주행 + PD 제어 + 미션 행동
>
> (차량 연결/하드웨어/팀 공용 ROS 패키지는 다른 팀원 담당이라 제외)

---

## 1. 딥러닝 파트 — `training/` (객체 인지)

신호등·표지판 **4객체**를 YOLO로 탐지하는 모델을 만들었습니다.

**클래스(id 순서):** `0 red_light, 1 green_light, 2 left_sign, 3 right_sign`

| 파일 | 역할 |
|---|---|
| `build_dataset.py` | 표지판 영상 → 프레임 추출 + **파란 원 자동 라벨링** → YOLO 데이터셋 |
| `extract_frames.py` | 영상에서 프레임 추출 |
| `train.py` | YOLO 학습 |
| `detect.py` | 추론 테스트 |
| `data.yaml`, `classes.txt` | 4클래스 정의 |
| `best_ncnn_model/` | **학습된 모델(NCNN, 차 배포용)** ⭐ |

**데이터 구축 방식**
- 표지판(좌/우): 영상이 한 종류라 **자동 라벨링**(OpenCV 파란 원 검출)
- 신호등(빨강/초록): **수동 라벨링**(labelImg) — 불빛 색이 배경과 섞여 자동 어려움
- venue(대리석 바닥) 영상: 실전 배경, 신호등+표지판 수동 라벨
- 총 305장, 4클래스

**학습 결과(베이스라인):** mAP50 ≈ 0.95 (val), 추론 ~9ms/장. → **`best_ncnn_model/`** 로 배포.

> ⚠️ val이 같은 영상 기준이라 점수가 후함. **실제 트랙 데이터로 재학습** 필요(대회장).
> 학습용 .pt/데이터셋은 용량 커서 git 제외(로컬 보관). 레포엔 **NCNN 모델만**.

---

## 2. OpenCV 주행 파트 — `opencv/` (주행 + 제어 + 미션)

카메라로 라인을 따라가고, 인지 결과로 출발/정지/좌우를 결정하는 주행 로직.

| 파일 | 역할 |
|---|---|
| `hsv_filter.py` | **HSV 색 필터 튜닝 도구** — 흰 선/노란 선 분리(트랙바) |
| `lane_follow.py` | **라인 검출(비전)** — HSV로 선 분리 → lane center → 목표 **각도(도)** |
| `pd_controller.py` | **PD 제어** — 목표 각도 → 부드러운 조향값(-1~1) |
| `yolo_detector.py` | **NCNN 인지** — `best_ncnn_model`로 4객체 탐지 |
| `mission_node.py` | **메인 주행 노드** — 인지+라인추종+행동 상태머신 (ROS) |
| `lane_node.py` | 라인추종만 하는 ROS 노드(테스트용) |

### 미션 상태머신 (`mission_node.py`)
```
[WAIT_START] 정지 대기
   │ 🟢 green_light 인식
   ▼
[DRIVING] throttle 0.20 시작 → 서서히 증가 / 조향=라인추종
   │  ⬅️ left_sign  → 잠깐 좌회전 강제
   │  ➡️ right_sign → 잠깐 우회전 강제
   │ 🔴 red_light 인식
   ▼
[STOPPED] 정지
```
- 초록불: `start_throttle`(0.20)에서 `ramp_rate`로 가속
- 빨간불: throttle 0
- 갈림길: 표지판 방향으로 `fork_duration`초 강제 조향
- 인지는 `min_area`로 **가까운(큰) 것만** 반응 → 헛동작 방지

---

## 3. 전체 데이터 흐름 (ROS)

```
camera_node ─/camera/image/compressed─→ mission_node
                                          ├ YOLO(NCNN) 인지 → 신호등/표지판
                                          ├ LaneFollower → 목표 각도
                                          └ PD + 상태머신 → steering, throttle
                                                 │ /control (control_msgs/Control)
                                                 ▼
                                          control_node(D-Racer) → 차
```

**토픽 인터페이스(팀 연동):** `mission_node` 가 `/control`(steering -1~1, throttle -1~1) 발행 → D-Racer `control_node` 가 받음.
- 팀 PID를 쓰면 `use_pd:=false`(lane_node) 로 두고 각도만 넘기는 것도 가능.

---

## 4. 실행 방법

**학습(PC):**
```bash
cd training
python build_dataset.py --src <영상폴더>   # 데이터셋
python train.py --epochs 100 --imgsz 320   # 학습 → best.pt
yolo export model=best.pt format=ncnn imgsz=320   # NCNN 변환
```

**주행(차):**
```bash
cd opencv
python3 mission_node.py --ros-args -p start_throttle:=0.2 -p fork_steer:=0.5
```

**색/라인 튜닝(영상으로):**
```bash
python3 hsv_filter.py --source track.mp4   # HSV 색 범위
python3 lane_follow.py --source track.mp4  # (튜너 별도 추가 예정)
```

---

## 5. 남은 일 (TODO)
- [ ] **실제 카메라 영상으로 HSV/ROI/lookahead 튜닝** (값은 현재 기본값)
- [ ] **실제 주행으로 Kp/Kd/속도 튜닝**
- [ ] 회전교차로(타이머 상태) 추가
- [ ] 동적 장애물(ArUco) 정지 로직 추가
- [ ] 대회장 트랙 데이터로 YOLO 재학습 (정확도 향상)

---
_작성: loun / 오늘 작업 기준_
