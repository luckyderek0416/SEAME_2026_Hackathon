"""NCNN YOLO 탐지기 — 어제 학습한 best_ncnn_model 로 4객체 탐지.

detect(frame) -> [(클래스명, 신뢰도, (x,y,w,h), 화면점유비율), ...]
점유비율(area_frac)로 '가까운(큰) 것만' 골라 미션 트리거에 쓴다.
"""

from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path, conf=0.5, imgsz=320):
        # model_path = NCNN export 폴더 경로 (best_ncnn_model)
        self.model = YOLO(model_path, task='detect')
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                               verbose=False)[0]
        h, w = frame.shape[:2]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                name = self.names[int(b.cls)]
                conf = float(b.conf)
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                area_frac = (x2 - x1) * (y2 - y1) / (w * h)
                out.append((name, conf, (x1, y1, x2 - x1, y2 - y1), area_frac))
        return out
