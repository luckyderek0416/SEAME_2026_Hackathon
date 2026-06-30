"""라인 추종(비전) — HSV로 선 분리 → lane center → 목표 조향 각도(도).

출력 = 각도(도). 라인 못 찾으면 0(직진). 값(HSV/gain 등)은 실제 영상으로 튜닝.
색 범위는 hsv_filter.py 로 맞춘 값을 넣으면 됨.
"""

import math

import cv2
import numpy as np


class LaneFollower:
    def __init__(self):
        # ===== 색 범위 (hsv_filter.py 로 맞춘 값) [H,S,V] =====
        self.white_low = np.array([0, 0, 180]);   self.white_high = np.array([180, 40, 255])
        self.yellow_low = np.array([10, 80, 80]); self.yellow_high = np.array([35, 255, 255])

        # ===== 주행 파라미터 (손잡이 — 영상 보며 튜닝) =====
        self.mode = 'white'       # 'white'=흰테두리 중점 / 'yellow'=노란중심선
        self.roi_top = 0.55       # 화면 위에서 이 비율 아래만 봄
        self.lookahead = 0.4      # ROI 안 얼마나 멀리(위) 볼지(0~1)
        self.max_angle = 30.0     # 각도 한계(도)

    def _lane_center(self, mask, w):
        roi_h = mask.shape[0]
        ly = int(roi_h * (1 - self.lookahead))
        band = mask[max(0, ly - 5):ly + 5, :]
        if self.mode == 'yellow':
            xs = np.where(band > 0)[1]
            return (int(xs.mean()) if len(xs) else None), ly
        lx = np.where(band[:, :w // 2] > 0)[1]
        rx = np.where(band[:, w // 2:] > 0)[1]
        if len(lx) and len(rx):
            cx = (lx.mean() + (rx.mean() + w // 2)) / 2
        elif len(lx):
            cx = lx.mean() + w * 0.25
        elif len(rx):
            cx = (rx.mean() + w // 2) - w * 0.25
        else:
            return None, ly
        return int(cx), ly

    def compute_angle(self, frame):
        """frame(BGR) -> (angle_deg, info). angle: -좌 ~ +우."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y0 = int(h * self.roi_top)
        roi = hsv[y0:, :]
        lo, hi = ((self.yellow_low, self.yellow_high) if self.mode == 'yellow'
                  else (self.white_low, self.white_high))
        mask = cv2.inRange(roi, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        cx, ly = self._lane_center(mask, w)
        if cx is None:
            return 0.0, (mask, None, None, y0)
        cy = y0 + ly
        angle = math.degrees(math.atan2(cx - w / 2, h - cy))
        angle = max(-self.max_angle, min(self.max_angle, angle))
        return angle, (mask, cx, cy, y0)
