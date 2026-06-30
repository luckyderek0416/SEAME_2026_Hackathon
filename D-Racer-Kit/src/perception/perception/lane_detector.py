"""OpenCV lane detector (rule-based, NOT deep learning) — HSV color version.

Synthesised from loun's HSV `lane_follow` (better than plain brightness
thresholding: robust to lighting, supports a yellow center line) but kept on
the kit's `offset` interface so decision_node's PID consumes it unchanged.

Takes a BGR frame, masks the lane color in a bottom region of interest (ROI),
finds the lane center, and returns a normalised steering offset in [-1, 1].
Every constant here is a TUNING knob for the real track. Use
`opencv/hsv_filter.py` to dial in the HSV ranges from a recorded video.

Strategy:
  1. crop the bottom (1 - roi_top_ratio) of the frame (road in front of car)
  2. HSV-threshold the lane color (white boundary lines, or yellow center)
  3. on a horizontal band at `lookahead`, find lane center:
       - white mode: midpoint of left-half / right-half marking means
       - yellow mode: mean x of the single center line
  4. offset = how far that center sits from image center, in [-1, 1]
"""

import cv2
import numpy as np


class LaneDetector:
    def __init__(self, mode='white', roi_top_ratio=0.55, lookahead=0.4,
                 single_line_offset=0.25,
                 white_low=(0, 0, 180), white_high=(180, 40, 255),
                 yellow_low=(10, 80, 80), yellow_high=(35, 255, 255)):
        self.mode = mode                           # 'white' (boundary) / 'yellow' (center line)
        self.roi_top_ratio = roi_top_ratio         # use only the bottom (1 - ratio) of the frame
        self.lookahead = lookahead                 # how far up inside the ROI to sample (0..1)
        self.single_line_offset = single_line_offset  # width fraction to offset when only 1 line seen
        self.white_low = np.array(white_low);   self.white_high = np.array(white_high)
        self.yellow_low = np.array(yellow_low); self.yellow_high = np.array(yellow_high)

    def _mask(self, roi_hsv):
        if self.mode == 'yellow':
            return cv2.inRange(roi_hsv, self.yellow_low, self.yellow_high)
        return cv2.inRange(roi_hsv, self.white_low, self.white_high)

    def process(self, bgr):
        h, w = bgr.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = bgr[roi_top:h, 0:w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = self._mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        roi_h = mask.shape[0]
        ly = int(roi_h * (1 - self.lookahead))
        band = mask[max(0, ly - 5):ly + 5, :]
        mid = w // 2

        lane_found = True
        if self.mode == 'yellow':
            xs = np.where(band > 0)[1]
            if xs.size:
                lane_center, num_lanes = float(xs.mean()), 1
            else:
                lane_center, lane_found, num_lanes = mid, False, 0
        else:
            lx = np.where(band[:, :mid] > 0)[1]
            rx = np.where(band[:, mid:] > 0)[1]
            if len(lx) and len(rx):
                lane_center = (lx.mean() + (rx.mean() + mid)) / 2.0
                num_lanes = 2
            elif len(lx):
                lane_center = lx.mean() + w * self.single_line_offset
                num_lanes = 1
            elif len(rx):
                lane_center = (rx.mean() + mid) - w * self.single_line_offset
                num_lanes = 1
            else:
                lane_center, lane_found, num_lanes = mid, False, 0

        # +1 => lane center far right of image center, -1 => far left
        offset = float(max(-1.0, min(1.0, (lane_center - mid) / (w / 2.0))))

        debug = self._draw_debug(roi, mask, lane_center, ly)
        return lane_found, offset, num_lanes, debug

    def _draw_debug(self, roi, mask, lane_center, ly):
        dbg = roi.copy()
        # tint the detected lane pixels so you can see the mask in the dashboard
        dbg[mask > 0] = (0, 255, 255)
        cx = int(max(0, min(dbg.shape[1] - 1, lane_center)))
        cv2.line(dbg, (cx, 0), (cx, dbg.shape[0]), (0, 0, 255), 2)        # lane center (red)
        cv2.line(dbg, (dbg.shape[1] // 2, 0), (dbg.shape[1] // 2, dbg.shape[0]), (0, 255, 0), 1)  # image center (green)
        cv2.line(dbg, (0, ly), (dbg.shape[1], ly), (255, 0, 0), 1)        # lookahead band (blue)
        return dbg
