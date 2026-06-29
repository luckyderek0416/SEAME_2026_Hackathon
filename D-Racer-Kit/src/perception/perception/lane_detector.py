"""OpenCV lane detector (rule-based, NOT deep learning).

Takes a BGR frame, looks at a bottom region of interest (ROI), finds the
bright lane markings on the left and right, and returns a normalised
steering offset. Every constant here is a TUNING knob for the real track.

Baseline strategy:
  1. crop the bottom part of the frame (the road right in front of the car)
  2. threshold for bright pixels (the white boundary lines)
  3. find the mean x of bright pixels on each half -> the two lane lines
  4. lane center = midpoint of the two lines
  5. offset = how far that center sits from the image center, in [-1, 1]

If only one line is visible we estimate the center from it. If the track
has a yellow center line you can add a second HSV mask and blend it in.
"""

import cv2
import numpy as np


class LaneDetector:
    def __init__(self, roi_top_ratio=0.55, bright_thresh=160, min_pixels=40,
                 single_line_offset=0.25):
        self.roi_top_ratio = roi_top_ratio        # use only the bottom (1 - ratio) of the frame
        self.bright_thresh = bright_thresh         # white-line brightness threshold (0-255)
        self.min_pixels = min_pixels               # min bright pixels per side to trust it
        self.single_line_offset = single_line_offset  # fraction of width to offset when only 1 line seen

    def process(self, bgr):
        h, w = bgr.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = bgr[roi_top:h, 0:w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.bright_thresh, 255, cv2.THRESH_BINARY)

        mid = w // 2
        left_x = self._mean_x(mask[:, 0:mid], x_offset=0)
        right_x = self._mean_x(mask[:, mid:w], x_offset=mid)

        lane_found = True
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2.0
            num_lanes = 2
        elif left_x is not None:
            lane_center = left_x + w * self.single_line_offset
            num_lanes = 1
        elif right_x is not None:
            lane_center = right_x - w * self.single_line_offset
            num_lanes = 1
        else:
            lane_center = mid
            lane_found = False
            num_lanes = 0

        # +1 => lane center far right of image center, -1 => far left
        offset = (lane_center - mid) / (w / 2.0)
        offset = float(max(-1.0, min(1.0, offset)))

        debug = self._draw_debug(roi, lane_center)
        return lane_found, offset, num_lanes, debug

    def _mean_x(self, half_mask, x_offset):
        xs = np.nonzero(half_mask)[1]
        if xs.size < self.min_pixels:
            return None
        return float(np.mean(xs)) + x_offset

    def _draw_debug(self, roi, lane_center):
        dbg = roi.copy()
        cx = int(max(0, min(dbg.shape[1] - 1, lane_center)))
        cv2.line(dbg, (cx, 0), (cx, dbg.shape[0]), (0, 0, 255), 2)        # detected lane center (red)
        cv2.line(dbg, (dbg.shape[1] // 2, 0), (dbg.shape[1] // 2, dbg.shape[0]), (0, 255, 0), 1)  # image center (green)
        return dbg
