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
                 single_line_offset=0.25, junction_side='right',
                 junction_dash_transitions=3, junction_min_row_pixels=2, junction_gap_rows=2,
                 mask_mode='hsv', use_white=True, use_yellow=True,
                 white_hsv_lo=(0, 0, 180), white_hsv_hi=(179, 60, 255),
                 yellow_hsv_lo=(18, 80, 80), yellow_hsv_hi=(38, 255, 255),
                 num_bands=4, morph_kernel=3, width_ema=0.1, smooth_alpha=0.5):
        self.roi_top_ratio = roi_top_ratio        # use only the bottom (1 - ratio) of the frame
        self.bright_thresh = bright_thresh         # gray-mode white-line brightness threshold (0-255)
        self.min_pixels = min_pixels               # min lane pixels per side to trust it
        self.single_line_offset = single_line_offset  # fallback offset when no lane width known yet
        # --- robust lane following on a curvy, marble-floor track ---
        self.num_bands = max(1, num_bands)         # horizontal look-ahead bands (curve anticipation)
        self.morph_kernel = morph_kernel           # MORPH_OPEN size; kills glare specks / dash bits (0=off)
        self.width_ema = width_ema                 # how fast the remembered lane width adapts
        self.smooth_alpha = smooth_alpha           # offset EMA (1=no smoothing, lower=smoother)
        self._lane_width = 0.0                     # remembered lane width (px), for single-line centering
        self._offset_ema = 0.0                     # smoothed offset state
        # roundabout junction = the DASHED marking at the entry/exit (on this track the
        # ring is a solid line and only the junction is dashed). Detect the dashes by
        # the lane line turning on/off down the side strip (solid line stays on).
        self.junction_side = junction_side                 # side the dashed junction shows on
        self.junction_dash_transitions = junction_dash_transitions  # vertical on/off changes => dashed
        self.junction_min_row_pixels = junction_min_row_pixels      # row counts as 'line' above this
        self.junction_gap_rows = junction_gap_rows         # full-opening fallback (few rows with line)
        # colour masking: 'hsv' detects white AND/OR yellow lines (robust, distinguishes
        # the yellow roundabout lane); 'gray' is the old brightness-only threshold.
        self.mask_mode = mask_mode
        self.use_white = use_white
        self.use_yellow = use_yellow
        self.white_hsv_lo = tuple(white_hsv_lo)
        self.white_hsv_hi = tuple(white_hsv_hi)
        self.yellow_hsv_lo = tuple(yellow_hsv_lo)
        self.yellow_hsv_hi = tuple(yellow_hsv_hi)

    def _build_mask(self, roi):
        """Return (lane_mask, yellow_ratio, yellow_offset).

        HSV mode = white|yellow lane mask; gray mode = brightness threshold.
        yellow_ratio  = fraction of ROI that is yellow (tells the yellow roundabout
                        from the white outer loop).
        yellow_offset = normalised x of the yellow centroid, [-1..1] (+ = yellow is
                        to the right). Lets decision steer toward the yellow branch
                        regardless of which side it is on (direction-agnostic).
        """
        if self.mask_mode == 'gray':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, self.bright_thresh, 255, cv2.THRESH_BINARY)
            return mask, 0.0, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        w = hsv.shape[1]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        if self.use_white:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, self.white_hsv_lo, self.white_hsv_hi))
        yellow_ratio, yellow_offset = 0.0, 0.0
        if self.use_yellow:
            ymask = cv2.inRange(hsv, self.yellow_hsv_lo, self.yellow_hsv_hi)
            mask = cv2.bitwise_or(mask, ymask)
            yellow_ratio = float(np.count_nonzero(ymask)) / float(ymask.size)
            xs = np.nonzero(ymask)[1]
            if xs.size >= self.min_pixels:
                yellow_offset = float((xs.mean() - w / 2.0) / (w / 2.0))
        return mask, yellow_ratio, yellow_offset

    def process(self, bgr):
        h, w = bgr.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = bgr[roi_top:h, 0:w]

        mask, yellow_ratio, yellow_offset = self._build_mask(roi)

        # (4) noise cleanup: MORPH_OPEN removes marble-floor glare specks and the small
        # gaps of a dashed line, so the lane centre is not pulled by stray pixels.
        if self.morph_kernel and self.morph_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
            clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        else:
            clean = mask

        # (1) multi-band look-ahead: split the ROI into horizontal bands (near..far),
        # find the lane centre in each. Near bands steer; far bands anticipate curves.
        roi_h = clean.shape[0]
        band_h = max(1, roi_h // self.num_bands)
        bands = []   # (center_x, weight, band_index)  band 0 = nearest (bottom of ROI)
        near_lanes = 0
        for i in range(self.num_bands):
            y1 = roi_h - i * band_h
            y0 = max(0, roi_h - (i + 1) * band_h)
            if y1 - y0 < 2:
                continue
            cx, nlanes = self._band_center(clean[y0:y1, :], w)
            if cx is not None:
                bands.append((cx, float(self.num_bands - i), i))  # nearer -> heavier
                if i == 0:
                    near_lanes = nlanes

        mid = w / 2.0
        if not bands:
            lane_found, num_lanes = False, 0
            lane_center, curvature = mid, 0.0
        else:
            lane_found = True
            num_lanes = near_lanes if near_lanes else 1
            tw = sum(wt for _, wt, _ in bands)
            lane_center = sum(cx * wt for cx, wt, _ in bands) / tw
            # (2) curvature = how far the lane drifts from near to far band, normalised.
            # sign = curve direction, magnitude = sharpness (used by decision to slow down).
            curvature = self._estimate_curvature(bands, w)

        raw_offset = float(max(-1.0, min(1.0, (lane_center - mid) / (w / 2.0))))
        # (5) temporal smoothing (EMA) to reduce frame-to-frame steering jitter.
        if lane_found:
            self._offset_ema = (self.smooth_alpha * raw_offset
                                + (1.0 - self.smooth_alpha) * self._offset_ema)
        offset = float(self._offset_ema)

        junction = self._detect_junction(clean, w)

        debug = self._draw_debug(roi, lane_center)
        return lane_found, offset, num_lanes, junction, yellow_ratio, yellow_offset, curvature, debug

    def _band_center(self, band, w):
        """Lane centre for one horizontal band. (3) When both lines are seen, remember
        the lane width; when only one is seen, place the centre half-a-width away — far
        more accurate on curves than a fixed fraction."""
        mid = w // 2
        lx = self._mean_x(band[:, 0:mid], 0)
        rx = self._mean_x(band[:, mid:w], mid)
        if lx is not None and rx is not None:
            width = rx - lx
            if width > 0:
                self._lane_width = (width if self._lane_width <= 0
                                    else (1 - self.width_ema) * self._lane_width
                                    + self.width_ema * width)
            return (lx + rx) / 2.0, 2
        half = (self._lane_width / 2.0) if self._lane_width > 0 else (w * self.single_line_offset)
        if lx is not None:
            return lx + half, 1
        if rx is not None:
            return rx - half, 1
        return None, 0

    @staticmethod
    def _estimate_curvature(bands, w):
        """Normalised lateral drift between the farthest and nearest detected band."""
        if len(bands) < 2:
            return 0.0
        near = min(bands, key=lambda b: b[2])
        far = max(bands, key=lambda b: b[2])
        return float(max(-1.0, min(1.0, (far[0] - near[0]) / (w / 2.0))))

    def _detect_junction(self, mask, w):
        """Junction = the DASHED roundabout entry/exit marking.

        Down the side strip, mark each row 'line present' or not. A SOLID line is
        present in (almost) every row -> ~0 on/off transitions. A DASHED line
        alternates present/absent -> many transitions. So `transitions` tells the
        dashes apart from a solid line. A full opening (very few present rows) is
        also treated as a junction (fallback).
        """
        strip = max(1, w // 3)
        side = mask[:, 0:strip] if self.junction_side == 'left' else mask[:, w - strip:w]
        row_present = (np.count_nonzero(side, axis=1) >= self.junction_min_row_pixels).astype(np.uint8)
        present_rows = int(row_present.sum())
        transitions = int(np.count_nonzero(np.diff(row_present)))  # on/off changes down the strip
        dashed = transitions >= self.junction_dash_transitions
        full_gap = present_rows <= self.junction_gap_rows
        return bool(dashed or full_gap)

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
