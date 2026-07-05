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
                 num_bands=4, morph_kernel=3, width_ema=0.1, smooth_alpha=0.5,
                 use_birdeye=False, birdeye_src_ratio=None, birdeye_dst_ratio=None,
                 use_guided_band=False, guide_margin_px=60, guide_margin_growth_px=10,
                 guide_min_pixels=20, guide_use_previous_frame=True, guide_max_jump_px=80,
                 use_lookahead_control=False, near_weight=0.7, lookahead_weight=0.3,
                 lookahead_band_index=-1, adaptive_lookahead=False,
                 curve_lookahead_weight=0.4, curve_lookahead_thresh=0.25,
                 crossline_roi_top_ratio=0.55, crossline_roi_bottom_ratio=0.90,
                 crossline_min_width_ratio=0.30, crossline_min_rows=4,
                 fork_scan_top_ratio=0.0, fork_scan_bottom_ratio=0.5,
                 fork_col_min_ratio=0.15, fork_min_groups=3, fork_span_ratio=0.65,
                 fork_seed_px=90):
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
        # --- bird-eye view (optional, default OFF = legacy behaviour) ---
        # src/dst are flat ratio lists [x1,y1, x2,y2, x3,y3, x4,y4] (TL,TR,BR,BL) in 0..1
        # of the ROI size, so they survive resolution changes.
        self.use_birdeye = use_birdeye
        self.birdeye_src_ratio = (list(birdeye_src_ratio) if birdeye_src_ratio
                                  else [0.25, 0.05, 0.75, 0.05, 0.95, 0.95, 0.05, 0.95])
        self.birdeye_dst_ratio = (list(birdeye_dst_ratio) if birdeye_dst_ratio
                                  else [0.20, 0.00, 0.80, 0.00, 0.80, 1.00, 0.20, 1.00])
        self._bev_matrix = None
        self._bev_key = None       # (w, h, src, dst) the cached matrix was built for
        self._bev_warned = False
        # --- guided band search (optional, default OFF = legacy multi-band) ---
        # Each band only searches around the previous band's centre so a far
        # exit/branch line cannot yank the lane centre (roundabout/fork robustness).
        self.use_guided_band = use_guided_band
        self.guide_margin_px = guide_margin_px
        self.guide_margin_growth_px = guide_margin_growth_px
        self.guide_min_pixels = guide_min_pixels
        self.guide_use_previous_frame = guide_use_previous_frame
        self.guide_max_jump_px = guide_max_jump_px
        self._prev_center = None   # last frame's lane_center (guide seed)
        # --- look-ahead steering blend (partial pure-pursuit; default OFF) ---
        # Legacy lane_center is the all-band weighted average. With this ON the
        # centre becomes near_weight*nearest_band + lookahead_weight*far_band, so
        # the car starts turning into a curve earlier. NOT a full pure-pursuit:
        # the PID in decision stays the controller, we only shape its input.
        self.use_lookahead_control = use_lookahead_control
        self.near_weight = near_weight
        self.lookahead_weight = lookahead_weight
        self.lookahead_band_index = lookahead_band_index   # -1 = farthest detected band
        self.adaptive_lookahead = adaptive_lookahead
        self.curve_lookahead_weight = curve_lookahead_weight  # lookahead weight on sharp curves
        self.curve_lookahead_thresh = curve_lookahead_thresh  # |curvature| that counts as sharp
        # --- yellow crossline (노란 가로선) detection ---
        # A wide horizontal yellow band only appears near the roundabout entry/exit
        # fork, so it is a POSITION signal. Detected on the yellow-only mask (never
        # the combined lane mask) and used purely as one extra vote in decision's
        # roundabout entry/exit logic — it does NOT touch lane-centre computation.
        self.crossline_roi_top_ratio = crossline_roi_top_ratio       # scan window top (0..1 of ROI h)
        self.crossline_roi_bottom_ratio = crossline_roi_bottom_ratio # scan window bottom
        self.crossline_min_width_ratio = crossline_min_width_ratio   # row is "wide" above this fraction of w
        self.crossline_min_rows = crossline_min_rows                 # need this many wide rows
        # --- 좌/우 갈림길 (fork) 감지 + 브랜치 선택 ---
        # 분기 구간은 도로가 두 브랜치로 갈라져, BEV 상단 스캔밴드에서 세로 라인 군집이
        # 3개 이상(각 브랜치 안/바깥선)이거나 바깥 라인 간격이 한 차선보다 훨씬 넓어진다.
        # 이 fork 플래그는 decision 의 turn_latch 해제(도로 재수렴) 기준으로만 쓰이고,
        # 차선 중심 계산은 건드리지 않는다.
        self.fork_scan_top_ratio = fork_scan_top_ratio        # 스캔밴드 상단 (BEV far)
        self.fork_scan_bottom_ratio = fork_scan_bottom_ratio  # 스캔밴드 하단
        self.fork_col_min_ratio = fork_col_min_ratio          # 컬럼이 '라인'으로 세는 세로픽셀 비율
        self.fork_min_groups = fork_min_groups                # 라인 군집 이 개수 이상 => 분기
        self.fork_span_ratio = fork_span_ratio                # 바깥라인 간격 이 폭 이상 => 분기(폴백)
        # fork_dir: decision 이 표결 확정한 방향('left'/'right'/None). lane_node 가
        # /decision/fork_dir 구독으로 매 프레임 갱신한다. 설정되면 guided-band 시드를
        # 그 브랜치 쪽으로 밀어(fork_seed_px) 한쪽 브랜치만 추종 => median(표지판 섬) 배제.
        self.fork_dir = None
        self.fork_seed_px = fork_seed_px

    def _build_mask(self, roi):
        """Return (lane_mask, yellow_ratio, yellow_offset, yellow_crossline).

        HSV mode = white|yellow lane mask; gray mode = brightness threshold.
        yellow_ratio  = fraction of ROI that is yellow (tells the yellow roundabout
                        from the white outer loop).
        yellow_offset = normalised x of the yellow centroid, [-1..1] (+ = yellow is
                        to the right). Lets decision steer toward the yellow branch
                        regardless of which side it is on (direction-agnostic).
        yellow_crossline = a wide horizontal yellow band is in view (roundabout
                        entry/exit fork position marker). Computed on the
                        yellow-only mask; never affects the lane centre.
        """
        if self.mask_mode == 'gray':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, self.bright_thresh, 255, cv2.THRESH_BINARY)
            return mask, 0.0, 0.0, False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        w = hsv.shape[1]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        if self.use_white:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, self.white_hsv_lo, self.white_hsv_hi))
        yellow_ratio, yellow_offset = 0.0, 0.0
        yellow_crossline = False
        if self.use_yellow:
            ymask = cv2.inRange(hsv, self.yellow_hsv_lo, self.yellow_hsv_hi)
            mask = cv2.bitwise_or(mask, ymask)
            yellow_ratio = float(np.count_nonzero(ymask)) / float(ymask.size)
            xs = np.nonzero(ymask)[1]
            if xs.size >= self.min_pixels:
                yellow_offset = float((xs.mean() - w / 2.0) / (w / 2.0))
            yellow_crossline = self._detect_crossline(ymask)
        return mask, yellow_ratio, yellow_offset, yellow_crossline

    def _detect_crossline(self, ymask):
        """Yellow HORIZONTAL line detector (cheap row-count scan, no contours/Hough).

        Scans the lower-middle window of the yellow-only mask: a row counts as
        "wide" when its yellow pixels span more than crossline_min_width_ratio of
        the image width (a normal lane line is a thin, near-vertical stripe and
        never gets that wide), and we need crossline_min_rows such rows.
        Runs on the BEV-warped mask when use_birdeye is on (roi is already warped
        before _build_mask), which makes the line's width perspective-stable.
        """
        h_m, w_m = ymask.shape[:2]
        y1 = int(h_m * self.crossline_roi_top_ratio)
        y2 = int(h_m * self.crossline_roi_bottom_ratio)
        if y2 <= y1:
            return False
        cross_roi = ymask[y1:y2, :]
        row_counts = np.count_nonzero(cross_roi, axis=1)
        rows_hit = row_counts > int(w_m * self.crossline_min_width_ratio)
        return bool(np.count_nonzero(rows_hit) >= int(self.crossline_min_rows))

    def process(self, bgr, draw_debug=True):
        h, w = bgr.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = bgr[roi_top:h, 0:w]

        roi_raw = roi   # keep the pre-warp ROI so the debug view can show the BEV source region
        if self.use_birdeye:
            roi = self._apply_birdeye(roi)   # falls back to raw ROI on failure

        mask, yellow_ratio, yellow_offset, yellow_crossline = self._build_mask(roi)

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
        band_windows = []   # guided-mode search windows, for the debug overlay
        if self.use_guided_band:
            bands, near_lanes, band_windows = self._guided_bands(clean, w, roi_h, band_h)
        else:
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
        near_cx, la_cx = None, None
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
            # partial pure-pursuit: blend nearest + look-ahead band centres instead of
            # the all-band average. OFF (default) keeps lane_center above unchanged.
            near_cx = bands[0][0]                    # nearest detected band
            la_cx = self._lookahead_center(bands)
            if self.use_lookahead_control and la_cx is not None:
                nw, lw = self.near_weight, self.lookahead_weight
                if self.adaptive_lookahead and abs(curvature) >= self.curve_lookahead_thresh:
                    lw = self.curve_lookahead_weight  # sharp curve: look further ahead
                    nw = max(0.0, 1.0 - lw)
                if nw + lw > 0.0:
                    lane_center = (nw * near_cx + lw * la_cx) / (nw + lw)

        raw_offset = float(max(-1.0, min(1.0, (lane_center - mid) / (w / 2.0))))
        # (5) temporal smoothing (EMA) to reduce frame-to-frame steering jitter.
        if lane_found:
            self._offset_ema = (self.smooth_alpha * raw_offset
                                + (1.0 - self.smooth_alpha) * self._offset_ema)
            self._prev_center = lane_center   # guide seed for the next frame
        offset = float(self._offset_ema)

        junction = self._detect_junction(clean, w)
        fork = self._detect_fork(clean, w)

        # 디버그 이미지는 웹 대시보드용일 뿐 주행에 안 쓰인다. 프레임을 띄엄띄엄
        # (lane_node 의 debug_hz) 보낼 때 그리기+인코딩을 통째로 건너뛰어 보드 부하를 던다.
        debug = (self._draw_debug(roi, clean, lane_center, offset, yellow_ratio,
                                  curvature, band_windows, roi_raw, near_cx, la_cx)
                 if draw_debug else None)
        return (lane_found, offset, num_lanes, junction, yellow_ratio, yellow_offset,
                curvature, yellow_crossline, fork, debug)

    def _band_center(self, band, w):
        """Lane centre for one horizontal band, searching the FULL width and
        splitting left/right at the fixed image middle (legacy behaviour)."""
        mid = w // 2
        lx = self._mean_x(band[:, 0:mid], 0)
        rx = self._mean_x(band[:, mid:w], mid)
        return self._combine_lr(lx, rx, w)

    def _combine_lr(self, lx, rx, w):
        """Combine left/right line x-positions into a lane centre. (3) When both
        lines are seen, remember the lane width; when only one is seen, place the
        centre half-a-width away — far more accurate on curves than a fixed fraction."""
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

    # ---------- guided band search (optional) ----------
    def _guided_bands(self, clean, w, roi_h, band_h):
        """Multi-band centres where each band only searches around the previous
        band's centre (± margin). Keeps _band_center's left/right split and
        lane-width memory — just windowed — so a far exit/branch line cannot
        yank the centre. NOT a sliding-window/polyfit fit.

        Returns (bands, near_lanes, band_windows) with bands in the legacy
        [(center_x, weight, band_index), ...] format and band_windows =
        [(x0, y0, x1, y1, cx_or_None), ...] for the debug overlay."""
        bands = []
        band_windows = []
        near_lanes = 0
        guide = None
        # 브랜치 선택: 방향이 확정됐으면(=분기 도착) 시드를 그 브랜치 쪽으로 밀어
        # band 0 부터 windowed search 로 한쪽 브랜치만 좇게 한다. guide_max_jump_px 클램프가
        # 다음 밴드에서 반대 브랜치로 튀는 것을 막아 락온을 유지한다.
        if self.fork_dir == 'left':
            guide = w / 2.0 - self.fork_seed_px
        elif self.fork_dir == 'right':
            guide = w / 2.0 + self.fork_seed_px
        for i in range(self.num_bands):
            y1 = roi_h - i * band_h
            y0 = max(0, roi_h - (i + 1) * band_h)
            if y1 - y0 < 2:
                continue
            band = clean[y0:y1, :]
            if guide is None:
                # no guide yet: full-width search (band 0, or all bands so far empty)
                cx, nlanes = self._band_center(band, w)
                x0, x1 = 0, w
                if (cx is None and i == 0 and self.guide_use_previous_frame
                        and self._prev_center is not None):
                    x0, x1, cx, nlanes = self._windowed_center(band, w, self._prev_center, i)
            else:
                x0, x1, cx, nlanes = self._windowed_center(band, w, guide, i)
            if cx is not None:
                if guide is not None:
                    jump = cx - guide
                    if abs(jump) > self.guide_max_jump_px:   # clamp sudden centre jumps
                        cx = guide + (self.guide_max_jump_px if jump > 0
                                      else -self.guide_max_jump_px)
                bands.append((cx, float(self.num_bands - i), i))  # nearer -> heavier
                if i == 0:
                    near_lanes = nlanes
                guide = cx
            band_windows.append((x0, y0, x1, y1, cx))
        return bands, near_lanes, band_windows

    def _windowed_center(self, band, w, guide, band_index):
        """_band_center restricted to [guide-margin, guide+margin], with the
        left/right split at the guide centre instead of the fixed w//2."""
        margin = self.guide_margin_px + band_index * self.guide_margin_growth_px
        x0 = int(max(0, min(w - 2, guide - margin)))
        x1 = int(max(x0 + 2, min(w, guide + margin)))
        mid = int(max(x0 + 1, min(x1 - 1, round(guide))))
        lx = self._mean_x(band[:, x0:mid], x0, self.guide_min_pixels)
        rx = self._mean_x(band[:, mid:x1], mid, self.guide_min_pixels)
        cx, nlanes = self._combine_lr(lx, rx, w)
        return x0, x1, cx, nlanes

    # ---------- bird-eye view (optional) ----------
    def invalidate_birdeye_cache(self):
        """Force the perspective matrix to be rebuilt (call after src/dst change)."""
        self._bev_key = None
        self._bev_warned = False

    def _apply_birdeye(self, roi):
        """Warp the ROI to a top-down view (same size). Returns the input ROI
        unchanged on any failure so lane following never dies from bad points."""
        h, w = roi.shape[:2]
        key = (w, h, tuple(self.birdeye_src_ratio), tuple(self.birdeye_dst_ratio))
        if key != self._bev_key:
            self._bev_key = key
            self._bev_matrix = None
            try:
                if len(self.birdeye_src_ratio) != 8 or len(self.birdeye_dst_ratio) != 8:
                    raise ValueError('birdeye_*_ratio must be 8 floats [x1,y1,...,x4,y4]')
                src = np.float32([(self.birdeye_src_ratio[i] * w,
                                   self.birdeye_src_ratio[i + 1] * h) for i in range(0, 8, 2)])
                dst = np.float32([(self.birdeye_dst_ratio[i] * w,
                                   self.birdeye_dst_ratio[i + 1] * h) for i in range(0, 8, 2)])
                self._bev_matrix = cv2.getPerspectiveTransform(src, dst)
            except Exception as exc:
                if not self._bev_warned:
                    print(f'[lane_detector] bird-eye disabled, using raw ROI: {exc}')
                    self._bev_warned = True
        if self._bev_matrix is None:
            return roi
        try:
            return cv2.warpPerspective(roi, self._bev_matrix, (w, h))
        except cv2.error as exc:
            if not self._bev_warned:
                print(f'[lane_detector] bird-eye warp failed, using raw ROI: {exc}')
                self._bev_warned = True
            return roi

    def _lookahead_center(self, bands):
        """Centre x of the look-ahead band: lookahead_band_index if that band was
        detected this frame, otherwise the farthest detected band (-1 = always
        farthest). bands are ordered near -> far."""
        if not bands:
            return None
        if self.lookahead_band_index >= 0:
            for cx, _, i in bands:
                if i == self.lookahead_band_index:
                    return cx
        return bands[-1][0]

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

    def _detect_fork(self, mask, w):
        """좌/우 갈림길(fork) 구간 감지.

        분기에서는 도로가 두 브랜치로 갈라져, BEV 상단 스캔밴드의 세로 라인 군집이
        늘어난다(한 차선=좌·우 2군집 -> 분기=각 브랜치의 안/바깥선으로 3~4군집). 또는
        가장 바깥 두 라인 간격(span)이 한 차선보다 훨씬 넓어진다. 둘 중 하나면 분기로 본다.
        컬럼별 세로픽셀 수를 세어, '라인' 컬럼의 연속 런(run) 개수와 바깥 span 으로 판정한다.
        decision 의 turn_latch 해제(도로 재수렴 = fork False) 기준으로만 쓰인다."""
        h_m = mask.shape[0]
        y1 = int(h_m * self.fork_scan_top_ratio)
        y2 = int(h_m * self.fork_scan_bottom_ratio)
        if y2 <= y1:
            return False
        band = mask[y1:y2, :]
        band_h = y2 - y1
        col_on = np.count_nonzero(band, axis=0) >= max(1, int(band_h * self.fork_col_min_ratio))
        if not bool(col_on.any()):
            return False
        # 연속 'on' 컬럼 런(=라인 군집) 개수
        groups = int(np.count_nonzero(np.diff(col_on.astype(np.int8)) == 1))
        if col_on[0]:
            groups += 1
        on_idx = np.nonzero(col_on)[0]
        span = float(on_idx[-1] - on_idx[0])
        return bool(groups >= self.fork_min_groups or span >= self.fork_span_ratio * w)

    def _mean_x(self, half_mask, x_offset, min_pixels=None):
        # min_pixels=None keeps the legacy threshold; guided windows are narrow so
        # they pass their own (lower) guide_min_pixels.
        xs = np.nonzero(half_mask)[1]
        if xs.size < (self.min_pixels if min_pixels is None else min_pixels):
            return None
        return float(np.mean(xs)) + x_offset

    def _draw_debug(self, roi, mask, lane_center, offset=0.0, yellow_ratio=0.0,
                    curvature=0.0, band_windows=(), roi_raw=None,
                    near_cx=None, la_cx=None):
        """Build the debug image the monitor shows on /perception/lane/debug.

        Bottom (always): the ANALYSED ROI (bird-eye warped if on) with the HSV mask
        pixels (cyan), guided search windows (magenta) + found centres (orange), the
        detected lane centre (red) and image centre (green), plus status text.

        Top (only when bird-eye is on): the RAW pre-warp ROI with the bird-eye SOURCE
        quad (yellow) drawn on it, so you can tune birdeye_src_ratio by eye instead of
        guessing from the warped result alone."""
        dbg = roi.copy()
        dbg[mask > 0] = (255, 255, 0)                                     # HSV-selected lane pixels (cyan)
        for x0, y0, x1, y1, bcx in band_windows:                          # guided search windows
            cv2.rectangle(dbg, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (255, 0, 255), 1)
            if bcx is not None:
                cv2.circle(dbg, (int(bcx), int((y0 + y1) / 2)), 2, (0, 165, 255), -1)
        cx = int(max(0, min(dbg.shape[1] - 1, lane_center)))
        cv2.line(dbg, (cx, 0), (cx, dbg.shape[0]), (0, 0, 255), 2)        # detected lane center (red)
        cv2.line(dbg, (dbg.shape[1] // 2, 0), (dbg.shape[1] // 2, dbg.shape[0]), (0, 255, 0), 1)  # image center (green)
        if self.use_lookahead_control:
            dh = dbg.shape[0]
            if near_cx is not None:   # near band centre = blue tick (bottom third)
                nx = int(max(0, min(dbg.shape[1] - 1, near_cx)))
                cv2.line(dbg, (nx, dh * 2 // 3), (nx, dh), (255, 0, 0), 1)
            if la_cx is not None:     # look-ahead centre = orange tick (top third)
                lx = int(max(0, min(dbg.shape[1] - 1, la_cx)))
                cv2.line(dbg, (lx, 0), (lx, dh // 3), (0, 165, 255), 1)
        # status text (frame is only ~320x72: keep the font tiny)
        mode = (f"BEV {'ON' if self.use_birdeye else 'OFF'}  "
                f"GUIDED {'ON' if self.use_guided_band else 'OFF'}  "
                f"LA {'ON' if self.use_lookahead_control else 'OFF'}")
        vals = f"off {offset:+.2f}  yr {yellow_ratio:.2f}  cv {curvature:+.2f}"
        cv2.putText(dbg, mode, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(dbg, vals, (2, dbg.shape[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # BEV on: stack the raw ROI (with the source quad) on top for src tuning.
        if self.use_birdeye and roi_raw is not None and roi_raw.shape == roi.shape:
            src_view = roi_raw.copy()
            h, w = src_view.shape[:2]
            pts = np.int32([(int(self.birdeye_src_ratio[i] * w),
                             int(self.birdeye_src_ratio[i + 1] * h)) for i in range(0, 8, 2)])
            cv2.polylines(src_view, [pts], True, (0, 255, 255), 1)        # BEV source quad (yellow)
            for px, py in pts:
                cv2.circle(src_view, (int(px), int(py)), 2, (0, 255, 255), -1)
            cv2.putText(src_view, 'SRC (raw ROI)', (2, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            return np.vstack([src_view, dbg])
        return dbg
