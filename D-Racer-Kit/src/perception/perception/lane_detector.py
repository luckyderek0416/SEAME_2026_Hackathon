"""OpenCV 차선 감지기 (규칙 기반, 딥러닝 아님).

BGR 프레임을 받아 하단 관심영역(ROI)을 보고, 좌우의 밝은 차선 마킹을
찾아 정규화된 조향 offset 을 반환한다. 여기의 모든 상수는 실제 트랙용
튜닝 노브다.

기본 전략:
  1. 프레임 하단(차 바로 앞의 도로)을 잘라낸다
  2. 밝은 픽셀(흰색 경계선)을 threshold 로 걸러낸다
  3. 좌우 절반마다 밝은 픽셀의 평균 x 를 구한다 -> 두 차선 라인
  4. 차선 중심 = 두 라인의 중점
  5. offset = 그 중심이 이미지 중심에서 얼마나 떨어졌는지, [-1, 1] 범위

라인이 하나만 보이면 그것으로 중심을 추정한다. 트랙에 노란 중앙선이
있다면 두 번째 HSV mask 를 추가해 섞을 수 있다.
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
                 crossline_roi_top_ratio=0.40, crossline_roi_bottom_ratio=0.95,
                 crossline_min_width_ratio=0.20, crossline_min_rows=4,
                 fork_scan_top_ratio=0.0, fork_scan_bottom_ratio=0.5,
                 fork_col_min_ratio=0.15, fork_min_groups=3, fork_span_ratio=0.65,
                 fork_seed_px=90,
                 follow_yellow=True, follow_yellow_ratio=0.03,
                 follow_yellow_exit_white_ratio=1.0, course='in',
                 lane_width_init=0.0):
        self.roi_top_ratio = roi_top_ratio        # 프레임 하단 (1 - ratio) 부분만 사용
        self.bright_thresh = bright_thresh         # gray 모드 흰색 라인 밝기 threshold (0-255)
        self.min_pixels = min_pixels               # 한쪽을 신뢰하기 위한 최소 차선 픽셀 수
        self.single_line_offset = single_line_offset  # 차선 폭을 아직 모를 때의 폴백 offset
        # --- 굴곡 많은 대리석 바닥 트랙에서의 견고한 차선 추종 ---
        self.num_bands = max(1, num_bands)         # 가로 look-ahead band 개수 (커브 선행 감지)
        self.morph_kernel = morph_kernel           # MORPH_OPEN 크기; 반사광 점/점선 조각 제거 (0=끔)
        self.width_ema = width_ema                 # 기억된 차선 폭이 적응하는 속도
        self.smooth_alpha = smooth_alpha           # offset EMA (1=스무딩 없음, 낮을수록 부드러움)
        self._lane_width = float(lane_width_init)  # 기억된 차선 폭(px), 단일 라인 중심 계산용
                                                   # (0=콜드스타트 학습대기; 실측 픽셀폭 넣으면 단선 구간 강건)
        self._offset_ema = 0.0                     # 스무딩된 offset 상태
        # 회전교차로 junction = 진입/진출부의 점선 마킹 (이 트랙에서 링은 실선이고
        # junction 만 점선이다). 사이드 스트립을 따라 차선 라인이 켜졌다 꺼졌다 하는
        # 것으로 점선을 감지한다 (실선은 계속 켜져 있음).
        self.junction_side = junction_side                 # 점선 junction 이 나타나는 쪽
        self.junction_dash_transitions = junction_dash_transitions  # 세로 on/off 변화 수 => 점선
        self.junction_min_row_pixels = junction_min_row_pixels      # 이 값 이상이면 행을 '라인'으로 간주
        self.junction_gap_rows = junction_gap_rows         # 완전 개방 폴백 (라인 있는 행이 적음)
        # 색상 마스킹: 'hsv' 는 흰색 그리고/또는 노란색 라인을 감지 (견고하며 노란
        # 회전교차로 차선을 구분함); 'gray' 는 예전의 밝기 전용 threshold.
        self.mask_mode = mask_mode
        self.use_white = use_white
        self.use_yellow = use_yellow
        self.white_hsv_lo = tuple(white_hsv_lo)
        self.white_hsv_hi = tuple(white_hsv_hi)
        self.yellow_hsv_lo = tuple(yellow_hsv_lo)
        self.yellow_hsv_hi = tuple(yellow_hsv_hi)
        # --- bird-eye view (선택, 기본 OFF = 기존 동작) ---
        # src/dst 는 ROI 크기 대비 0..1 비율의 평탄한 리스트 [x1,y1, x2,y2, x3,y3, x4,y4]
        # (TL,TR,BR,BL) 라서 해상도가 바뀌어도 유효하다.
        self.use_birdeye = use_birdeye
        self.birdeye_src_ratio = (list(birdeye_src_ratio) if birdeye_src_ratio
                                  else [0.25, 0.05, 0.75, 0.05, 0.95, 0.95, 0.05, 0.95])
        self.birdeye_dst_ratio = (list(birdeye_dst_ratio) if birdeye_dst_ratio
                                  else [0.20, 0.00, 0.80, 0.00, 0.80, 1.00, 0.20, 1.00])
        self._bev_matrix = None
        self._bev_key = None       # 캐시된 행렬이 만들어진 기준 (w, h, src, dst)
        self._bev_warned = False
        # --- guided band 탐색 (선택, 기본 OFF = 기존 multi-band) ---
        # 각 band 가 이전 band 중심 주변만 탐색하므로 멀리 있는 진출/분기 라인이
        # 차선 중심을 잡아끌 수 없다 (회전교차로/fork 견고성).
        self.use_guided_band = use_guided_band
        self.guide_margin_px = guide_margin_px
        self.guide_margin_growth_px = guide_margin_growth_px
        self.guide_min_pixels = guide_min_pixels
        self.guide_use_previous_frame = guide_use_previous_frame
        self.guide_max_jump_px = guide_max_jump_px
        self._prev_center = None   # 직전 프레임의 lane_center (guide 시드)
        # --- look-ahead 조향 블렌딩 (부분 pure-pursuit; 기본 OFF) ---
        # 기존 lane_center 는 전체 band 가중 평균이다. 이 옵션을 켜면 중심이
        # near_weight*가장 가까운 band + lookahead_weight*먼 band 가 되어, 차가
        # 커브에 더 일찍 진입하며 돌기 시작한다. 완전한 pure-pursuit 은 아님:
        # decision 의 PID 가 여전히 컨트롤러이고, 우리는 그 입력만 다듬는다.
        self.use_lookahead_control = use_lookahead_control
        self.near_weight = near_weight
        self.lookahead_weight = lookahead_weight
        self.lookahead_band_index = lookahead_band_index   # -1 = 감지된 가장 먼 band
        self.adaptive_lookahead = adaptive_lookahead
        self.curve_lookahead_weight = curve_lookahead_weight  # 급커브에서의 lookahead 가중치
        self.curve_lookahead_thresh = curve_lookahead_thresh  # 급커브로 간주하는 |curvature|
        # --- 노란 가로선 (yellow crossline) 감지 ---
        # 넓은 가로 방향 노란 band 는 회전교차로 진입/진출 fork 근처에서만 나타나므로
        # 위치 신호다. 노란색 전용 mask 에서 감지하며 (결합 차선 mask 에서는 절대 안 함)
        # decision 의 회전교차로 진입/진출 로직에서 추가 표결 하나로만 쓰인다 —
        # 차선 중심 계산은 전혀 건드리지 않는다.
        self.crossline_roi_top_ratio = crossline_roi_top_ratio       # 스캔 창 상단 (ROI 높이의 0..1)
        self.crossline_roi_bottom_ratio = crossline_roi_bottom_ratio # 스캔 창 하단
        self.crossline_min_width_ratio = crossline_min_width_ratio   # 성분의 가로 스팬 최소 비율 (w 기준)
        self.crossline_min_rows = crossline_min_rows                 # (미사용; 옛 행-스캔 버전 하위호환)
        self.crossline_max_angle_deg = 50.0   # 주축이 수평에서 이 각도 이내면 가로선 (대각선 허용)
        self.crossline_min_area_px = 60       # 성분 최소 픽셀 수 (노이즈/점선 dash 필터)
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
        # --- 노란색 우선 추종 (In 코스: 노란 진입 커브/회전교차로 링) ---
        # 노란색이 ROI 의 follow_yellow_ratio 이상이면, 밴드 차선 중심을 흰+노랑 합친
        # 마스크가 아니라 노란색 전용 마스크에서 계산한다 -> 중앙 흰 점선/바깥 흰선에
        # 끌리지 않고 노란 차선만 추종. junction/fork 감지는 합친 마스크 그대로 사용.
        # In 코스 색상 추종 상태머신 (히스테리시스):
        #   WHITE 모드(기본): 흰색 전용 마스크 추종. yellow_ratio >= follow_yellow_ratio
        #     (노란 점선이 조금이라도 보이기 시작) -> YELLOW 모드 진입.
        #   YELLOW 모드: 노란 전용 마스크 추종(흰 점선/실선 완전 무시). WHITE 복귀는
        #     "노란색이 사실상 사라지고"(yellow_ratio < 진입문턱×exit_yellow_frac)
        #     "흰색이 우세"(white > exit_white_ratio × yellow)한 상태가
        #     follow_yellow_exit_frames 프레임 연속일 때만. 진입/해제 조건을 상호배타로
        #     두는 이유: 전환 구간엔 노란 점선(픽셀 적음)+흰 실선(픽셀 많음)이 공존해
        #     "노랑 보임"과 "흰>노랑"이 동시에 참 -> 조건이 겹치면 Y/W 가 매 프레임 튐.
        #     단 drive_mode == 'ROUNDABOUT' 동안은 강제 YELLOW 유지 (링 위에서 바깥
        #     흰 루프가 보여도 튀지 않게).
        # course != 'in' 이면 전체 비활성(기존 합친-마스크 동작; Out 코스에서 노란
        # 갈림길에 끌려가는 것 방지).
        self.follow_yellow = follow_yellow
        self.follow_yellow_ratio = follow_yellow_ratio
        self.follow_yellow_exit_white_ratio = follow_yellow_exit_white_ratio
        self.follow_yellow_exit_yellow_frac = 0.5   # 해제 노랑 문턱 = 진입문턱 × 이 비율
        self.follow_yellow_exit_frames = 10         # 해제 조건 연속 프레임 수 (~0.3s@30fps)
        self.course = course
        self._following_yellow = False   # 현재 YELLOW 모드인지 (프레임 간 유지)
        self._yellow_exit_count = 0      # 해제 조건 연속 카운터
        # 현재 주행 모드(상태머신 상태 문자열). decision 이 publish 하고 lane_node 가
        # 매 프레임 갱신 -> BEV 디버그 화면에 표시. '' 면 표시 안 함(decision 미실행).
        self.drive_mode = ''

    def _build_mask(self, roi):
        """(lane_mask, white_mask, yellow_mask, yellow_ratio, yellow_offset,
        yellow_crossline) 을 반환.

        HSV 모드 = 흰색|노란색 차선 mask; gray 모드 = 밝기 threshold (w/y mask 는 None).
        white_mask/yellow_mask 는 색상별 전용 mask — In 코스 색상 추종 상태머신이
        추종 대상을 고를 때 쓴다 (lane_mask 는 둘을 합친 것; junction/fork 감지용).
        yellow_ratio  = ROI 중 노란색 비율 (노란 회전교차로와 흰색 바깥
                        루프를 구분해 줌).
        yellow_offset = 노란색 무게중심의 정규화된 x, [-1..1] (+ = 노란색이
                        오른쪽에 있음). 노란 브랜치가 어느 쪽에 있든 decision 이
                        그쪽으로 조향할 수 있게 한다 (방향 무관).
        yellow_crossline = 넓은 가로 방향 노란 band 가 시야에 있음 (회전교차로
                        진입/진출 fork 위치 표식). 노란색 전용 mask 에서
                        계산하며 차선 중심에는 절대 영향 없음.
        """
        if self.mask_mode == 'gray':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, self.bright_thresh, 255, cv2.THRESH_BINARY)
            return mask, None, None, 0.0, 0.0, False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        w = hsv.shape[1]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        wmask = None
        ymask = None
        if self.use_white:
            wmask = cv2.inRange(hsv, self.white_hsv_lo, self.white_hsv_hi)
            mask = cv2.bitwise_or(mask, wmask)
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
        return mask, wmask, ymask, yellow_ratio, yellow_offset, yellow_crossline

    def _detect_crossline(self, ymask):
        """노란 '가로선'(정지선) 감지 — 기울기 허용 버전 (연결성분 + 주축 각도).

        옛 행(row) 폭 스캔은 가로선이 화면에 수평으로 보인다고 가정했는데, 커브에서
        비스듬한 헤딩으로 접근하면 가로선이 BEV 에 대각선으로 찍혀 구조적으로 실패했다.
        헤딩이 θ 틀어지면 차선은 수직에서 θ, 가로선은 수평에서 θ 기울므로,
        "수평에 가까운 주축인가"로 판정하면 기울어도 차선과 항상 구분된다:
          - 연결성분의 가로 스팬 >= crossline_min_width_ratio × w  (dash 는 작아서 탈락)
          - 주축의 수평 기준 각도 <= crossline_max_angle_deg       (차선은 수직쪽이라 탈락)
        use_birdeye ON 이면 BEV 워프된 mask 위에서 동작해 원근 무관.
        (crossline_min_rows 는 이 버전에서 미사용 — 하위호환으로만 남김)
        """
        h_m, w_m = ymask.shape[:2]
        y1 = int(h_m * self.crossline_roi_top_ratio)
        y2 = int(h_m * self.crossline_roi_bottom_ratio)
        if y2 <= y1:
            return False
        cross_roi = ymask[y1:y2, :]
        if cv2.countNonZero(cross_roi) < int(self.crossline_min_area_px):
            return False
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cross_roi, connectivity=8)
        min_len = float(w_m) * float(self.crossline_min_width_ratio)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < int(self.crossline_min_area_px):
                continue
            ys, xs = np.nonzero(labels == i)
            xspan = int(xs.max() - xs.min())
            if xspan < 4:          # 세로 줄무늬(가로 퍼짐 없음)는 fit 무의미
                continue
            # 컬럼별 평균 y 로 주축 기울기 추정 (수평 기준). 정지선이 양끝에서
            # 세로 차선과 붙어 한 성분이 돼도, 컬럼 평균은 대부분 정지선 위라 강건.
            cols = xs - xs.min()
            cnt = np.bincount(cols)
            ysum = np.bincount(cols, weights=ys.astype(np.float64))
            valid = cnt > 0
            xcols = np.nonzero(valid)[0].astype(np.float64)
            ymean = ysum[valid] / cnt[valid]
            if xcols.size < 4:
                continue
            slope = float(np.polyfit(xcols, ymean, 1)[0])
            angle_deg = float(np.degrees(np.arctan(abs(slope))))
            # 주축 길이 = 가로 스팬 / cos(기울기): 대각선이라 창에 짧게 걸려도
            # 실제 선 길이로 평가된다 (bbox 폭 기준의 구조적 실패 회피).
            axis_len = float(xspan) * float(np.hypot(1.0, slope))
            if angle_deg <= float(self.crossline_max_angle_deg) and axis_len >= min_len:
                return True
        return False

    def process(self, bgr, draw_debug=True):
        h, w = bgr.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = bgr[roi_top:h, 0:w]

        roi_raw = roi   # 디버그 뷰에서 BEV 소스 영역을 보여줄 수 있게 워프 전 ROI 를 보관
        if self.use_birdeye:
            roi = self._apply_birdeye(roi)   # 실패 시 원본 ROI 로 폴백

        (mask, wmask, ymask, yellow_ratio,
         yellow_offset, yellow_crossline) = self._build_mask(roi)

        # (4) 노이즈 정리: MORPH_OPEN 이 대리석 바닥 반사광 점과 점선의 작은 틈을
        # 제거해, 떠도는 픽셀이 차선 중심을 잡아끌지 않게 한다.
        use_morph = bool(self.morph_kernel and self.morph_kernel > 1)
        k = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
             if use_morph else None)
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k) if use_morph else mask

        # In 코스 색상 추종 상태머신 (히스테리시스; __init__ 주석 참고):
        #   WHITE 모드 = 흰색 전용 추종, 노란색이 조금이라도 보이면 YELLOW 로 진입.
        #   YELLOW 모드 = 노란 전용 추종(흰선 완전 무시), 흰색이 노란색보다 많이
        #   검출되면 WHITE 복귀. ROUNDABOUT 상태 동안은 강제 YELLOW 유지.
        # junction/fork 감지는 아래에서 합친 마스크(clean)를 그대로 쓴다.
        track = clean
        if (self.follow_yellow and self.course == 'in'
                and ymask is not None and wmask is not None):
            if not self._following_yellow:
                if yellow_ratio >= self.follow_yellow_ratio:
                    self._following_yellow = True
                    self._yellow_exit_count = 0
            elif self.drive_mode != 'ROUNDABOUT':
                # WHITE 복귀 = "노랑이 사실상 사라짐" AND "흰 우세" 가 연속 N프레임.
                # 노랑이 조금이라도(진입문턱×frac 이상) 보이는 동안엔 절대 안 풀림 ->
                # 점선 노랑 + 흰 실선 공존 구간에서 Y/W 플리커 방지.
                wcount = int(np.count_nonzero(wmask))
                ycount = int(np.count_nonzero(ymask))
                yellow_gone = yellow_ratio < (self.follow_yellow_ratio
                                              * self.follow_yellow_exit_yellow_frac)
                white_dom = wcount > self.follow_yellow_exit_white_ratio * max(1, ycount)
                if yellow_gone and white_dom:
                    self._yellow_exit_count += 1
                    if self._yellow_exit_count >= int(self.follow_yellow_exit_frames):
                        self._following_yellow = False
                        self._yellow_exit_count = 0
                else:
                    self._yellow_exit_count = 0
            else:
                self._yellow_exit_count = 0   # ROUNDABOUT: 강제 YELLOW 유지
            sel = ymask if self._following_yellow else wmask
            track = cv2.morphologyEx(sel, cv2.MORPH_OPEN, k) if use_morph else sel
        else:
            self._following_yellow = False
            self._yellow_exit_count = 0

        # (1) multi-band look-ahead: ROI 를 가로 band 들(가까움..멂)로 나누고 각각에서
        # 차선 중심을 찾는다. 가까운 band 는 조향에, 먼 band 는 커브 선행 감지에 쓴다.
        roi_h = track.shape[0]
        band_h = max(1, roi_h // self.num_bands)
        band_windows = []   # guided 모드 탐색 창, 디버그 오버레이용
        if self.use_guided_band:
            bands, near_lanes, band_windows = self._guided_bands(track, w, roi_h, band_h)
        else:
            bands = []   # (center_x, weight, band_index)  band 0 = 가장 가까움 (ROI 하단)
            near_lanes = 0
            for i in range(self.num_bands):
                y1 = roi_h - i * band_h
                y0 = max(0, roi_h - (i + 1) * band_h)
                if y1 - y0 < 2:
                    continue
                cx, nlanes = self._band_center(track[y0:y1, :], w)
                if cx is not None:
                    bands.append((cx, float(self.num_bands - i), i))  # 가까울수록 -> 가중치 큼
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
            # (2) curvature = 차선이 가까운 band 에서 먼 band 로 얼마나 흘러갔는지, 정규화됨.
            # 부호 = 커브 방향, 크기 = 급한 정도 (decision 이 감속하는 데 사용).
            curvature = self._estimate_curvature(bands, w)
            # 부분 pure-pursuit: 전체 band 평균 대신 가장 가까운 band + look-ahead band
            # 중심을 블렌딩한다. OFF(기본값)면 위의 lane_center 를 그대로 유지.
            near_cx = bands[0][0]                    # 감지된 가장 가까운 band
            la_cx = self._lookahead_center(bands)
            if self.use_lookahead_control and la_cx is not None:
                nw, lw = self.near_weight, self.lookahead_weight
                if self.adaptive_lookahead and abs(curvature) >= self.curve_lookahead_thresh:
                    lw = self.curve_lookahead_weight  # 급커브: 더 멀리 내다봄
                    nw = max(0.0, 1.0 - lw)
                if nw + lw > 0.0:
                    lane_center = (nw * near_cx + lw * la_cx) / (nw + lw)

        raw_offset = float(max(-1.0, min(1.0, (lane_center - mid) / (w / 2.0))))
        # (5) 시간축 스무딩(EMA)으로 프레임 간 조향 떨림을 줄인다.
        if lane_found:
            self._offset_ema = (self.smooth_alpha * raw_offset
                                + (1.0 - self.smooth_alpha) * self._offset_ema)
            self._prev_center = lane_center   # 다음 프레임을 위한 guide 시드
        offset = float(self._offset_ema)

        junction = self._detect_junction(clean, w)
        fork = self._detect_fork(clean, w)

        # 디버그 이미지는 웹 대시보드용일 뿐 주행에 안 쓰인다. 프레임을 띄엄띄엄
        # (lane_node 의 debug_hz) 보낼 때 그리기+인코딩을 통째로 건너뛰어 보드 부하를 던다.
        # 노란 추종 중이면 실제 추종 대상인 track(노란 전용)을 그려 튜닝에 도움 준다.
        debug = (self._draw_debug(roi, track, lane_center, offset, yellow_ratio,
                                  curvature, band_windows, roi_raw, near_cx, la_cx)
                 if draw_debug else None)
        return (lane_found, offset, num_lanes, junction, yellow_ratio, yellow_offset,
                curvature, yellow_crossline, fork, debug)

    def _band_center(self, band, w):
        """가로 band 하나의 차선 중심. 전체 폭을 탐색하고 고정된 이미지
        중앙에서 좌/우를 나눈다 (기존 동작)."""
        mid = w // 2
        lx = self._mean_x(band[:, 0:mid], 0)
        rx = self._mean_x(band[:, mid:w], mid)
        return self._combine_lr(lx, rx, w)

    def _combine_lr(self, lx, rx, w):
        """좌/우 라인의 x 위치를 결합해 차선 중심을 구한다. (3) 두 라인이 다
        보이면 차선 폭을 기억하고, 하나만 보이면 그 라인에서 반 폭만큼 떨어진
        곳을 중심으로 둔다 — 고정 비율보다 커브에서 훨씬 정확하다."""
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

    # ---------- guided band 탐색 (선택) ----------
    def _guided_bands(self, clean, w, roi_h, band_h):
        """각 band 가 이전 band 중심 주변(± margin)만 탐색하는 multi-band 중심.
        _band_center 의 좌/우 분할과 차선 폭 기억을 그대로 유지하되 창만
        제한하므로, 멀리 있는 진출/분기 라인이 중심을 잡아끌 수 없다.
        sliding-window/polyfit 피팅이 아님.

        (bands, near_lanes, band_windows) 를 반환하며 bands 는 기존
        [(center_x, weight, band_index), ...] 형식이고 band_windows =
        [(x0, y0, x1, y1, cx_or_None), ...] 는 디버그 오버레이용이다."""
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
                # 아직 guide 없음: 전체 폭 탐색 (band 0 이거나, 지금까지 모든 band 가 비었을 때)
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
                    if abs(jump) > self.guide_max_jump_px:   # 갑작스러운 중심 점프를 클램프
                        cx = guide + (self.guide_max_jump_px if jump > 0
                                      else -self.guide_max_jump_px)
                bands.append((cx, float(self.num_bands - i), i))  # 가까울수록 -> 가중치 큼
                if i == 0:
                    near_lanes = nlanes
                guide = cx
            band_windows.append((x0, y0, x1, y1, cx))
        return bands, near_lanes, band_windows

    def _windowed_center(self, band, w, guide, band_index):
        """[guide-margin, guide+margin] 범위로 제한된 _band_center. 좌/우 분할
        지점이 고정된 w//2 대신 guide 중심이다."""
        margin = self.guide_margin_px + band_index * self.guide_margin_growth_px
        x0 = int(max(0, min(w - 2, guide - margin)))
        x1 = int(max(x0 + 2, min(w, guide + margin)))
        mid = int(max(x0 + 1, min(x1 - 1, round(guide))))
        lx = self._mean_x(band[:, x0:mid], x0, self.guide_min_pixels)
        rx = self._mean_x(band[:, mid:x1], mid, self.guide_min_pixels)
        cx, nlanes = self._combine_lr(lx, rx, w)
        return x0, x1, cx, nlanes

    # ---------- bird-eye view (선택) ----------
    def invalidate_birdeye_cache(self):
        """원근 변환 행렬을 강제로 다시 만들게 한다 (src/dst 변경 후 호출)."""
        self._bev_key = None
        self._bev_warned = False

    def _apply_birdeye(self, roi):
        """ROI 를 탑다운 뷰로 워프한다 (같은 크기). 어떤 실패에서도 입력 ROI 를
        그대로 반환해, 잘못된 점 때문에 차선 추종이 죽는 일이 없게 한다."""
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
        """look-ahead band 의 중심 x: 이번 프레임에 lookahead_band_index band 가
        감지됐으면 그것을, 아니면 감지된 가장 먼 band 를 쓴다 (-1 = 항상 가장
        먼 것). bands 는 가까운 것 -> 먼 것 순서다."""
        if not bands:
            return None
        if self.lookahead_band_index >= 0:
            for cx, _, i in bands:
                if i == self.lookahead_band_index:
                    return cx
        return bands[-1][0]

    @staticmethod
    def _estimate_curvature(bands, w):
        """감지된 가장 먼 band 와 가장 가까운 band 사이의 정규화된 횡방향 드리프트."""
        if len(bands) < 2:
            return 0.0
        near = min(bands, key=lambda b: b[2])
        far = max(bands, key=lambda b: b[2])
        return float(max(-1.0, min(1.0, (far[0] - near[0]) / (w / 2.0))))

    def _detect_junction(self, mask, w):
        """junction = 회전교차로 진입/진출부의 '점선' 마킹.

        사이드 스트립을 따라 각 행에 '라인 존재' 여부를 표시한다. 실선은 (거의)
        모든 행에 존재 -> on/off 변화가 ~0회. 점선은 존재/부재가 번갈아 나타남
        -> 변화가 많음. 따라서 `transitions` 로 점선과 실선을 구분할 수 있다.
        완전 개방(존재하는 행이 아주 적음)도 junction 으로 처리한다 (폴백).
        """
        strip = max(1, w // 3)
        side = mask[:, 0:strip] if self.junction_side == 'left' else mask[:, w - strip:w]
        row_present = (np.count_nonzero(side, axis=1) >= self.junction_min_row_pixels).astype(np.uint8)
        present_rows = int(row_present.sum())
        transitions = int(np.count_nonzero(np.diff(row_present)))  # 스트립을 따라 내려가며 on/off 변화 수
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
        # min_pixels=None 이면 기존 threshold 를 유지; guided 창은 좁아서
        # 자체적인 (더 낮은) guide_min_pixels 를 넘겨준다.
        xs = np.nonzero(half_mask)[1]
        if xs.size < (self.min_pixels if min_pixels is None else min_pixels):
            return None
        return float(np.mean(xs)) + x_offset

    def _draw_debug(self, roi, mask, lane_center, offset=0.0, yellow_ratio=0.0,
                    curvature=0.0, band_windows=(), roi_raw=None,
                    near_cx=None, la_cx=None):
        """모니터가 /perception/lane/debug 에서 보여주는 디버그 이미지를 만든다.

        하단 (항상): 분석에 쓰인 ROI (bird-eye 켜져 있으면 워프된 것) 위에 HSV mask
        픽셀(시안), guided 탐색 창(마젠타) + 찾은 중심(주황), 감지된 차선 중심(빨강)과
        이미지 중심(초록), 그리고 상태 텍스트를 그린다.

        상단 (bird-eye 켜졌을 때만): 워프 전 원본 ROI 위에 bird-eye 소스 사각형
        (노랑)을 그려, 워프 결과만 보고 추측하는 대신 눈으로 보며
        birdeye_src_ratio 를 튜닝할 수 있게 한다."""
        dbg = roi.copy()
        dbg[mask > 0] = (255, 255, 0)                                     # HSV 로 선택된 차선 픽셀 (시안)
        for x0, y0, x1, y1, bcx in band_windows:                          # guided 탐색 창
            cv2.rectangle(dbg, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (255, 0, 255), 1)
            if bcx is not None:
                cv2.circle(dbg, (int(bcx), int((y0 + y1) / 2)), 2, (0, 165, 255), -1)
        cx = int(max(0, min(dbg.shape[1] - 1, lane_center)))
        cv2.line(dbg, (cx, 0), (cx, dbg.shape[0]), (0, 0, 255), 2)        # 감지된 차선 중심 (빨강)
        cv2.line(dbg, (dbg.shape[1] // 2, 0), (dbg.shape[1] // 2, dbg.shape[0]), (0, 255, 0), 1)  # 이미지 중심 (초록)
        if self.use_lookahead_control:
            dh = dbg.shape[0]
            if near_cx is not None:   # 가까운 band 중심 = 파란 눈금 (하단 1/3)
                nx = int(max(0, min(dbg.shape[1] - 1, near_cx)))
                cv2.line(dbg, (nx, dh * 2 // 3), (nx, dh), (255, 0, 0), 1)
            if la_cx is not None:     # look-ahead 중심 = 주황 눈금 (상단 1/3)
                lx = int(max(0, min(dbg.shape[1] - 1, la_cx)))
                cv2.line(dbg, (lx, 0), (lx, dh // 3), (0, 165, 255), 1)
        # 상태 텍스트 (프레임이 겨우 ~320x72: 폰트를 아주 작게 유지)
        follow_tag = ''
        if self.follow_yellow and self.course == 'in':
            follow_tag = '  FOLLOW-Y' if self._following_yellow else '  FOLLOW-W'
        mode = (f"BEV {'ON' if self.use_birdeye else 'OFF'}  "
                f"GUIDED {'ON' if self.use_guided_band else 'OFF'}  "
                f"LA {'ON' if self.use_lookahead_control else 'OFF'}"
                f"{follow_tag}")
        vals = f"off {offset:+.2f}  yr {yellow_ratio:.2f}  cv {curvature:+.2f}"
        # 현재 주행 모드를 크게 맨 위에 표시. decision 미실행이면 생략.
        # 색상 추종 상태를 함께 표기: [Y]=노란선 추종(노란 글씨) / [W]=흰선 추종(흰 글씨)
        # -> 대시보드에서 "DRIVE 인데 어느 색을 따라가는 중인지" 한눈에 구분.
        y_mode = 10
        if self.drive_mode:
            txt = f"MODE: {self.drive_mode}"
            color = (0, 255, 255)
            if self.follow_yellow and self.course == 'in':
                txt += ' [Y]' if self._following_yellow else ' [W]'
                color = (0, 255, 255) if self._following_yellow else (255, 255, 255)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(dbg, (0, 0), (tw + 6, th + 8), (0, 0, 0), -1)
            cv2.putText(dbg, txt, (3, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_mode = th + 18
        cv2.putText(dbg, mode, (2, y_mode), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(dbg, vals, (2, dbg.shape[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # BEV 켜짐: src 튜닝을 위해 (소스 사각형이 그려진) 원본 ROI 를 위에 쌓는다.
        if self.use_birdeye and roi_raw is not None and roi_raw.shape == roi.shape:
            src_view = roi_raw.copy()
            h, w = src_view.shape[:2]
            pts = np.int32([(int(self.birdeye_src_ratio[i] * w),
                             int(self.birdeye_src_ratio[i + 1] * h)) for i in range(0, 8, 2)])
            cv2.polylines(src_view, [pts], True, (0, 255, 255), 1)        # BEV 소스 사각형 (노랑)
            for px, py in pts:
                cv2.circle(src_view, (int(px), int(py)), 2, (0, 255, 255), -1)
            cv2.putText(src_view, 'SRC (raw ROI)', (2, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            return np.vstack([src_view, dbg])
        return dbg
