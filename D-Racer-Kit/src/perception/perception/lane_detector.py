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
                 mask_mode='hsv', use_white=True, use_yellow=True, use_red=True,
                 white_hsv_lo=(0, 0, 180), white_hsv_hi=(179, 60, 255),
                 yellow_hsv_lo=(18, 80, 80), yellow_hsv_hi=(38, 255, 255),
                 red_hsv_lo1=(0, 80, 60), red_hsv_hi1=(10, 255, 255),
                 red_hsv_lo2=(170, 80, 60), red_hsv_hi2=(179, 255, 255),
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
                 fork_col_min_ratio=0.15, fork_min_groups=3, fork_span_ratio=0.0,
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
        self._lane_width_init = float(lane_width_init)  # EMA 학습 하한 가드 기준 (0=가드 없음)
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
        # 빨간 노면 검출 (ArUco 장애물이 놓이는 빨간 도로 구간). 빨강은 HSV 색상환의
        # 양끝(H≈0, H≈179)에 걸쳐 있어 두 구간을 따로 잡아 합친다. 차선 mask 에는
        # 합치지 않으므로 조향/차선중심에는 전혀 영향이 없다.
        self.use_red = use_red
        self.red_hsv_lo1 = tuple(red_hsv_lo1)
        self.red_hsv_hi1 = tuple(red_hsv_hi1)
        self.red_hsv_lo2 = tuple(red_hsv_lo2)
        self.red_hsv_hi2 = tuple(red_hsv_hi2)
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
        # 최종 중심 연속성 가드 (07-11 run9 실측): 분기 목에서 점선(한쪽)+실선(반대쪽)이
        # 좌/우 경계로 잡히는 순간 nl=2 중점이 직전 중심에서 한 프레임에 반 차폭 가까이
        # 점프해 (raw -0.57 -> -0.21 -> +0.05) 조향을 분기 쐐기로 끌고 갔다. 차선 중심은
        # 물리적으로 프레임당 그만큼 못 움직이므로, 최종 lane_center 의 프레임당 이동을
        # (반 차폭 x 이 비율)로 제한한다. 단선 경로의 연속성 분류(_combine_lr)와 같은
        # 원리를 두-선 중점 경로까지 확장한 것. 밴드별 중심이 아닌 최종 중심에만 걸어
        # 곡률 추정(밴드 간 흐름)은 건드리지 않는다. 0 = off.
        # 트레이드오프: 차선 소실 후 재획득 시 수렴이 프레임당 이 한도로 느려진다.
        # 07-11 run10/11 실측: 0.10 = 쐐기 당김 동결 성공 / 0.15 = 쐐기 당김(프레임당
        # 0.05~0.08)이 상한 아래로 통과해 좌커밋을 반납 -> 0.10 확정. 좌커밋 깊이는
        # 가드가 아니라 진입 병합(1L) 모드가 담당한다.
        self.center_jump_max_ratio = 0.10
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
        # 직선성 검사 (곡선 실선의 수평 구간 오인 방지): 선형 피팅 잔차가
        # max_resid_px 이하인 컬럼 비율이 min_inlier_frac 이상이어야 정지선.
        # (구 컬럼평균 피팅 잔여 파라미터 — Hough 방식에서 미사용, 하위호환용)
        self.crossline_max_resid_px = 2.5
        self.crossline_min_inlier_frac = 0.75
        self.crossline_max_col_px = 0
        # HoughLinesP: 누적 투표 임계 / 같은 직선으로 이어붙일 최대 간격(px).
        self.crossline_hough_thresh = 25
        # 07-10 실측: gap<=8 이면 진짜 정지선(실선)도 마스크의 작은 구멍 때문에 이어지지
        # 않아 검출률 0% 가 된다. gap=10 이 진짜 정지선을 100% 잡으면서 점선은 잇지 않는
        # 지점이었다 (점선 구간 정지 측정 12초, 긴 선분 0개).
        self.crossline_hough_max_gap = 10
        # 선분 위 픽셀 충실도 하한. 이어붙인 점선을 배제하는 값싼 안전망.
        # 실측: 진짜 정지선 0.86~1.00, 점선을 이어붙인 선분 0.52~0.73.
        # (다만 링 안의 '진짜 실선'이 직각으로 보이는 경우는 이걸로 못 거른다 — 그건
        #  상태머신의 gate_blank_s 가 막아야 한다.)
        self.crossline_min_solidity = 0.80
        # True 면 첫 채택 후에도 모든 선분을 진단 목록에 남긴다 (임계 튜닝용).
        self.crossline_debug_all = False
        # 수직성 게이트: 정지선은 BEV 에서 차선과 직교한다. 차선 주축을 dx/dy = a_h 라
        # 하면 차선 방향은 (a_h, 1), 정지선 방향은 (1, slope) 이고 직교 조건은
        #   a_h + slope = 0  =>  slope = -a_h
        # 반면 노란 '차선' 성분을 잡으면 그 컬럼평균 기울기는 1/a_h 가 나오므로,
        # 1/a_h = -a_h (a_h^2 = -1) 는 불가능 -> 어떤 헤딩에서도 원리적으로 구분된다.
        # (07-10 실측: 헤딩 43° 좌커브에서 slope=+1.07 = 1/0.93 인 '차선'이 정지선으로
        #  오검출돼 RA 오진입. 절대각 임계 50° 로는 못 거른다.)
        # 0 이하면 게이트 비활성(구버전 동작).
        # 07-10: 절대 기울기차 |slope + a_h| 는 헤딩이 클수록 허용각이 좁아지는 결함이
        # 있어(비스듬히 접근하는 곡선 위 정지선에 불리) 각도 공간으로 바꿨다.
        # 판정은 정규화된 |cos(차선, 선분)| <= sin(perp_tol_deg).
        # 차선을 선분으로 잘못 뽑으면 slope = 1/a_h 이므로 cos 이 a_h 와 무관하게 정확히
        # 1 (평행) 이 되어, 어떤 헤딩에서도 통과할 수 없다.
        self.crossline_perp_tol_deg = 20.0
        # BEV 가로/세로 스케일비 sx/sy. 지면 직각을 BEV 에서 판정하려면 필요.
        # 07-10 실측: r^2 = 3.63 -> r = 1.91. birdeye_dst_ratio 를 바꾸면 재측정 필요.
        self.crossline_bev_aspect = 1.91
        self.lane_heading_alpha = 0.2      # a_h EMA 계수 (폴백 경로 전용)
        self._lane_heading_init = False
        # 후보 선분에서 이 거리(px) 이내 픽셀은 차선 헤딩 피팅에서 제외한다.
        self.crossline_exclude_px = 6.0
        # 직전 프레임 track 의 crossline 창 픽셀 (ys, xs). _detect_crossline 이
        # _build_mask 안에서 track 보다 먼저 호출되므로 1프레임 지연값을 쓴다.
        self._crossline_track_pts = None
        self.crossline_perp_tol = 0.35   # (구 절대값 임계 — 미사용, 하위호환)
        # 직전 프레임의 차선 주축 기울기(dx/dy). _detect_crossline 은 track 이 만들어지기
        # 전에 호출되므로 1프레임 지연값을 쓴다 (20Hz 에서 50ms, 무시 가능).
        self._lane_heading = 0.0
        # 임시 진단: _detect_crossline 이 마지막 프레임에 평가한 후보 목록.
        self.last_crossline_cands = []
        # --- 좌/우 갈림길 (fork) 감지 + 브랜치 선택 ---
        # 분기 구간은 도로가 두 브랜치로 갈라져, BEV 상단 스캔밴드에서 세로 라인 군집이
        # 3개 이상(각 브랜치 안/바깥선)이거나 바깥 라인 간격이 한 차선보다 훨씬 넓어진다.
        # 이 fork 플래그는 decision 의 turn_latch 해제(도로 재수렴) 기준으로만 쓰이고,
        # 차선 중심 계산은 건드리지 않는다.
        self.fork_scan_top_ratio = fork_scan_top_ratio        # 스캔밴드 상단 (BEV far)
        self.fork_scan_bottom_ratio = fork_scan_bottom_ratio  # 스캔밴드 하단
        self.fork_col_min_ratio = fork_col_min_ratio          # 컬럼이 '라인'으로 세는 세로픽셀 비율
        self.fork_min_groups = fork_min_groups                # 라인 군집 이 개수 이상 => 분기
        # 바깥라인 간격 폴백. 0 이하면 비활성 (기본 비활성).
        # 07-10 실측: BEV 가 잘못 캘리된 상태(구 birdeye_src_ratio)에서는 평범한 직선에서도
        # span 이 0.85w 라 임계 0.65 를 항상 넘겨 fork 가 96% 오발화했고, 그 탓에
        # state_machine 의 reconverged(= _fork_seen and not fork) 가 영영 False 가 되어
        # turn_latch 가 fork_hold_s(8초) 타임아웃으로만 풀렸다.
        # BEV 재캘리 후에는 span 이 0.60~0.65w 로 내려와 임계 0.65 에서 오발화가 사라진다.
        # 그러나 직선(0.60~0.65w)과 임계 사이 여유가 0~8% 뿐이라 분기 판별력이 없다.
        # groups>=fork_min_groups 만으로 충분하므로(직선에서 0% 오발화) 폴백은 끈다.
        self.fork_span_ratio = fork_span_ratio
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
        # 진입 병합(1L) 모드 (07-11 run9~11 실측 후 팀 합의): 진입 좌회전 중
        # "안쪽 실선이 시야에서 사라진 순간"(dash 폴백 진입)부터 N프레임 동안,
        # 밴드의 좌/우 분할을 끄고 노란 픽셀 전체 평균을 "선 하나"로 취급해
        # 항상 우측(바깥) 경계로 분류: 중심 = 선 - 반차폭.
        # 근거: 인코스 분기는 항상 좌회전이고, 안쪽 실선이 사라진 뒤 보이는 노란
        # 선은 구조상 바깥선(점선부+실선부가 이어진 하나의 경계)이다. 이 점선부와
        # 실선부가 별개 클러스터로 잡혀 nl=2 중점(+0.05)이 조향을 분기 쐐기로
        # 끌던 것(run9/11 동일 서명)을 분할 자체를 없애 원천 차단.
        # 발동 시점 주의 (07-11 run14): 래치 '순간'에 무장하면 아직 잘 보이는
        # 안쪽 실선까지 바깥선으로 오인해 그 왼쪽(도로 밖 안쪽)을 겨냥, 코너를
        # 깎으며 안쪽 선을 넘는다 (off -0.82 실측). 안쪽 실선이 보이는 동안은
        # 기존 단선 연속성 분류가 좌측 경계로 옳게 처리하므로 그대로 두고,
        # 실선 소실 순간에만 무장한다. 래치당 1회 (폴백 깜빡임 재발동 방지).
        # 07-11 run17: 배터리 열화로 실효 속도가 떨어지자 2초(40) 창이 회전 미완 상태로
        # 만료 -> 인계 직후 쐐기(nl=2 중점, 가드 스킵)에 격추. 속도 편차 흡수를 위해 4초로.
        self.entry_oneline_frames = 80   # ~4s@20fps. 0=off
        # 1L 창에서 사용할 하단(가까운) 밴드 수 (07-12 run40 디버그 프레임 실증):
        # 전체 밴드 평균은 시야 상단에 들어온 링의 원거리 호들까지 노란 질량으로
        # 흡수해 평균이 왼쪽으로 오염됨 -> 목표 = 평균-반차폭이 화면 좌단에 박혀
        # 급좌 다이브 (off -0.75, 인코스 안쪽 이탈 반복). 가까운 밴드에는 진입
        # 점선만 있으므로 하단 N개로 제한하면 원인(카메라 피치/환경)과 무관하게
        # 오염이 구조적으로 차단된다.
        self.oneline_near_bands = 2
        self._oneline_left = 0
        self._oneline_used = False       # 이번 래치에서 이미 무장했는지
        self._ra_seen = False            # ROUNDABOUT 을 겪은 뒤에는 재무장 안 함
        # 개구부 1L 창 (07-12, run27~29 영상+텔레메트리 실증 / IMG_4129 설계):
        # RA 진입·해제 '전이 엣지'에서 위 1L 기계를 재무장한다. 개구부의 두 점선
        # 열이 nl=2 로 잡히면 중점이 열 사이 쐐기(도로 밖)를 겨냥 — 분기 진입
        # 쐐기(run9)와 동일 메커니즘, 최근 5회 통과 중 4회 이탈 (run29 v39 프레임).
        # 1L 은 노란 질량 전체 = 우측(바깥) 경계 하나로 봐서 쐐기가 원천 차단됨.
        # 진입 창 = 빨간 점선(개구부→링), 해제 창 = 파란 점선(링→탈출로).
        # 07-12 run31 A/B 뒤 사용자 결정: 전이 창 2개는 0(off), 대신 아래
        # ra_opening_oneline_frames(실선 소실 트리거)로 랩 종료 개구부를 덮는다.
        self.ra_entry_oneline_frames = 100   # ~5s@20fps. RA 진입 전이. 0=off
        self.ra_exit_oneline_frames = 60     # ~3s@20fps. RA->DRIVE 해제 전이. 0=off
        self._prev_drive_mode = ''
        # (미사용 — 07-12 run32 로 폐기) 개구부 1L 실선 소실 트리거. 링 중간에서
        # 실선이 FOV 를 벗어나는 순간에도 오발했고, 노란 질량 극소 상태의
        # "전체평균-반차폭" 추적이 안쪽 나선(양성 피드백)이 됨 (off -0.65, 섬 안
        # 완전 이탈). 대체 = "RA 점선 폴백 금지" (process() 의 sel 결정부 주석).
        self.ra_opening_oneline_frames = 0   # 0=off (폐기)
        # YELLOW 추종 중 점선(dash) 무시: 조향용 track 마스크에서 세로 길이가 짧은
        # 노란 성분(점선 dash, 그리고 가로 정지선)을 제거해 "실선만" 추종한다.
        # raw ymask 는 yellow_ratio/crossline/Y모드 진입판정에 그대로 쓰이므로 무관.
        self.filter_yellow_dashes = True
        self.yellow_solid_min_h_ratio = 0.30   # ROI 높이의 이 비율 이상 세로로 이어져야 "실선"
        # 실선 소실 폴백: 급좌회전에선 안쪽 실선이 단일 전방 카메라 시야에서
        # 빠져나가 실선 track 이 비고, 그대로 두면 lane_found=False -> 직진 폴백으로
        # 커브(회전교차로 진입로)를 통째로 놓친다. 실선 픽셀이 이 문턱 미만이면
        # 그 프레임은 점선 포함 raw ymask 로 추종을 유지한다 (보이는 유일한
        # 노란 단서가 바깥 점선인 구간 대응).
        self.yellow_dash_fallback_px = 120
        # 폴백 해제 히스테리시스: 폴백 진입은 즉시(차선 소실 대응), 해제(실선
        # 전용 복귀)는 실선이 이 프레임 수 연속 문턱 이상일 때만. 실선이 화면
        # 구석에 막 걸친 순간 점선을 뚝 버려 추종 대상이 사라지는 급전환 방지 —
        # 전환 구간 동안 track 에 점선+실선이 공존해 두-선 중점으로 부드럽게 이어짐.
        self.dash_fallback_exit_frames = 30   # 1s@30fps (07-08 트랙: 0.33s 는 너무 짧음)
        self._dash_fallback_on = False
        self._solid_ok_count = 0
        # 헤어핀(급커브) 헤딩 보정 게인: FOLLOW-Y 주행 중 track 주축이 수직에서
        # 기울어진 만큼(dx/dy) offset 에 조향 보정을 더한다. 도로가 BEV 에서 옆으로
        # 누우면 "선 옆 half-width = 중심" 가로 계산이 무효가 되는 것에 대한 대응.
        # 0=off. ROUNDABOUT 은 제외(링 유지 편향과 중복 방지).
        self.yellow_heading_gain = 0.4
        # 단선 좌/우 분류용 점선 힌트 (FOLLOW-Y 전용, 프레임마다 갱신):
        # 실선은 항상 점선의 반대편 경계이므로, 실선 하나만 보일 때 화면(분할점)
        # 기준이 아니라 "점선 대비 어느 쪽인가"로 좌/우를 정한다. 오른쪽 실선이
        # 화면 왼쪽에 재등장해도 왼쪽 경계로 오분류하지 않는다. None = 힌트 없음.
        self._yellow_dash_cx = None
        # 현재 주행 모드(상태머신 상태 문자열). decision 이 publish 하고 lane_node 가
        # 매 프레임 갱신 -> BEV 디버그 화면에 표시. '' 면 표시 안 함(decision 미실행).
        self.drive_mode = ''
        # 합류부 창(yaw 위치창) 활성 플래그. decision 이 /decision/merge_zone 으로
        # publish -> "단선=우측 경계" 규칙을 이 창 안으로만 스코프 (07-11 run25 폐기
        # 후 사용자 지정: RA 전체 적용은 과범위).
        self.merge_zone = False
        # 재획득 우측 고정 모드 (07-11 사용자 지정 조건): 창 안에서 "소실(0) -> 단선(1)"
        # 전이로 잡힌 선에만 우측 경계 고정을 적용한다. 이미 연속 추적 중이던 단선은
        # 기존 연속성 분류 그대로. 두 선(nl=2)이 보이면 기하가 복원된 것이므로 해제.
        self._reacq_right_active = False
        self._prev_found = False   # 직전 프레임 lane_found
        self._prev_nl = 0          # 직전 프레임 num_lanes

    def _build_mask(self, roi):
        """(lane_mask, white_mask, yellow_mask, yellow_ratio, yellow_offset,
        yellow_crossline, red_ratio) 을 반환.

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
        red_ratio     = ROI 중 빨간 노면 비율. 빨간 도로(ArUco 장애물 구간)에 진입했는지
                        판단하는 신호. 차선 mask 에 합치지 않으므로 조향에 영향 없음.
        """
        if self.mask_mode == 'gray':
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, self.bright_thresh, 255, cv2.THRESH_BINARY)
            return mask, None, None, 0.0, 0.0, False, 0.0
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
        # 빨간 노면: HSV 색상환에서 빨강은 H=0 과 H=179 양끝으로 갈라지므로 두 구간을 합친다.
        # 차선 mask 와 무관하게 비율만 재며, 조향에는 절대 영향을 주지 않는다.
        red_ratio = 0.0
        if self.use_red:
            rmask = cv2.bitwise_or(
                cv2.inRange(hsv, self.red_hsv_lo1, self.red_hsv_hi1),
                cv2.inRange(hsv, self.red_hsv_lo2, self.red_hsv_hi2),
            )
            red_ratio = float(np.count_nonzero(rmask)) / float(rmask.size)
        return mask, wmask, ymask, yellow_ratio, yellow_offset, yellow_crossline, red_ratio

    def _filter_yellow_dashes(self, ymask):
        """YELLOW 추종용 실선 필터: 세로 연장이 짧은 노란 성분을 제거한다.

        점선 dash 는 세로로 짧고(수십 px 미만), 가로 정지선도 세로가 얇다.
        반면 실선 차선은 (커브로 휘어도) ROI 세로의 상당 부분을 관통한다.
        -> bbox HEIGHT >= yellow_solid_min_h_ratio × ROI높이 인 성분만 남긴다.
        조향(track)용으로만 쓰이고, raw ymask 신호(yellow_ratio/crossline/진입판정)는
        건드리지 않는다. 필터 결과가 비면 lane_found=False -> 직진 폴백(기존 동작)."""
        h_m = ymask.shape[0]
        min_h = max(2, int(h_m * float(self.yellow_solid_min_h_ratio)))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(ymask, connectivity=8)
        out = np.zeros_like(ymask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_HEIGHT] >= min_h:
                out[labels == i] = 255
        return out

    @staticmethod
    def _segment_solidity(mask, seg):
        """선분 위 픽셀이 실제로 얼마나 채워져 있는가 (0..1).

        HoughLinesP 는 maxLineGap 안의 조각들을 한 직선으로 이어붙인다. 그래서 회전교차로
        진입/진출부의 '점선' 마킹이 진짜 정지선(실선)과 같은 길이·기울기·직교성을 갖는
        선분으로 잡힌다 (07-10 실측: 오검출 길이 최대 156px > 진짜 정지선 148px).
        기하만으로는 구분할 수 없다. 실선은 선분 위가 빽빽하고 이어붙인 점선은 구멍이
        많으므로, 선분을 따라 샘플링해 mask 가 켜진 비율로 가른다.
        """
        x1, y1, x2, y2 = (float(v) for v in seg)
        n = max(8, int(np.hypot(x2 - x1, y2 - y1)))
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)
        h, w = mask.shape[:2]
        xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
        # 두께 1px 오차를 허용: 위/아래 한 픽셀도 함께 본다.
        hit = (mask[yi, xi] > 0)
        hit |= (mask[np.clip(yi - 1, 0, h - 1), xi] > 0)
        hit |= (mask[np.clip(yi + 1, 0, h - 1), xi] > 0)
        return float(np.mean(hit))

    def _lane_heading_excluding(self, seg, fallback):
        """정지선 후보 선분 seg 의 픽셀을 뺀 뒤 차선 주축 a_h = dx/dy 를 잰다.

        정지선은 track 에서 세로 차선과 한 성분이라 제거되지 않고, 가까워질수록
        차선 피팅을 통째로 끌어당긴다 (07-10 실측: 한 번의 통과 중 a_h 가
        +2.50 -> -1.57 로 부호까지 뒤집혀, 진짜 정지선이 직교오차 59~73° 로 거절됐다).
        후보를 하나 지우고 남은 픽셀로 재면 순환 없이 풀린다:
          - 진짜 정지선을 지우면 남는 건 차선 -> a_h 정확 -> 직각 -> 채택
          - 차선 선분을 지우면 남는 것도 차선 -> 그 선분은 a_h 와 평행 -> 거절
        """
        pts = self._crossline_track_pts
        if pts is None:
            return fallback
        ys, xs = pts
        if xs.size < 150:
            return fallback
        x1, y1, x2, y2 = seg
        dx, dy = float(x2 - x1), float(y2 - y1)
        seg_len = float(np.hypot(dx, dy))
        if seg_len < 1e-6:
            return fallback
        # 점-직선 거리 = |cross((p - p1), dir)| / |dir|
        dist = np.abs((xs - x1) * dy - (ys - y1) * dx) / seg_len
        keep = dist > float(self.crossline_exclude_px)
        if int(keep.sum()) < 150:
            return fallback
        try:
            a = float(np.polyfit(ys[keep].astype(np.float64),
                                 xs[keep].astype(np.float64), 1)[0])
        except Exception:
            return fallback
        return float(np.clip(a, -2.5, 2.5))

    def _detect_crossline(self, ymask):
        """노란 '가로선'(정지선) 감지 — HoughLinesP + 차선 직교성 게이트.

        07-10 실측으로 옛 방식(연결성분의 컬럼평균에 최소제곱 직선 피팅 + 잔차
        인라이어 비율)이 구조적으로 실패함을 확인했다. 정지선은 양끝에서 세로 노란
        차선과 T 자로 붙어 한 성분이 되는데, 그 차선 픽셀이 컬럼평균을 통째로 끌어당겨
        진짜 정지선 앞 313프레임에서 inlier 가 0.05 까지 떨어져 100% 미검출이었다.
        (반대로 비스듬히 누운 '차선' 자체는 잘 맞는 직선이라 항상 채택 -> 회전교차로
         오진입.) 두꺼운 컬럼을 빼는 우회로는 임계가 거리에 종속돼 무너졌다.

        Hough 변환은 오염 픽셀을 미리 지울 필요가 없다 — 투표로 직선을 찾고 나머지는
        그냥 표를 못 얻는다. 실측: 정지선 앞 30/30 프레임 검출(옛 방식 최적값 28/30,
        적응형 0/30).

        판정: 길이 >= crossline_min_width_ratio × w 인 선분 중,
              차선과 직교( |slope + a_h| <= crossline_perp_tol )하는 것이 하나라도 있으면 True.

        직교성이 왜 차선을 원리적으로 배제하는가:
          차선 주축을 dx/dy = a_h 라 하면 차선 방향은 (a_h, 1), 정지선 방향은 (1, slope).
          직교 조건 a_h + slope = 0 => 정지선의 slope = -a_h.
          반면 '차선'을 선분으로 뽑으면 그 slope = 1/a_h 이므로
            perp_err = |1/a_h + a_h| = |(1 + a_h^2)/a_h| >= 2   (산술-기하 평균, 등호는 |a_h|=1)
          부호와 무관하게 항상 2 이상이라 tol 0.35 로는 절대 통과하지 못한다.
        """
        self.last_crossline_cands = []
        h_m, w_m = ymask.shape[:2]
        y1 = int(h_m * self.crossline_roi_top_ratio)
        y2 = int(h_m * self.crossline_roi_bottom_ratio)
        if y2 <= y1:
            return False
        cross_roi = ymask[y1:y2, :]
        if cv2.countNonZero(cross_roi) < int(self.crossline_min_area_px):
            return False

        min_len = float(w_m) * float(self.crossline_min_width_ratio)
        lines = cv2.HoughLinesP(cross_roi, 1, np.pi / 180.0,
                                threshold=int(self.crossline_hough_thresh),
                                minLineLength=int(min_len),
                                maxLineGap=int(self.crossline_hough_max_gap))
        if lines is None or len(lines) == 0:
            return False
        # OpenCV 버전에 따라 (N,1,4) 또는 (N,4) 로 온다.
        lines = np.asarray(lines).reshape(-1, 4)

        # BEV 는 가로/세로 스케일이 달라(sx/sy = r) 지면의 직각이 영상에서 직각이 아니다.
        # 07-10 실측(진짜 정지선 11프레임): r^2 = 3.63, 변동계수 0.08 로 상수 확인.
        # 지면 방향으로 되돌린 뒤 직교를 본다: (dx, dy)_bev -> (dx/r, dy)_ground.
        r = max(1e-3, float(self.crossline_bev_aspect))
        tol_deg = float(self.crossline_perp_tol_deg)
        cos_max = float(np.sin(np.radians(tol_deg))) if tol_deg > 0.0 else 1.0
        fallback_ah = float(self._lane_heading)
        found = False
        for seg in lines:
            x1, yy1, x2, yy2 = seg
            dx = float(x2) - float(x1)
            dy = float(yy2) - float(yy1)
            length = float(np.hypot(dx, dy))
            if length < min_len:
                continue
            # 이 후보를 지우고 남은 픽셀로 차선 방향을 잰다 (순환 회피).
            ah = self._lane_heading_excluding(seg, fallback_ah)
            lane_g = (ah / r, 1.0)
            lane_norm = float(np.hypot(*lane_g))
            # 지면 공간에서의 |cos(차선, 선분)|. 0 = 직교(정지선), 1 = 평행(차선).
            seg_g = (dx / r, dy)
            seg_norm = float(np.hypot(*seg_g))
            if seg_norm < 1e-6:
                continue
            perp_cos = abs(lane_g[0] * seg_g[0] + lane_g[1] * seg_g[1]) / (lane_norm * seg_norm)
            # 실선 판정: 점선을 이어붙인 선분은 구멍이 많아 여기서 탈락한다.
            solidity = self._segment_solidity(cross_roi, seg)
            ok = (perp_cos <= cos_max
                  and solidity >= float(self.crossline_min_solidity))
            slope = (dy / dx) if abs(dx) > 1e-6 else float('inf')
            if ok:
                verdict = 'ACCEPT'
            elif perp_cos > cos_max:
                verdict = 'reject_perp'
            else:
                verdict = 'reject_dashed'
            self.last_crossline_cands.append(
                (round(slope, 3) if np.isfinite(slope) else 'vert',
                 round(float(np.degrees(np.arcsin(min(1.0, perp_cos)))), 1),
                 round(length, 1), round(solidity, 3), verdict,
                 round(ah, 3), round(perp_cos, 3)))
            if ok:
                found = True
                if not self.crossline_debug_all:
                    break                    # 진단이 필요 없으면 첫 채택에서 종료
        return found

    def process(self, bgr, draw_debug=True):
        h, w = bgr.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = bgr[roi_top:h, 0:w]

        roi_raw = roi   # 디버그 뷰에서 BEV 소스 영역을 보여줄 수 있게 워프 전 ROI 를 보관
        if self.use_birdeye:
            roi = self._apply_birdeye(roi)   # 실패 시 원본 ROI 로 폴백

        (mask, wmask, ymask, yellow_ratio,
         yellow_offset, yellow_crossline, red_ratio) = self._build_mask(roi)

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
            if self.drive_mode == 'ROUNDABOUT':
                self._ra_seen = True
            if not self._following_yellow:
                if yellow_ratio >= self.follow_yellow_ratio:
                    self._following_yellow = True
                    self._yellow_exit_count = 0
                    self._oneline_used = False   # 새 래치 -> 1L 1회 사용권 리셋
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
                        self._oneline_left = 0   # 래치 해제 -> 병합 모드도 종료
                else:
                    self._yellow_exit_count = 0
            else:
                self._yellow_exit_count = 0   # ROUNDABOUT: 강제 YELLOW 유지
            if self._following_yellow:
                # 실선만 추종: 점선 dash·가로 정지선(세로로 짧은 성분)을 track 에서 제거.
                solid = (self._filter_yellow_dashes(ymask)
                         if self.filter_yellow_dashes else ymask)
                # 실선 소실 폴백 (히스테리시스): 실선 픽셀이 문턱 미만이면 즉시
                # 점선 포함 raw 로 전환(급좌회전에서 안쪽 실선이 시야 밖 -> 바깥
                # 점선이라도 따라간다). 실선 전용 복귀는 dash_fallback_exit_frames
                # 연속 충족 시에만 — 실선이 막 걸친 순간의 급전환/차선 소실 방지.
                if self.filter_yellow_dashes:
                    if cv2.countNonZero(solid) < int(self.yellow_dash_fallback_px):
                        if not self._dash_fallback_on:
                            # 안쪽 실선 소실 '순간' = 1L 무장 시점 (__init__ 주석 참고).
                            # RA 이전 + DRIVE + 이번 래치에서 미사용일 때만.
                            if (not self._oneline_used and not self._ra_seen
                                    and self.drive_mode == 'DRIVE'
                                    and self.entry_oneline_frames > 0):
                                self._oneline_left = int(self.entry_oneline_frames)
                                self._oneline_used = True
                        self._dash_fallback_on = True
                        self._solid_ok_count = 0
                    elif self._dash_fallback_on:
                        self._solid_ok_count += 1
                        if self._solid_ok_count >= int(self.dash_fallback_exit_frames):
                            self._dash_fallback_on = False
                            self._solid_ok_count = 0
                    # (07-12 시도/원복 이력) RA 중 점선 폴백 금지(실선만)를 run33 에
                    # 시험 -> 진입 구간(RA+2~4s)에서 개구부 건너편 먼 실선을 잡고
                    # 직진 이탈. 사용자 결정으로 원래 동작(점선 폴백 허용) 복원.
                    sel = ymask if self._dash_fallback_on else solid
                else:
                    sel = ymask
                # 점선 힌트 갱신: 필터로 제거된 점선 픽셀들의 평균 x (track 과 같은
                # BEV 공간). 폴백으로 sel=ymask 면 차분이 비어 힌트 없음(기존 동작).
                dash = cv2.subtract(ymask, sel)
                if cv2.countNonZero(dash) >= 30:
                    self._yellow_dash_cx = float(np.nonzero(dash)[1].mean())
                else:
                    self._yellow_dash_cx = None
            else:
                sel = wmask
                self._yellow_dash_cx = None
                self._dash_fallback_on = False
                self._solid_ok_count = 0
            track = cv2.morphologyEx(sel, cv2.MORPH_OPEN, k) if use_morph else sel
        else:
            self._following_yellow = False
            self._yellow_exit_count = 0
            self._oneline_left = 0
            self._yellow_dash_cx = None
            self._dash_fallback_on = False
            self._solid_ok_count = 0

        # 차선 주축 기울기 a_h = dx/dy 를 갱신한다 (다음 프레임의 정지선 직교 게이트용).
        # track 은 추종 대상(흰 or 노란 실선)이라 정지선 성분이 대부분 빠져 있어,
        # 정지선을 밟는 순간에도 '차선 방향'을 준다.
        #
        # 07-10 실측: track '전체'에 직선을 피팅하면 링의 휜 실선에서는 국소 접선이 아니라
        # 곡선 전체의 평균 기울기가 나와, 정지선 앞에서 a_h=-0.59 (정답 +0.18 근처) 로
        # 부호까지 뒤집혔다. 그 탓에 진짜 정지선이 직교오차 38~44° 로 전량 거절됐다.
        # => 정지선 스캔 창과 '같은 행 범위'에서만 재서 국소 접선을 얻는다.
        th = track.shape[0]
        wy1 = int(th * self.crossline_roi_top_ratio)
        wy2 = int(th * self.crossline_roi_bottom_ratio)
        band = track[wy1:wy2, :] if wy2 > wy1 else track
        ys_lh, xs_lh = np.nonzero(band)
        # 다음 프레임의 '후보 배제 후 헤딩' 계산용으로 창 안 track 픽셀을 보관한다.
        self._crossline_track_pts = (ys_lh, xs_lh) if xs_lh.size >= 150 else None
        if xs_lh.size < 150:                 # 창 안이 비면 전체로 폴백
            ys_lh, xs_lh = np.nonzero(track)
        if xs_lh.size >= 150:
            raw_ah = float(np.clip(
                np.polyfit(ys_lh.astype(np.float64), xs_lh.astype(np.float64), 1)[0],
                -2.5, 2.5))
            # 정지선은 track 에서 완전히 제거되지 않아(차선과 한 성분) 밟는 순간 피팅을
            # 끌어당긴다. 07-10 실측: a_h 가 60ms 만에 +1.09 -> -1.05 로 부호까지 뒤집혔다.
            # 정지선은 1초 안팎의 과도 현상이므로 EMA 로 차선 헤딩을 유지한다.
            a = float(self.lane_heading_alpha)
            self._lane_heading = (a * raw_ah + (1.0 - a) * self._lane_heading
                                  if self._lane_heading_init else raw_ah)
            self._lane_heading_init = True

        # 재획득 우측 고정 모드 갱신 (직전 프레임 결과 기준; __init__ 주석 참고):
        # 합류부 창 안에서 직전 프레임이 소실이었으면 -> 이번에 잡히는 단선은
        # "소실에서 재획득한 선" = 구조상 우측 경계. nl=2 로 복원되면 해제.
        if self.drive_mode == 'ROUNDABOUT' and self.merge_zone:
            if not self._prev_found:
                self._reacq_right_active = True
            elif self._prev_nl == 2:
                self._reacq_right_active = False
            # _prev_nl == 1 인 동안은 유지 (재획득한 그 선을 계속 추적 중)
        else:
            self._reacq_right_active = False

        # 개구부 1L 재무장 (모드 전이 엣지 — __init__ 의 ra_*_oneline_frames 주석 참고).
        # RA 진입 시엔 Y-latch 가 이미 켜져 있고 RA 동안 강제 유지되므로 즉시 발효,
        # 해제 창은 Y-latch 가 흰 복귀로 풀리는 순간 _oneline_left=0 으로 자연 종료된다.
        if self.drive_mode != self._prev_drive_mode:
            if (self.drive_mode == 'ROUNDABOUT'
                    and int(self.ra_entry_oneline_frames) > 0):
                self._oneline_left = int(self.ra_entry_oneline_frames)
            elif (self._prev_drive_mode == 'ROUNDABOUT'
                    and self.drive_mode == 'DRIVE'
                    and int(self.ra_exit_oneline_frames) > 0):
                self._oneline_left = int(self.ra_exit_oneline_frames)
            self._prev_drive_mode = self.drive_mode

        # (1) multi-band look-ahead: ROI 를 가로 band 들(가까움..멂)로 나누고 각각에서
        # 차선 중심을 찾는다. 가까운 band 는 조향에, 먼 band 는 커브 선행 감지에 쓴다.
        roi_h = track.shape[0]
        band_h = max(1, roi_h // self.num_bands)
        band_windows = []   # guided 모드 탐색 창, 디버그 오버레이용
        oneline = (self._oneline_left > 0 and self._following_yellow)
        if oneline:
            # 진입 병합(1L) 모드: 좌/우 분할 없이 밴드의 노란 픽셀 전체 평균 = 선 하나,
            # 항상 우측(바깥) 경계로 분류 -> 중심 = 선 - 반차폭 (__init__ 주석 참고)
            self._oneline_left -= 1
            bands = []
            near_lanes = 0
            half = (self._lane_width / 2.0) if self._lane_width > 0 \
                else (w * self.single_line_offset)
            # 하단(가까운) oneline_near_bands 개만 사용 — 원거리 호 오염 차단
            # (__init__ 주석, 07-12 run40 실증)
            for i in range(min(int(self.oneline_near_bands), self.num_bands)):
                y1 = roi_h - i * band_h
                y0 = max(0, roi_h - (i + 1) * band_h)
                if y1 - y0 < 2:
                    continue
                # 선 위치 = 전체평균이 아니라 '최우측 클러스터'(x 85퍼센타일).
                # 07-12 run41 프레임 실증: 접근로 안쪽 실선이 가로로 누워 들어오면
                # 높이 기준 실선 판정을 통과 못 해 1L 이 실선이 보이는 채로 무장되고
                # (run14 수정이 기하에 뚫림), 전체평균이 실선+점선 사이로 끌려
                # 목표 = 평균-반차폭이 실선 왼쪽(코너 안) -> off -0.86 다이브.
                # 최우측 질량 = 바깥 점선이므로 P85 - 반차폭 = 올바른 차선 중앙.
                # 점선만 보이는 정상 1L 은 P85 ~= 평균 (동작 거의 동일).
                xs = np.nonzero(track[y0:y1, :])[1]
                line = (float(np.percentile(xs, 85))
                        if xs.size >= self.min_pixels else None)
                if line is not None:
                    bands.append((line - half, float(self.num_bands - i), i))
                    if i == 0:
                        near_lanes = 1
        elif self.use_guided_band:
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
            # 최종 중심 연속성 가드: 프레임당 이동을 반 차폭 x 비율로 제한.
            # nl=2(두 경계가 다 보이는 정상 두-선 추종)는 중점을 신뢰하고 가드를
            # 건너뛴다 (07-11 팀 결정 — 분기 목의 병적 nl=2 는 진입 병합(1L) 모드가
            # 분할 자체를 없애 원천 차단하므로, 가드는 단선/소실 쪽만 지킨다).
            if (self.center_jump_max_ratio > 0.0 and self._prev_center is not None
                    and self._lane_width > 0 and near_lanes != 2):
                j = self.center_jump_max_ratio * (self._lane_width / 2.0)
                d = lane_center - self._prev_center
                if abs(d) > j:
                    lane_center = self._prev_center + (j if d > 0 else -j)

        raw_offset = float(max(-1.0, min(1.0, (lane_center - mid) / (w / 2.0))))
        # 헤어핀 헤딩 보정: FOLLOW-Y 주행(DRIVE) 중 track 주축이 수직에서 크게
        # 기울면(급커브 도로가 BEV 에서 옆으로 누움) 밴드 가로 중심만으로는 조향이
        # 반대로 나올 수 있다 -> 주축 기울기(dx/dy)를 헤딩 오차로 보고 보정.
        # dx/dy > 0 = 멀수록(위로 갈수록) 왼쪽 = 좌커브 -> 음(좌조향) 보정.
        if (self._following_yellow and self.drive_mode == 'DRIVE'
                and lane_found and float(self.yellow_heading_gain) > 0.0):
            ys_h, xs_h = np.nonzero(track)
            if xs_h.size >= 150:
                a_h = float(np.polyfit(ys_h.astype(np.float64),
                                       xs_h.astype(np.float64), 1)[0])
                a_h = max(-2.5, min(2.5, a_h))
                raw_offset = float(max(-1.0, min(
                    1.0, raw_offset - float(self.yellow_heading_gain) * a_h)))
        junction = self._detect_junction(clean, w)
        fork = self._detect_fork(clean, w)
        # 개구부 fk 소실 규칙 (07-12 run37 실측): RA+5s 무장(merge_zone) 이후
        # 갈림길 서명(fork) 프레임은 차선 소실로 취급한다. 개구부의 두 점선 열이
        # 벌어지는 부챗살 = fork 검출기의 서명 그 자체 — nl=2 중점이 열 사이
        # 틈(도로 밖)을 겨냥하는 쐐기 추종(랩 종료 이탈 5전 5패)을 프레임 단위로
        # 차단하고, decision 의 ra_blind_bias(0.41 링 호)가 접선 헤딩을 유지한 채
        # 개구부를 건너게 한다. 재획득 시 기존 0->1 규칙이 이어받는다.
        # 실측 근거 (run31/37): 링 순항(RA+5~11s) fk 레벨 0% (오발 없음),
        # 이탈 직전 쐐기 구간 38~63% 점화. EMA 갱신 전에 차단해 오염 방지.
        if self.drive_mode == 'ROUNDABOUT' and self.merge_zone and fork:
            lane_found = False
            num_lanes = 0
        # (5) 시간축 스무딩(EMA)으로 프레임 간 조향 떨림을 줄인다.
        if lane_found:
            self._offset_ema = (self.smooth_alpha * raw_offset
                                + (1.0 - self.smooth_alpha) * self._offset_ema)
            self._prev_center = lane_center   # 다음 프레임을 위한 guide 시드
        offset = float(self._offset_ema)

        # 다음 프레임의 "소실->재획득" 전이 판정용 직전 상태 저장
        self._prev_found = bool(lane_found)
        self._prev_nl = int(num_lanes)

        # 디버그 이미지는 웹 대시보드용일 뿐 주행에 안 쓰인다. 프레임을 띄엄띄엄
        # (lane_node 의 debug_hz) 보낼 때 그리기+인코딩을 통째로 건너뛰어 보드 부하를 던다.
        # 노란 추종 중이면 실제 추종 대상인 track(노란 전용)을 그려 튜닝에 도움 준다.
        debug = (self._draw_debug(roi, track, lane_center, offset, yellow_ratio,
                                  curvature, band_windows, roi_raw, near_cx, la_cx)
                 if draw_debug else None)
        return (lane_found, offset, num_lanes, junction, yellow_ratio, yellow_offset,
                curvature, yellow_crossline, fork, red_ratio, debug)

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
            # 폭 학습 하한 가드: 단선 하나가 탐색창 중앙에 걸쳐 좌/우로 갈라지면
            # width 가 선 두께(~6px)로 붕괴해 EMA 를 오염시킨다. 실측 프리셋
            # (lane_width_init)의 절반 미만 폭은 "같은 선의 오분할"로 보고 학습 스킵.
            floor = 0.5 * self._lane_width_init
            if width > 0 and width >= floor:
                self._lane_width = (width if self._lane_width <= 0
                                    else (1 - self.width_ema) * self._lane_width
                                    + self.width_ema * width)
            return (lx + rx) / 2.0, 2
        half = (self._lane_width / 2.0) if self._lane_width > 0 else (w * self.single_line_offset)
        line = lx if lx is not None else rx
        if line is None:
            return None, 0
        # 합류부 재획득 단선 수기 보정 (매뉴얼 §7 확정 규칙, 07-11 사용자 지정 조건):
        # 합류부 창 안 + "소실(0)->단선(1) 재획득" 상황에서만 — 재획득 단선을 연속성
        # 분류가 좌측 경계로 오분류하면 중심이 선 오른쪽에 놓여 바깥으로 밀린다
        # (run22~24 3연속 언더스티어 서명). 재획득 선은 구조상 바깥(우측) 경계다
        # -> 우측 경계 고정, 중심 = 선 - 반차폭. 연속 추적 중이던 단선은 기존 로직.
        # (run25 폐기 교훈: RA 전체/전 단선 적용은 과범위 — 링 추종을 오염시킴)
        if (self.drive_mode == 'ROUNDABOUT' and self.merge_zone
                and self._reacq_right_active):
            return line - half, 1
        # 대신 "점선의 반대편 경계"라는 코스 구조로 좌/우를 정한다. 오른쪽 실선이
        # 차 왼쪽에 보여도 (점선이 그보다 더 왼쪽에 있으면) 오른쪽 경계로 맞게
        # 분류돼 중심을 실선의 왼쪽(-half)에 둔다.
        if self._yellow_dash_cx is not None:
            if line < self._yellow_dash_cx:
                return line + half, 1   # 실선이 점선보다 왼쪽 = 왼쪽 경계
            return line - half, 1       # 실선이 점선보다 오른쪽 = 오른쪽 경계
        # 시간 연속성 분류: 화면 반쪽 기준(아래 폴백)은 단선이 화면 중앙을 넘나드는
        # 커브에서 프레임마다 좌/우가 뒤집혀 중심이 ±half 로 널뛴다.
        # 07-11 실주행 실측: off 가 -0.53 <-> +0.07 로 요동, steer +0.81 <-> -0.50
        # -> 차가 차선 밖으로 튕겨나가 완전 소실 (자율주행 첫 시도 2회 연속 이 원인).
        # 차선 중심은 한 프레임(50ms)에 반 차선폭을 점프할 수 없으므로, 직전 프레임
        # 중심에 더 가까운 해석을 고른다. 출발선에서 두 선으로 정확한 중심을 잡고
        # 시작하므로 단선 전환 순간에도 올바른 분류가 연속적으로 이어진다.
        if self._prev_center is not None:
            cand_left = line + half      # 이 선이 왼쪽 경계일 때의 중심
            cand_right = line - half     # 이 선이 오른쪽 경계일 때의 중심
            if abs(cand_left - self._prev_center) <= abs(cand_right - self._prev_center):
                return cand_left, 1
            return cand_right, 1
        if lx is not None:
            return lx + half, 1
        return rx - half, 1

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
        늘어난다(한 차선=좌·우 2군집 -> 분기=각 브랜치의 안/바깥선으로 3~4군집).
        컬럼별 세로픽셀 수를 세어 '라인' 컬럼의 연속 런(run) 개수로 판정한다.
        바깥 span 폴백은 fork_span_ratio>0 일 때만 쓰며, 기본 비활성이다(위 주석 참조).
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
        if groups >= self.fork_min_groups:
            return True
        if self.fork_span_ratio <= 0.0:
            return False
        on_idx = np.nonzero(col_on)[0]
        span = float(on_idx[-1] - on_idx[0])
        return bool(span >= self.fork_span_ratio * w)

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
            if self._oneline_left > 0:
                follow_tag += '[1L]'   # 진입 병합 모드 활성 (남은 프레임 있음)
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
