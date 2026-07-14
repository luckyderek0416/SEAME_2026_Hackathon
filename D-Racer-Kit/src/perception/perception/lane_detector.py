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
                 crossline_min_width_ratio=0.20,
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
        # 최종 중심 연속성 가드: 프레임당 중심 이동 <= 반차폭 x 이 비율. 0=off.
        # run9 분기 쐐기 점프 차단; 0.15 는 쐐기 당김(프레임당 0.05~0.08) 통과 -> 0.10 확정.
        # 부작용: 소실 후 재획득 수렴이 이 한도로 느려짐. 최종 중심에만 적용(곡률 무관).
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
        # RA 진입/진출 위치 신호 (decision 표결용). 노란 전용 mask 에서만 계산, 차선
        # 중심에는 무영향.
        self.crossline_roi_top_ratio = crossline_roi_top_ratio       # 스캔 창 상단 (ROI 높이의 0..1)
        self.crossline_roi_bottom_ratio = crossline_roi_bottom_ratio # 스캔 창 하단
        self.crossline_min_width_ratio = crossline_min_width_ratio   # 성분의 가로 스팬 최소 비율 (w 기준)
        self.crossline_min_area_px = 60       # 성분 최소 픽셀 수 (노이즈/점선 dash 필터)
        # HoughLinesP: 누적 투표 임계 / 같은 직선으로 이어붙일 최대 간격(px).
        self.crossline_hough_thresh = 25
        # 07-10 실측: gap<=8 은 진짜 정지선도 못 이음(검출 0%), gap=10 = 정지선 100%
        # + 점선 이어붙임 0건.
        self.crossline_hough_max_gap = 10
        # 선분 위 픽셀 충실도 하한 (실측: 진짜 정지선 0.86~1.00 / 이어붙인 점선
        # 0.52~0.73). 링 안 '진짜 직각 실선'은 못 거름 — 상태머신 몫.
        self.crossline_min_solidity = 0.80
        # SW 교차 게이트 (07-13 사용자 설계): SW 코리도 창이 락 중이면, 추적 피팅선을
        # '수평으로 가로지르는' 선분만 정지선 후보로 인정 — 경로 밖 가로선(링 건너편
        # 도로 선, ~30-35% 지점 약피처)을 원천 배제해 군집 카운트 신호를 청소한다.
        # 피팅이 없으면(창 밖/락 실패) 기존 동작 폴백. 0=off.
        self.crossline_sw_gate = 0
        self.crossline_sw_margin = 40.0   # 피팅 x 대비 선분 스팬 여유(px)
        # True 면 첫 채택 후에도 모든 선분을 진단 목록에 남긴다 (임계 튜닝용).
        self.crossline_debug_all = False
        # 수직성 게이트: 지면 공간에서 |cos(차선, 선분)| <= sin(perp_tol_deg) 면 직교.
        # 차선을 선분으로 잘못 뽑으면 cos=1(평행)이라 어떤 헤딩에서도 통과 불가 —
        # 절대각 임계(50°)로 못 거르던 오검출(07-10 헤딩 43° RA 오진입)의 해법.
        # 0 이하 = 게이트 비활성.
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
        # 바깥라인 간격 폴백. 0 이하 = 비활성. 07-10 실측: 직선 span 0.60~0.65w 로
        # 임계와 여유 0~8% — 판별력 없음. groups 판정만으로 충분해 꺼 둔다.
        self.fork_span_ratio = fork_span_ratio
        # fork_dir: decision 이 표결 확정한 방향('left'/'right'/None). lane_node 가
        # /decision/fork_dir 구독으로 매 프레임 갱신한다. 설정되면 guided-band 시드를
        # 그 브랜치 쪽으로 밀어(fork_seed_px) 한쪽 브랜치만 추종 => median(표지판 섬) 배제.
        self.fork_dir = None
        self.fork_seed_px = fork_seed_px
        # --- 노란색 우선 추종 (In 코스 색상 상태머신, 히스테리시스) ---
        # yr >= ratio -> YELLOW(노란 전용 마스크 추종, 흰 무시). WHITE 복귀 = "노랑
        # 소실(yr < ratio x exit_frac) AND 흰 우세" 연속 exit_frames — 진입/해제를
        # 상호배타로 둬 전환 구간 Y/W 플리커 방지. ROUNDABOUT 은 강제 YELLOW.
        # junction/fork 는 합친 마스크 그대로. course != 'in' 이면 전체 비활성.
        self.follow_yellow = follow_yellow
        self.follow_yellow_ratio = follow_yellow_ratio
        self.follow_yellow_exit_white_ratio = follow_yellow_exit_white_ratio
        self.follow_yellow_exit_yellow_frac = 0.5   # 해제 노랑 문턱 = 진입문턱 × 이 비율
        self.follow_yellow_exit_frames = 10         # 해제 조건 연속 프레임 수 (~0.3s@30fps)
        # 탈출(SW 창) 중에는 해제를 빠르게 — run59: 해제가 늦어 흰 정렬 시작도 늦음. 0=전역값.
        self.follow_yellow_exit_frames_exit = 4
        # 흰 인계(핸드오버) 창 (07-12 run59 실증): RA 탈출 후 Y래치 해제 순간부터
        # N프레임 동안 ① 흰 track 점선 제거 — 이 트랙 흰 도로는 양측 실선뿐, 점선은
        # 합류부 개구부 표식이라 차선이 아님 ② 노란 꼬리를 track 에 유지 — 병합
        # 코리도의 양변 = 노랑(안쪽)+흰 실선(바깥) -> nl=2 기하 유지 ③ 흰 track
        # 주축 기울기로 조향 보정 — 사선 도달 시 offset~0 이라 PID 가 못 보는
        # 헤딩 오차(run59 EX+7~10 45° 관통)를 선 방향으로 정렬.
        self.w_align_frames = 60        # 창 길이 (~3s@20fps, run59 리플레이: 2s 는 사선 구간 전에 만료). 0=off
        self.w_align_gain = 0.4         # 기울기 -> offset 보정 게인. 0=보정만 off
        self.w_align_min_px = 80        # 점선 필터 결과 이 미만이면 raw 흰 폴백
        self.w_align_dash_fallback = 0  # 0(기본)=폴백 금지: 점선 잡지 말고 무검출 -> 결정층 브리지 바이어스에 위임
        # 재래치 억제 (07-14 병합 플랩 대응 — lane_node 주석 참고): RA 후 w_align
        # 창 활성 중 Y래치 재진입 차단. 창 track 이 노란 꼬리를 포함하므로 무손실.
        self.w_align_block_relatch = 1
        # Y+W 선행 혼합 (07-13 run103, 사용자 설계): RA 후 DRIVE[Y] 중 흰 '실선'을
        # track 에 미리 합성. 높이 게이트(_filter_yellow_dashes)가 흰 점선(개구부
        # 표식)과 가로로 누운 먼 도로선을 걸러 평행해진 흰 실선만 통과 —
        # ① 좌곡에서 안쪽 노랑 순간 소실 시 픽 유지 ② 해제 전 nl=2 인계 기하.
        # 무장 게이트 (07-13 run105: 탈출 스윙 중 노랑이 시야를 벗어난 순간
        # 아웃코스 흰 호를 혼합해 좌훅 이탈 — 높이 게이트만으론 부족):
        # 탈출 코리도 창 밖 + 노란 실선 위에 실제로 앉은 프레임에서만 무장하고,
        # 노랑 소실 시 carry 프레임 동안만 흰 실선이 픽을 인계한다(분기 딥 구제).
        self.yw_premix = 0
        self.yw_premix_carry_frames = 30   # ~1.5s@20fps (run103 딥 0.75s 커버)
        self._yw_carry = 0
        self.follow_yellow_blind_release_frames = 12  # 흰 우세 없이도 노랑 소실 지속 시 해제 (0=off)
        self._yellow_blind_count = 0
        self._w_align_left = 0
        self.course = course
        self._following_yellow = False   # 현재 YELLOW 모드인지 (프레임 간 유지)
        self._yellow_exit_count = 0      # 해제 조건 연속 카운터
        # 진입 병합(1L) 모드: '안쪽 실선 소실 순간'(dash 폴백 진입)부터 N프레임 동안
        # 좌/우 분할 없이 노란 질량을 "바깥선 하나"로 취급 (중심 = 선 - 반차폭) —
        # 분기 목 nl=2 쐐기(run9/11) 원천 차단. 래치 '순간' 무장 금지: 실선이 보이는
        # 동안 무장하면 실선을 바깥선으로 오인, 코너를 깎는다(run14, off -0.82).
        # 4s = 저속 대비 (run17: 2s 는 회전 미완 만료). 07-13 에 6s(120f) 시도했다가
        # run89 후 사용자 결정으로 4s 원복 (리플레이 A/B 는 4s=6s 동일 — 보수 선택).
        self.entry_oneline_frames = 80   # ~4s@20fps. 0=off
        # 1L 은 하단(가까운) 밴드만 사용 — 원거리 링 호가 전체평균을 좌로 오염해
        # 급좌 다이브(run40, off -0.75)하던 것을 구조적으로 차단.
        self.oneline_near_bands = 2
        self._oneline_left = 0
        self._oneline_used = False       # 이번 래치에서 이미 무장했는지
        self._ra_seen = False            # ROUNDABOUT 을 겪은 뒤에는 재무장 안 함
        # 개구부 1L 창: RA 진입/해제 전이 엣지에서 1L 재무장 (개구부 두 점선 열의
        # nl=2 쐐기 차단, run27~29 실증). run31 A/B 뒤 사용자 결정으로 둘 다 0(off).
        # 대체안 '실선 소실 트리거'는 run32 오발(안쪽 나선)로 폐기, 코드 제거됨.
        self.ra_entry_oneline_frames = 100   # ~5s@20fps. RA 진입 전이. 0=off
        self.ra_exit_oneline_frames = 60     # ~3s@20fps. RA->DRIVE 해제 전이. 0=off
        self._prev_drive_mode = ''
        # YELLOW 추종 중 점선(dash) 무시: 조향용 track 마스크에서 세로 길이가 짧은
        # 노란 성분(점선 dash, 그리고 가로 정지선)을 제거해 "실선만" 추종한다.
        # raw ymask 는 yellow_ratio/crossline/Y모드 진입판정에 그대로 쓰이므로 무관.
        self.filter_yellow_dashes = True
        self.yellow_solid_min_h_ratio = 0.30   # ROI 높이의 이 비율 이상 세로로 이어져야 "실선"
        # 실선 소실 폴백: 실선 픽셀이 문턱 미만이면 점선 포함 raw ymask 로 추종 유지
        # (급좌회전서 안쪽 실선이 FOV 이탈 -> 직진 폴백으로 진입로 놓침 방지).
        self.yellow_dash_fallback_px = 120
        # 해제 히스테리시스: 폴백 진입 즉시 / 해제는 실선 연속 N프레임 — 실선이 막
        # 걸친 순간 점선을 뚝 버리는 급전환 방지.
        self.dash_fallback_exit_frames = 30   # 1s@30fps (07-08 트랙: 0.33s 는 너무 짧음)
        self._dash_fallback_on = False
        self._solid_ok_count = 0
        # 헤어핀 헤딩 보정 게인 (track 주축 기울기 -> 조향 보정). 07-11: a_h 가 ±2.5
        # 로 널뛰어 offset 을 통째로 흔듦 -> 기본 0(off). ROUNDABOUT 제외.
        self.yellow_heading_gain = 0.4
        # 단선 좌/우 분류용 점선 힌트: 실선은 항상 점선의 반대편 경계 — 화면 기준이
        # 아니라 점선 대비 위치로 분류. None = 힌트 없음.
        self._yellow_dash_cx = None
        self._yellow_dash_px = 0
        # 현재 주행 모드(상태머신 상태 문자열). decision 이 publish 하고 lane_node 가
        # 매 프레임 갱신 -> BEV 디버그 화면에 표시. '' 면 표시 안 함(decision 미실행).
        self.drive_mode = ''
        # 합류부 창 플래그 (/decision/merge_zone). "단선=우측 경계" 규칙을 이 창
        # 안으로만 스코프 (RA 전체 적용은 과범위 — run25 폐기).
        self.merge_zone = False
        # 재획득 우측 고정 모드 (07-11 사용자 지정 조건): 창 안에서 "소실(0) -> 단선(1)"
        # 전이로 잡힌 선에만 우측 경계 고정을 적용한다. 이미 연속 추적 중이던 단선은
        # 기존 연속성 분류 그대로. 두 선(nl=2)이 보이면 기하가 복원된 것이므로 해제.
        self._reacq_right_active = False
        self._prev_found = False   # 직전 프레임 lane_found
        self._prev_nl = 0          # 직전 프레임 num_lanes
        # --- 슬라이딩 윈도우(SW) 코리도 추적 (RA 진입/탈출 전이 창 전용) ---
        # 시간 기반 개루프 호가 배터리별 속도로 런마다 다른 호를 그리는 문제(run47
        # 통과/run48 실패 동일 코드)의 위치 기반 폐루프 대체: 전이 창 동안 상자
        # 체인+2차 피팅으로 기대 곡률 방향의 선을 문다. 진입 창 = 좌향 기대(우향
        # = 반대 가지 기각), 탈출 창 = 우향 기대(초반 한정, gate/open 주석 참고).
        # 안전 계약: 락 프레임만 밴드 대체, 실패 프레임은 기존 결과 통과(폴백),
        # 창 0(기본)이면 전 경로 무변경.
        # 입력: 탈출 = raw(좌측 경계가 점선), 진입 = solid(병합 사선 노이즈 제거).
        self.sw_entry_frames = 0        # RA 진입 창 길이(프레임). 0=off. 권장 140(~7s@20fps)
        self.sw_exit_frames = 0         # RA 탈출 창 길이(프레임). 0=off. 권장 60(~3s)
        self.sw_entry_input = 'solid'   # 진입 창 입력: 'solid'(dash 제거) | 'raw'
        self.sw_out_always = 1          # OUT 코스: DRIVE 전 구간 코리도 상시 (S자 좌우 교대 소실 대응).
                                        # dir=0 -> 곡률가드/탈출규칙/STOPLINE 비활성, 입력 raw. in 코스 무영향
        # IN 접근 코리도 (07-14 사용자 설계): DRIVE[Y] 첫 래치 + 첫 좌회전 완료(1L 종료
        # + nl=2 연속 5f) 시점부터 RA 래치까지 노란 코리도. B 에서 좌측 합류부 차선이
        # 밴드 '평균'을 당기던 것(런 실측)을 락 추적으로 대체하고, 코리도 활성 = crossline
        # sw_gate 자동 무장이라 B 가짜 정지선도 함께 기각(run76~80 검증 게이트 재사용).
        # dir=-1 무장 → 우곡률 가지 기각(sw_curv_max_a, run87)·실선 입력 자동 적용.
        # 해제 = RA 래치 시 진입 코리도가 덮어씀. 0=off (라이브 킬스위치).
        self.sw_approach_frames = 600   # failsafe 상한 (~30s@20fps; RA 래치가 정상 종료)
        # IN 전구간 코리도 (07-14 밤, 사용자 설계): DRIVE[W] 래치 순간부터 상시 SW.
        # 실측 근거(22:10 런): 출발~첫 좌회전이 일반 파이프라인 + 1L 인 동안 누운 노란
        # 단선이 crossline 으로 오인돼 race_t=11.8s 가짜 RA 진입 → 이탈. W 구간은 흰
        # 페어(양쪽 상자 체인 → 중점), Y래치 엣지에 노란 마스크로 재시드 + dir=-1
        # (approach 와 동일: wrongdir 게이트 + run87 우곡 가드 + solid 입력 자동).
        # RA 진입/탈출은 기존 entry/exit 창이 엣지에서 덮어쓰고, 탈출~병합 완료
        # (_merge_done 전)엔 재무장하지 않는다(exit 기계 전담). 락 실패 프레임 폴백 =
        # 기존 파이프라인(1L 포함) 그대로 — 기존 스택은 전부 보존된 안전망.
        self.sw_drive_always = 1        # 0=off (라이브 킬스위치; 끄면 즉시 기존 스택 복귀)
        # crossline 직교 게이트의 차선 헤딩 소스 교체 (07-14 밤): 후보 소거 후 잔여
        # 픽셀이 없는 프레임(첫 좌회전의 누운 단선 = 마스크 대부분이 후보 자신)에서
        # _lane_heading_excluding 이 EMA 폴백 → 누운 차선이 '직교'로 오인. 코리도
        # 활성 중엔 추적 피팅의 국소 접선(2a·y+b)이 그 프레임의 진짜 차선 방향.
        self.crossline_sw_heading = 1   # 0=구식(EMA 폴백) 복귀
        self._sw_prev_yellow = False    # Y래치 엣지 감지 (drive 창 재시드/방향 전환용)
        self._sw_kind = ''              # 활성 창 종류: entry/exit/approach/out/drive (킬스위치 매핑)
        self._sw_nl2_count = 0          # 접근 무장용 nl=2 연속 카운터
        # 1L 조기 해제 → SW 인계 (07-14 사용자 설계): 1L 중 노란 실선 '페어'(차폭 간격
        # 두 기둥)가 K프레임 연속 복원되면 = 좌회전 종료 이벤트 → 80f 만료를 안 기다리고
        # 즉시 1L 종료 + 접근 코리도 무장 (시간창 속도의존 제거; 80f 만료는 failsafe 잔존).
        # 분기 목 쐐기(run9/11) 오인 방어: min_hold(쐐기는 1L 초반에만) + 질량/간격 + K연속.
        self.oneline_release_pair_k = 5    # 페어 연속 프레임 (0=이벤트 해제 off)
        self.oneline_release_min_hold = 20 # 1L 최소 유지 프레임 (~1s, 쐐기 구간 통과)
        self._oneline_pair_count = 0
        self.fork_blind_frac = 0.40     # OUT 갈림길: 확정 방향 반대쪽 컬럼 마스킹 비율 (0=off)
        self.fork_blind_frames = 60     # 마스킹 지속 상한(프레임, ~3s@20fps)
        self._fork_blind_left = 0
        self._prev_fork_dir = None
        self._fork_cut = None           # (side, px) — 이번 프레임 적용된 마스킹
        self.sw_exit_input = 'raw'      # 탈출 창 입력: 점선이 좌측 경계라 raw 필수
        self.sw_num_boxes = 9           # 상자 개수 (ROI 세로 분할)
        self.sw_curv_max_a = 0.003      # 진입 창 우곡률 상한 (run87 실측: 링 -0.013~+0.0015,
                                        # B 가지 +0.006 — 초과 피팅 기각 = 개구부 가지 오물림 방지)
        # STOPLINE 분류기 (07-13 재설계): RA+코리도 중 "관통(coverage)+정면(각도)" 프레임만
        # 정지선으로 인정. B 개구부 스침은 가장자리/비스듬이라 원리적으로 탈락 (실측:
        # 건강 궤적에서 A cov 0.42~0.71/각 5~19도 vs B cov<=0.29/각 21도+).
        self.stopline_mode = 0          # 1=활성 (기본 off — 리플레이 검증 후 라이브 전환)
        self.stopline_ang_max = 15.0    # 정면 판정: 직교로부터의 각도 상한(도)
        self.stopline_cov_min = 0.25    # 차로내부 합집합 커버리지 하한 (f59 실측 0.28 포용, B 정면후보는 0)
        self.stopline_sol_min = 0.55    # 정면 후보 solidity 하한 (레거시 0.80 대비 완화 = 재현율)
        self.last_stopline_cov = 0.0    # 디버그: 직전 프레임 커버리지
        self._sw_interior = None        # ('pair',fl,fr,_)|('single',abc,None,side) — 차로내부 정의
        self.sw_box_margin = 30         # 상자 반폭(px)
        self.sw_max_shift = 20          # 상자당 중심 이동 상한(px) — 누운 실선 끌림/반대 가지 점프 차단
        self.sw_min_box_px = 8          # 상자 '적중' 최소 픽셀
        self.sw_min_boxes = 3           # 유효 피팅 최소 적중 상자 수 (2상자 노이즈로 곡선 금지)
        self.sw_min_pixels = 50         # 유효 피팅 최소 총 픽셀 (run32 소량질량 나선 차단)
        self.sw_wrongdir_px = 8.0       # 방향 게이트: 기대 반대 방향 기욺이 이 px 넘으면 기각
        self.sw_max_resid_px = 12.0     # 피팅 평균 잔차 상한(px)
        self.sw_peak_min_px = 40        # 시드 히스토그램 피크 최소 질량(세로 합)
        self.sw_max_peaks = 3           # 프레임당 시도할 피크 시드 수
        self.sw_cross_row_frac = 0.45   # 이 비율 이상 켜진 가로줄(정지선) 행을 입력에서 제거
        self.sw_side_default = -1       # 단선 분류 폴백: -1=우측 경계(중심=선-half; reacq 규칙과 동일 근거)
        self.sw_exit_straight_k = 5     # 탈출 조기 해제: '직선' 연속 프레임 수
        self.sw_exit_straight_px = 8.0  # |기욺| 이 미만이면 '직선'
        self._sw_left = 0               # 남은 창 프레임 (0=창 밖)
        self._sw_dir = 0                # 기대 곡률 방향: -1=좌향(진입) +1=우향(탈출) 0=비활성
        self._sw_prev_fit = None        # 직전 유효 피팅 (a,b,c) — 다음 프레임 시드 연속성
        self._sw_last_side = 0.0        # 추적선이 좌(-1)/우(+1) 경계인지 (0=미정/양선)
        self._sw_straight = 0           # 탈출 직선 연속 카운터
        self.sw_exit_wsteady_k = 20     # 흰 인계 종결자: 흰 추종 연속 K프레임 -> 창 종료 (0=off)
        # 탈출 방향 게이트 적용 프레임: 탈출로 = 우커브 -> 좌굽이 2단계라 창 내내
        # 걸면 병합 꼬리 전부 기각(run53 리플레이). 이 프레임 수 이후 게이트 해제.
        self.sw_exit_gate_frames = 40   # ~2s@20fps
        # 탈출 개방 구간 (run54 실증): 발화 순간 코리도가 좌향으로 들어와 +1 게이트가
        # 코리도 자체를 기각 -> 맹목 표류. 발화 직후 이 프레임 수 동안 방향 게이트
        # 해제 + 시드 전체 높이(코리도가 정지선 너머 상단에만 있는 프레임 대응),
        # 대신 |기욺| 상한으로 정지선 대각선(실측 102~148px)만 기각 — 코리도는 완만.
        self.sw_exit_open_frames = 30   # 개방 구간 길이. 0=off(기존 동작과 동일)
        # 탈출 개구부 링쪽 컬럼 컷 (07-14 밤 run_c 실측): 발화 순간 개구부에서 진입
        # 커넥터 점선(링쪽)과 탈출 선이 '차폭 간격 가짜 페어'를 이뤄 코리도가 신목
        # 한가운데를 추종 → 좌이탈. 발화 직후 이 프레임 수 동안 SW 입력에서 탈출
        # 반대쪽(fork_dir 반대) 컬럼을 제거해 코리도가 탈출측 선만 보게 한다.
        # 컷으로 락 실패 시 폴백 = 일반 파이프라인(fork 가이드 우측 시드) — 원 설계.
        self.sw_exit_cut_frames = 40    # ~2s@20fps. 0=off
        self.sw_exit_cut_frac = 0.45    # 제거 컬럼 비율
        self.sw_open_max_lean_px = 90.0 # 개방 구간 |기욺| 상한 (정지선 대각선 실측 102+)
        self._sw_open = False           # 이번 프레임이 개방 구간인지
        self._sw_len = 0                # 무장 시점의 창 길이 (age 계산용)
        self._sw_gate_dir = 0           # 이번 프레임의 방향 게이트 (0=게이트 없음)
        self._sw_wsteady = 0            # 흰 인계 연속 카운터 (구 종결자 — 07-14 흰이벤트로 대체, 미사용)
        self._sw_last_lean = 0.0        # 마지막 락 프레임의 기욺(top_x - bottom_x, px)
        self._sw_locked_dbg = False     # 디버그 표기용: 이번 프레임 락 여부
        # --- 탈출→흰병합 재구현 (07-14 사용자 설계, _sw_dir>0 한정) ---
        # 항목1: 탈출 단선 락의 점선/실선 판별. 피팅 곡선을 따라 마스크 점유율을 재
        # (curve occupancy) 이 문턱 초과면 '실선' → SW 락 포기(None, 일반 파이프라인
        # 인계). 이하면 '점선' → 무조건 바깥(우측) 경계 고정. ⚠️ 실측 캘리 필요.
        self.sw_exit_dash_occupancy_max = 0.72
        # 항목2: 탈출창 종료 = 흰 '실선' 이벤트. 최하단 흰 후보를 위로 추적한 연속
        # 스팬(점선 토막 최대길이 초과 = 실선)이 문턱 이상 + 스팬 내 점유율 이상이
        # 연속 confirm_frames 프레임. 방향 무관(bbox 높이 아님, 스팬 기반).
        # ⚠️ 네 값 전부 임시 — BEV 점선 토막 길이/흰선 두께로 실측 캘리 필요.
        self.sw_exit_white_bottom_px = 40     # 최하단 밴드 흰 최소 픽셀 (후보 성립)
        self.sw_exit_white_min_span_px = 55   # 연속 스팬 하한 (점선 토막 최대길이 초과)
        self.sw_exit_white_solidity_min = 0.80  # 스팬 내 점유율 하한 (long-but-sparse 배제)
        self.sw_exit_white_confirm_frames = 4   # 발화 디바운스 (공간=스팬 / 시간=이 값 이중방어)
        self._sw_white_confirm = 0      # 흰 실선 이벤트 연속 카운터
        # 원웨이 병합 래치 (항목2): post-RA 에서 Y→W 전환이 한 번 일어나면(경로 무관)
        # 남은 런 동안 Y 재래치 금지. 런 시작(디텍터 인스턴스)마다 False.
        self._merge_done = False
        self._sw_fit_dbg = []           # 디버그 오버레이용 피팅 곡선 [(x,y),...] 리스트

    def _build_mask(self, roi):
        """(lane_mask, white_mask, yellow_mask, yellow_ratio, yellow_offset,
        yellow_crossline, red_ratio) 반환.

        HSV 모드 = 흰|노랑 mask (gray 모드는 밝기 threshold, w/y=None). w/y 전용
        mask 는 색상 추종 상태머신용, lane_mask(합침)는 junction/fork 용.
        yr/yo = 노란 비율/무게중심, xl = 가로선 감지, rr = 빨간 노면 비율 —
        전부 신호 전용, 차선 중심 계산에는 무영향.
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
        """YELLOW 추종용 실선 필터: bbox 높이 >= min_h_ratio x ROI높이 인 성분만
        남김 (점선 dash/가로 정지선 제거). track 전용 — raw ymask 신호는 무관."""
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
        """선분 위 픽셀 충실도 (0..1). Hough 가 이어붙인 점선은 기하로는 진짜 정지선과
        구분 불가(07-10: 길이·기울기·직교성 전부 겹침) — 선분 샘플링 점유율로 가른다."""
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

    def _curve_occupancy(self, mask, abc, y_lo, y_hi):
        """피팅 곡선 x=a·y²+b·y+c 를 y_lo~y_hi 로 샘플링해 마스크 점유율(0..1) 반환.
        _segment_solidity 와 같은 철학이되 직선 대신 곡선을 따라간다. 점선은 토막 사이
        공백으로 점유율이 낮고, 실선은 ~1.0 (탈출 단선 점선/실선 판별용, _sw_dir>0 전용)."""
        a, b, c = abc
        n = max(8, int(abs(y_hi - y_lo)))
        ys = np.linspace(y_lo, y_hi, n)
        xs = a * ys * ys + b * ys + c
        h, w = mask.shape[:2]
        xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
        hit = (mask[yi, xi] > 0)
        # ±1px 가로 오차 허용 (피팅 미세 오프셋; 점선의 세로 공백은 그대로 드러남)
        hit |= (mask[yi, np.clip(xi - 1, 0, w - 1)] > 0)
        hit |= (mask[yi, np.clip(xi + 1, 0, w - 1)] > 0)
        return float(np.mean(hit))

    def _white_solid_span(self, wmask, roi_h, band_h):
        """최하단 밴드의 흰 후보를 위로 한 행씩 추적해 (연속 스팬 px, 점유율) 반환.
        점선 토막은 gap 에서 끊겨 스팬이 짧고, 실선은 밴드 너머로 이어진다 —
        bbox 높이(방향 의존) 대신 스팬(방향 무관)이 진짜 판별자. 탈출창 종료 전용."""
        h, w = wmask.shape[:2]
        y0 = max(0, roi_h - band_h)
        if int(np.count_nonzero(wmask[y0:roi_h, :])) < int(self.sw_exit_white_bottom_px):
            return 0.0, 0.0
        cols = np.count_nonzero(wmask[y0:roi_h, :], axis=0)
        x = float(np.argmax(cols))      # 최하단 밴드에서 가장 두꺼운 흰 열
        margin = 12
        gap = 0
        top_y = roi_h
        hit = 0
        for y in range(roi_h - 1, -1, -1):
            x0 = int(max(0, x - margin))
            x1 = int(min(w, x + margin + 1))
            nz = np.nonzero(wmask[y, x0:x1])[0]
            if nz.size > 0:
                x = x0 + float(nz.mean())
                top_y = y
                gap = 0
                hit += 1
            else:
                gap += 1
                if gap > 3:             # 3px 연속 공백 = 연속 스팬 종료 (점선 gap)
                    break
        span = float(roi_h - top_y)
        return span, hit / max(1.0, span)

    def _two_pillar_check(self, mask):
        """하단 절반 히스토그램에서 '차폭 간격의 두 기둥'(페어) 존재 여부 (0/1).
        SW 시드 피크 로직과 동일 철학 — 1L 조기 해제(좌회전 종료) 이벤트 판정 전용."""
        h_m = mask.shape[0]
        hist = np.count_nonzero(mask[h_m // 2:, :], axis=0)
        col_on = np.nonzero(hist >= 2)[0]
        if col_on.size == 0:
            return False
        splits = np.nonzero(np.diff(col_on) > 8)[0]
        peaks = []
        for g in np.split(col_on, splits + 1):
            mass = int(hist[g].sum())
            if mass >= int(self.sw_peak_min_px):
                peaks.append(float((g * hist[g]).sum()) / mass)
        if len(peaks) < 2:
            return False
        width = (self._lane_width if self._lane_width > 0
                 else self._lane_width_init) or 192.0
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                sep = abs(peaks[j] - peaks[i])
                if 0.7 * width <= sep <= 1.4 * width:
                    return True
        return False

    def _lane_heading_excluding(self, seg, fallback):
        """후보 선분 seg 의 픽셀을 뺀 나머지로 차선 주축 a_h 를 잰다 (순환 회피).
        정지선이 차선 피팅을 끌어당겨 a_h 부호가 뒤집히던 것(07-10) 대응:
        진짜 정지선을 지우면 a_h 정확 -> 채택, 차선 선분을 지우면 평행 -> 거절."""
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
        """노란 가로선(정지선) 감지 — HoughLinesP + 차선 직교성 게이트.

        판정: 길이 >= min_width_ratio x w 이고 차선과 직교(perp_tol_deg)한 선분이
        하나라도 있으면 True. Hough 채택 이유(07-10): 정지선이 차선과 T 자 한 성분이라
        컬럼평균 피팅은 100% 미검출 — Hough 는 오염 픽셀이 표를 못 얻을 뿐이라
        30/30 검출. 직교 게이트는 차선을 원리적으로 배제(차선 선분은 a_h 와 평행).
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

        # STOPLINE 모드 준비: RA + 코리도 활성 + 추적선 좌우 side 확정 시에만
        stopline_active = (int(getattr(self, 'stopline_mode', 0)) == 1
                           and self.drive_mode == 'ROUNDABOUT'
                           and self._sw_left > 0
                           and getattr(self, '_sw_interior', None) is not None)
        # 분류기 비활성 억제 (07-13 run102, 사용자 방어 설계): stopline_mode 1 인데
        # RA 중 분류기 전제(코리도 락/내부 기하)가 안 서는 프레임은 구식 경로로
        # 폴백하지 않고 crossline 자체를 억제한다. B 누출(0.06~0.18s 가변)은 전부
        # "분류기가 눈 감은 프레임"에서 나왔다 — 눈 감은 검출은 신뢰하지 않는다.
        # A 재도달 순간 코리도가 하필 언락이면 미발화 -> 2랩 자가복구 (늦는 쪽 안전).
        if (int(getattr(self, 'stopline_mode', 0)) == 1
                and self.drive_mode == 'ROUNDABOUT' and not stopline_active):
            return False
        span_ivs = []
        if stopline_active:
            _im, _ia, _ib, _iside = self._sw_interior
            s_half = self._lane_width / 2.0

            def _interior_at(ym):
                xa = _ia[0] * ym * ym + _ia[1] * ym + _ia[2]
                if _im == 'pair':
                    xb = _ib[0] * ym * ym + _ib[1] * ym + _ib[2]
                    return (xa, xb) if xa <= xb else (xb, xa)
                xb = xa + _iside * 2.0 * s_half
                return (xa, xb) if xa <= xb else (xb, xa)
        self.last_stopline_cov = 0.0

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
            # 차선 방향(a_h) 소스 (07-14 밤, __init__ crossline_sw_heading 주석):
            # 코리도 활성이면 추적 피팅의 국소 접선 — 후보가 마스크 대부분인 프레임
            # (누운 단선)에서도 진짜 차선 방향이 나와 '평행=차선' 기각이 성립한다.
            # 아니면 구식: 후보를 지우고 남은 픽셀로 잰다 (순환 회피).
            if (int(getattr(self, 'crossline_sw_heading', 0))
                    and self._sw_left > 0 and self._sw_prev_fit is not None):
                pa_h, pb_h, _pc_h = self._sw_prev_fit
                ym_roi = (float(yy1) + float(yy2)) / 2.0 + y1
                ah = float(np.clip(2.0 * pa_h * ym_roi + pb_h, -2.5, 2.5))
            else:
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
            ang_deg = float(np.degrees(np.arcsin(min(1.0, perp_cos))))
            if stopline_active:
                # 관통+정면 분류: 이진 채택 대신 정면 후보의 내부 구간을 모아
                # 프레임 커버리지로 판정한다 (아래 루프 종료 후).
                y_mid = (float(yy1) + float(yy2)) / 2.0 + y1
                ilo, ihi = _interior_at(y_mid)
                frontal = (ang_deg <= float(self.stopline_ang_max)
                           and solidity >= float(self.stopline_sol_min))
                if frontal:
                    a0 = max(ilo, min(float(x1), float(x2)))
                    b0 = min(ihi, max(float(x1), float(x2)))
                    if b0 > a0:
                        span_ivs.append((a0, b0))
                sl = (dy / dx) if abs(dx) > 1e-6 else float('inf')
                self.last_crossline_cands.append(
                    (round(sl, 3) if np.isfinite(sl) else 'vert',
                     round(ang_deg, 1), round(length, 1), round(solidity, 3),
                     'FRONTAL' if frontal else ('edge_ang' if ang_deg
                      > float(self.stopline_ang_max) else 'edge_sol'),
                     round(ah, 3), round(perp_cos, 3),
                     [int(x1), int(yy1) + y1, int(x2), int(yy2) + y1]))
                continue
            # SW 교차 게이트: 추적 중인 코리도 선을 수평으로 걸치는 선분만 인정
            sw_ok = True
            if int(getattr(self, 'crossline_sw_gate', 0)) and self._sw_prev_fit is not None \
                    and self._sw_left > 0:
                pa, pb, pc = self._sw_prev_fit
                y_mid = (float(yy1) + float(yy2)) / 2.0 + y1   # ROI 좌표로 복원
                x_fit = pa * y_mid * y_mid + pb * y_mid + pc
                mg = float(self.crossline_sw_margin)
                sw_ok = (min(float(x1), float(x2)) - mg <= x_fit
                         <= max(float(x1), float(x2)) + mg)
            ok = (perp_cos <= cos_max
                  and solidity >= float(self.crossline_min_solidity)
                  and sw_ok)
            slope = (dy / dx) if abs(dx) > 1e-6 else float('inf')
            if ok:
                verdict = 'ACCEPT'
            elif perp_cos > cos_max:
                verdict = 'reject_perp'
            elif not sw_ok:
                verdict = 'reject_sw'
            else:
                verdict = 'reject_dashed'
            self.last_crossline_cands.append(
                (round(slope, 3) if np.isfinite(slope) else 'vert',
                 round(float(np.degrees(np.arcsin(min(1.0, perp_cos)))), 1),
                 round(length, 1), round(solidity, 3), verdict,
                 round(ah, 3), round(perp_cos, 3),
                 [int(x1), int(yy1) + y1, int(x2), int(yy2) + y1]))
            if ok:
                found = True
                if not self.crossline_debug_all:
                    break                    # 진단이 필요 없으면 첫 채택에서 종료
        if stopline_active:
            merged = []
            for a0, b0 in sorted(span_ivs):
                if merged and a0 <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b0)
                else:
                    merged.append([a0, b0])
            _ilo, _ihi = _interior_at((y1 + y2) / 2.0)
            cov = sum(b0 - a0 for a0, b0 in merged) / max(1e-6, _ihi - _ilo)
            self.last_stopline_cov = round(cov, 3)
            return cov >= float(self.stopline_cov_min)
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

        # In 코스 색상 추종 상태머신 (__init__ 주석 참고). junction/fork 는 합친
        # 마스크(clean) 그대로.
        track = clean
        if (self.follow_yellow and self.course == 'in'
                and ymask is not None and wmask is not None):
            if self.drive_mode == 'ROUNDABOUT':
                self._ra_seen = True
            # 탈출창 활성 여부 (07-14 항목2): 흰 실선 이벤트가 유일 전환 트리거인 구간.
            # 이 동안 아래 레거시 해제 2종을 억제하고, 원웨이 래치 후 재래치를 막는다.
            # 창 만료(failsafe, _sw_left→0) 후에만 레거시가 되살아난다. RA 전엔 전부 무영향.
            exit_active = (self._ra_seen and self._sw_dir > 0 and self._sw_left > 0)
            if not self._following_yellow:
                # 재래치 억제 ①구설계(w_align 창 중) ②원웨이(post-RA 병합 완료 후)
                relatch_blocked = (
                    (int(getattr(self, 'w_align_block_relatch', 0)) != 0
                     and self._ra_seen and self._w_align_left > 0)
                    or (self._ra_seen and self._merge_done))
                if yellow_ratio >= self.follow_yellow_ratio and not relatch_blocked:
                    self._following_yellow = True
                    self._yellow_exit_count = 0
                    self._oneline_used = False   # 새 래치 -> 1L 1회 사용권 리셋
            elif self.drive_mode != 'ROUNDABOUT' and not exit_active:
                # WHITE 복귀 = "노랑 소실" AND "흰 우세" 연속 N프레임 (__init__ 주석).
                # ⚠️ 탈출창(exit_active) 중에는 평가 금지 — 흰 실선 이벤트가 전담.
                # 창 failsafe 만료 후에만 이 경로가 구조 수단으로 되살아난다.
                wcount = int(np.count_nonzero(wmask))
                ycount = int(np.count_nonzero(ymask))
                # 해제 문턱 스코프 (07-13 run100 분석): exit_yellow_frac 3.0
                # (yr<0.06 조기 해제)은 병합(RA 후) 전용. 진입/북상 구간은 yr 이
                # 0.05대라 전역 적용 시 래치가 12프레임마다 플랩한다 — RA 전에는
                # 검증값(0.75) 상한으로 캡.
                eff_frac = (float(self.follow_yellow_exit_yellow_frac)
                            if self._ra_seen
                            else min(0.75, float(self.follow_yellow_exit_yellow_frac)))
                yellow_gone = yellow_ratio < (self.follow_yellow_ratio * eff_frac)
                white_dom = wcount > self.follow_yellow_exit_white_ratio * max(1, ycount)
                if yellow_gone and white_dom:
                    self._yellow_exit_count += 1
                    eff_k = int(self.follow_yellow_exit_frames)
                    if self._yellow_exit_count >= eff_k:
                        self._following_yellow = False
                        self._yellow_exit_count = 0
                        self._oneline_left = 0   # 래치 해제 -> 병합 모드도 종료
                        if self._ra_seen:
                            self._merge_done = True   # 원웨이: 이후 Y 재래치 금지
                            if int(self.w_align_frames) > 0:
                                self._w_align_left = int(self.w_align_frames)
                else:
                    self._yellow_exit_count = 0
                # 블라인드 해제 (07-13): 병합부에서 노랑도 흰도 안 보이면 흰 우세
                # 조건이 영영 안 걸려 DRIVE[Y] 에 고착된다 (run83/84 의 3~4s 무검출).
                # 노랑 소실만 지속돼도 해제해 브리지(w_align + 결정층 바이어스)를 무장.
                if yellow_gone and int(self.follow_yellow_blind_release_frames) > 0:
                    self._yellow_blind_count += 1
                    if (self._following_yellow and self._yellow_blind_count
                            >= int(self.follow_yellow_blind_release_frames)):
                        self._following_yellow = False
                        self._yellow_exit_count = 0
                        self._yellow_blind_count = 0
                        self._oneline_left = 0
                        if self._ra_seen:
                            self._merge_done = True
                            if int(self.w_align_frames) > 0:
                                self._w_align_left = int(self.w_align_frames)
                else:
                    self._yellow_blind_count = 0
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
                self._yellow_dash_px = int(cv2.countNonZero(dash))
                if self._yellow_dash_px >= 30:
                    self._yellow_dash_cx = float(np.nonzero(dash)[1].mean())
                else:
                    self._yellow_dash_cx = None
                # Y+W 선행 혼합 (__init__ yw_premix 주석): 병합 접근에서만.
                # 점선 힌트(dash) 계산 뒤에 섞어 노랑 전용 신호는 오염되지 않는다.
                # _prev_drive_mode 조건 = RA->DRIVE 전이 프레임 차단 (코리도 창이
                # 같은 프레임 뒤쪽에서 무장되므로 _sw_left 만으론 엣지가 새어든다).
                if (int(getattr(self, 'yw_premix', 0))
                        and self._ra_seen and self.drive_mode != 'ROUNDABOUT'
                        and self._prev_drive_mode != 'ROUNDABOUT'):
                    if (self._sw_left <= 0
                            and cv2.countNonZero(solid)
                            >= int(self.yellow_dash_fallback_px)):
                        self._yw_carry = int(self.yw_premix_carry_frames)
                    elif self._yw_carry > 0:
                        self._yw_carry -= 1
                    if self._yw_carry > 0:
                        wf_pre = self._filter_yellow_dashes(wmask)
                        # 근거리 한정 (07-13 run108): 분기 너머 원거리 흰 도로가
                        # 세로 성분으로 높이 게이트를 통과, S커브에서 직선 오추종
                        # -> 우곡 이탈. 병합 표적 흰 실선은 하단 절반에 크게
                        # 들어오므로 상단 절반은 버린다.
                        wf_pre[:wf_pre.shape[0] // 2, :] = 0
                        if cv2.countNonZero(wf_pre) >= int(self.w_align_min_px):
                            sel = cv2.bitwise_or(sel, wf_pre)
                else:
                    self._yw_carry = 0
            else:
                sel = wmask
                if self._w_align_left > 0:
                    # 인계 창: 흰 점선(합류부 표식) 제거 + 노란 꼬리 유지
                    wf = self._filter_yellow_dashes(wmask)
                    if cv2.countNonZero(wf) >= int(self.w_align_min_px):
                        sel = wf
                    elif not int(getattr(self, 'w_align_dash_fallback', 1)):
                        # 실선 부족 = 점선(분기 표식)만 보이는 프레임: 잡지 않는다.
                        # 무검출로 두면 결정층 병합 브리지가 완만 좌호로 끌고 간다.
                        sel = np.zeros_like(wmask)
                    sel = cv2.bitwise_or(sel, ymask)
                self._yellow_dash_cx = None
                self._dash_fallback_on = False
                self._solid_ok_count = 0
                self._yw_carry = 0
            track = cv2.morphologyEx(sel, cv2.MORPH_OPEN, k) if use_morph else sel
        else:
            self._following_yellow = False
            self._yellow_exit_count = 0
            self._oneline_left = 0
            self._yellow_dash_cx = None
            self._dash_fallback_on = False
            self._solid_ok_count = 0
            self._yw_carry = 0

        # 차선 주축 a_h = dx/dy 갱신 (다음 프레임 정지선 직교 게이트용). 정지선 스캔
        # 창과 '같은 행 범위'에서만 잰다 — 전체 피팅은 휜 실선에서 국소 접선이 아닌
        # 평균 기울기가 나와 진짜 정지선을 전량 거절시켰다 (07-10 실측).
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
        # OUT 코스 상시 코리도 (07-13): S자에서 좌/우 경계가 번갈아 소실돼도
        # 시드 연속성으로 관성 추적. 락 실패 프레임은 기존 파이프라인 폴백(안전 계약).
        if (self.course == 'out' and int(getattr(self, 'sw_out_always', 0)) == 1
                and self.drive_mode == 'DRIVE' and self._sw_left <= 0):
            self._sw_left = 999999
            self._sw_len = self._sw_left
            self._sw_dir = 0
            self._sw_kind = 'out'
            self._sw_straight = 0
        # IN 전구간 코리도 무장 (07-14 밤, __init__ sw_drive_always 주석): DRIVE 인
        # 동안 창이 없으면 즉시 무장. RA 탈출~병합 완료 전은 exit 기계 전담이라 제외.
        if (self.course == 'in' and int(getattr(self, 'sw_drive_always', 0)) == 1
                and self.drive_mode == 'DRIVE' and self._sw_left <= 0
                and (not self._ra_seen or self._merge_done)):
            self._sw_left = 999999
            self._sw_len = self._sw_left
            self._sw_dir = -1 if self._following_yellow else 0
            self._sw_kind = 'drive'
            self._sw_prev_fit = None
            self._sw_interior = None
            self._sw_straight = 0
            self._sw_prev_yellow = self._following_yellow
        # 1L 조기 해제 → SW 즉시 인계 (07-14, __init__ oneline_release_* 주석):
        # 1L 진행 중 노란 실선 페어가 K프레임 연속 복원되면 좌회전 종료로 보고
        # 1L 을 닫고 같은 프레임에 접근 코리도를 무장한다.
        if (self.course == 'in' and not self._ra_seen
                and int(getattr(self, 'oneline_release_pair_k', 0)) > 0
                and self._oneline_left > 0 and self._following_yellow
                and (int(self.entry_oneline_frames) - self._oneline_left)
                >= int(getattr(self, 'oneline_release_min_hold', 20))):
            if self._two_pillar_check(track):
                self._oneline_pair_count += 1
            else:
                self._oneline_pair_count = 0
            if self._oneline_pair_count >= int(self.oneline_release_pair_k):
                self._oneline_left = 0
                self._oneline_pair_count = 0
                if (int(getattr(self, 'sw_approach_frames', 0)) > 0
                        and self._sw_left <= 0):
                    self._sw_left = int(self.sw_approach_frames)
                    self._sw_len = self._sw_left
                    self._sw_dir = -1
                    self._sw_kind = 'approach'
                    self._sw_prev_fit = None
                    self._sw_interior = None
                    self._sw_straight = 0
        else:
            self._oneline_pair_count = 0
        # IN 접근 코리도 무장 (07-14, __init__ sw_approach_frames 주석): Y래치 중 +
        # 1L 종료 후 + nl=2 연속 5f(첫 좌회전 완료 확인) + RA 전. dir=-1.
        # (위 1L 이벤트 인계가 주 경로; 이 블록은 1L 이 만료로 끝났거나 아예 안 뜬
        #  런의 예비 무장 경로.)
        if (self.course == 'in'
                and int(getattr(self, 'sw_approach_frames', 0)) > 0
                and self.drive_mode == 'DRIVE' and not self._ra_seen
                and self._following_yellow and self._oneline_left <= 0
                and self._sw_left <= 0):
            self._sw_nl2_count = self._sw_nl2_count + 1 if self._prev_nl == 2 else 0
            if self._sw_nl2_count >= 5:
                self._sw_left = int(self.sw_approach_frames)
                self._sw_len = self._sw_left
                self._sw_dir = -1            # 좌향 기대 (곡률 가드 + 실선 입력 자동)
                self._sw_kind = 'approach'
                self._sw_prev_fit = None
                self._sw_interior = None
                self._sw_straight = 0
        else:
            self._sw_nl2_count = 0

        if self.drive_mode != self._prev_drive_mode:
            if (self.drive_mode == 'ROUNDABOUT'
                    and int(self.ra_entry_oneline_frames) > 0):
                self._oneline_left = int(self.ra_entry_oneline_frames)
            elif (self._prev_drive_mode == 'ROUNDABOUT'
                    and self.drive_mode == 'DRIVE'
                    and int(self.ra_exit_oneline_frames) > 0):
                self._oneline_left = int(self.ra_exit_oneline_frames)
            # SW 코리도 창 무장 (같은 전이 엣지; __init__ 의 sw_* 주석 참고).
            # 시계가 아니라 상태 전환 이벤트 기준이라 런별 속도 차에 시작점이 안 밀린다.
            if (self.drive_mode == 'ROUNDABOUT'
                    and int(self.sw_entry_frames) > 0):
                self._sw_left = int(self.sw_entry_frames)
                self._sw_len = self._sw_left
                self._sw_dir = -1            # 진입 = 좌향 기대
                self._sw_kind = 'entry'
                self._sw_prev_fit = None     # 직전 직선 주행 피팅 잔재 유입 방지
                self._sw_interior = None
                self._sw_straight = 0
            elif (self._prev_drive_mode == 'ROUNDABOUT'
                    and self.drive_mode == 'DRIVE'
                    and int(self.sw_exit_frames) > 0):
                self._sw_kind = 'exit'
                self._sw_left = int(self.sw_exit_frames)
                self._sw_len = self._sw_left
                self._sw_dir = +1            # 탈출 = 우향 기대 (초반 한정, gate_frames 주석)
                self._sw_prev_fit = None
                self._sw_interior = None     # RA 밖에선 STOPLINE 분류기 비활성 보장
                self._sw_straight = 0
            self._prev_drive_mode = self.drive_mode

        # OUT 갈림길 시야 마스킹 (07-13 사용자 설계): 표지판 확정 방향의 반대쪽
        # 컬럼을 가린다. 조향 편향(제어)은 차선 소실 시 증발하지만, 마스크는 잘못된
        # 가지 락온 자체를 차단한다. nl==2(양차선 복원)면 해소로 보고 일시 중지.
        self._fork_cut = None
        if self.course == 'out' and float(self.fork_blind_frac) > 0.0:
            fd = self.fork_dir if self.fork_dir in ('left', 'right') else None
            if fd is not None and self._prev_fork_dir != fd:
                self._fork_blind_left = int(self.fork_blind_frames)
            self._prev_fork_dir = fd
            if fd is not None and self._fork_blind_left > 0:
                self._fork_blind_left -= 1
                if self._prev_nl != 2:
                    wcut = int(track.shape[1] * float(self.fork_blind_frac))
                    if wcut > 0:
                        track = track.copy()
                        if fd == 'left':
                            track[:, track.shape[1] - wcut:] = 0
                            self._fork_cut = ('right', wcut)
                        else:
                            track[:, :wcut] = 0
                            self._fork_cut = ('left', wcut)

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
                # 선 위치 = 최우측 클러스터(x P85), 전체평균 아님 — 누운 실선이
                # 시야에 남은 채 무장되면 평균이 실선 쪽으로 끌려 다이브(run41,
                # off -0.86). 최우측 질량 = 바깥 점선이라 P85 - 반차폭이 정답.
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

        # SW 코리도 추적 시도 (__init__ 의 sw_* 주석 참고). 기존 밴드 파이프라인은
        # 위에서 '항상' 먼저 돌았다 — 폭 학습/래치/폴백 상태기계가 평소처럼 갱신돼,
        # 창 종료 순간 run47 로직이 신선한 상태를 이어받는다(이음새 없는 복귀).
        # 유효 락 프레임만 밴드 리스트를 대체하고, 실패 프레임은 이 프레임의 기존
        # 결과(bands/near_lanes)를 그대로 통과시킨다 = 최악의 경우 현재와 동일.
        sw_locked = False
        self._sw_locked_dbg = False
        # 디버그 오버레이는 매 프레임 무조건 비운다 — src 미가용 프레임(창 활성 중
        # Y-latch 해제 등)에서 마지막 락 프레임의 곡선이 고착 표시되던 것 방지
        # (07-12 검증 리뷰 확정). 락 프레임에서 _sw_corridor_bands 가 재채움.
        self._sw_fit_dbg = []
        # 라이브 킬스위치 (07-12 검증 리뷰): 창은 무장 시점에 _sw_left 로 래치되므로
        # 주행 중 `ros2 param set /lane_node sw_entry_frames 0` 만으로는 안 풀렸다.
        # 0=off 규약 일관성 + "끄면 즉시 run47 복귀" 보장을 위해 매 프레임 확인.
        # 07-14: dir 기반 → 창 종류(_sw_kind) 매핑으로 교체 (approach 신설 + out 도
        # 자기 파라미터로 꺼지도록 — 구 코드는 out 이 sw_exit_frames 에 잘못 묶임).
        # Y래치 엣지 재타겟 (07-14 밤): drive 창 활성 중 래치가 흰↔노랑으로 바뀌는
        # 순간 입력 마스크가 통째로 바뀌므로 직전 피팅 시드를 버리고 새 마스크의
        # 히스토그램 피크로 재시드한다. Y 구간은 dir=-1 (approach 와 동일 가드).
        if (getattr(self, '_sw_kind', '') == 'drive' and self._sw_left > 0
                and self._following_yellow != self._sw_prev_yellow):
            self._sw_prev_fit = None
            self._sw_interior = None
            self._sw_dir = -1 if self._following_yellow else 0
        self._sw_prev_yellow = self._following_yellow
        if self._sw_left > 0:
            live = {'entry': self.sw_entry_frames,
                    'exit': self.sw_exit_frames,
                    'approach': getattr(self, 'sw_approach_frames', 0),
                    'out': getattr(self, 'sw_out_always', 0),
                    'drive': getattr(self, 'sw_drive_always', 0)}.get(
                        getattr(self, '_sw_kind', ''), 1)
            if int(live) <= 0:
                self._sw_left = 0
        if self._sw_left > 0:
            self._sw_left -= 1
            src = None
            # 탈출 창은 Y래치 해제 후에도 노란 입력 유지 (run53: 점선 꼬리가 병합
            # 방향의 마지막 정보 — 눈 감으면 직진 표류 13s). 꼬리 소진까지 추적,
            # 끝나면 락 자연 실패 -> 흰 추종 인계 (아래 wsteady 종결자).
            if ymask is not None and (self._following_yellow
                                      or self._sw_dir > 0):
                # dir<=0 = 진입/접근/drive[Y] 계열: solid 입력 (병합 사선 노이즈 제거)
                mode_in = (self.sw_exit_input if self._sw_dir > 0
                           else self.sw_entry_input)
                sw_src = (self._filter_yellow_dashes(ymask)
                          if str(mode_in) == 'solid' else ymask)
                src = (cv2.morphologyEx(sw_src, cv2.MORPH_OPEN, k)
                       if use_morph else sw_src)
            elif (self._sw_dir == 0 and wmask is not None
                    and (self.course == 'out'
                         or getattr(self, '_sw_kind', '') == 'drive')):
                # 상시 코리도 흰 입력 (OUT 전구간 / IN drive[W] 구간):
                # 양쪽 실선 -> pair 락(중점) 적합
                src = (cv2.morphologyEx(wmask, cv2.MORPH_OPEN, k)
                       if use_morph else wmask)
            if src is not None and self._fork_cut is not None:
                side, wcut = self._fork_cut
                src = src.copy()
                if side == 'right':
                    src[:, src.shape[1] - wcut:] = 0
                else:
                    src[:, :wcut] = 0
            # 탈출 개구부 컷 (__init__ sw_exit_cut_frames 주석): 발화 직후 링쪽 제거
            if (src is not None and self._sw_kind == 'exit' and self._sw_dir > 0
                    and int(getattr(self, 'sw_exit_cut_frames', 0)) > 0
                    and (self._sw_len - self._sw_left)
                    <= int(self.sw_exit_cut_frames)
                    and self.fork_dir in ('left', 'right')):
                wcut = int(src.shape[1] * float(self.sw_exit_cut_frac))
                if wcut > 0:
                    src = src.copy()
                    if self.fork_dir == 'right':
                        src[:, :wcut] = 0                      # 링(좌) 제거
                    else:
                        src[:, src.shape[1] - wcut:] = 0       # 링(우) 제거
            if src is not None:
                self._sw_gate_dir = self._sw_dir
                self._sw_open = False
                if self._sw_dir > 0:
                    age = self._sw_len - self._sw_left
                    if age <= int(self.sw_exit_open_frames):
                        # 개방 구간 (run54): 좌향 코리도 허용, 정지선은
                        # _sw_corridor_bands 의 |기욺| 절대 상한이 기각
                        self._sw_gate_dir = 0
                        self._sw_open = True
                    elif age > int(self.sw_exit_gate_frames):
                        self._sw_gate_dir = 0   # 링 배제 임무 종료 -> 방향 게이트 해제
                sw = self._sw_corridor_bands(src, w, roi_h, band_h)
                if sw is not None:
                    bands, near_lanes, sw_boxes = sw
                    band_windows = list(band_windows) + sw_boxes
                    sw_locked = True
                    self._sw_locked_dbg = True
            # 탈출창 종료 = 흰 '실선' 이벤트 (07-14 재설계, 항목2). 시간/프레임 창이
            # 아니라 "끊김 없이 이어지는 흰 실선이 최하단에 나타나면" 전환한다.
            # 판별: 연속 스팬(점선 토막 최대길이 초과 = 실선) + 스팬내 점유율 + N프레임
            # 디바운스(공간·시간 이중방어). bbox 높이 필터는 세로 점선을 실선으로 오판해
            # 부적합 — 스팬 추적을 쓴다. sw_exit_input='raw'라 SW 입력엔 흰이 안 들어오므로
            # 흰 검출은 여기서 wmask 를 직접 본다.
            if self._sw_dir > 0:
                wsolid = False
                if wmask is not None:
                    wsrc = (cv2.morphologyEx(wmask, cv2.MORPH_OPEN, k)
                            if use_morph else wmask)
                    span, sol = self._white_solid_span(wsrc, roi_h, band_h)
                    wsolid = (span >= float(self.sw_exit_white_min_span_px)
                              and sol >= float(self.sw_exit_white_solidity_min))
                self._sw_white_confirm = self._sw_white_confirm + 1 if wsolid else 0
                if self._sw_white_confirm >= int(self.sw_exit_white_confirm_frames):
                    # 확정: 그 프레임부터 탈출창 종료 + Y래치 해제. 다음 프레임부터
                    # 기존 흰 추종(else: sel=wmask)이 자연 인계. 원웨이 병합 래치 장전.
                    self._following_yellow = False
                    self._sw_left = 0
                    self._sw_white_confirm = 0
                    self._merge_done = True
                    if self._ra_seen and int(self.w_align_frames) > 0:
                        self._w_align_left = int(self.w_align_frames)
        else:
            self._sw_dir = 0

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
        # 흰 인계 헤딩 정렬 (w_align 창): 사선 관통 프레임은 offset~0 이라 기울기
        # (_lane_heading, track 주축 EMA)로만 관측된다. 선 방향으로 기수 정렬.
        # 나란히 만나면 기울기~0 = 무작용.
        if self._w_align_left > 0 and not self._following_yellow:
            self._w_align_left -= 1
            if lane_found and float(self.w_align_gain) > 0.0:
                raw_offset = float(max(-1.0, min(
                    1.0, raw_offset - float(self.w_align_gain) * self._lane_heading)))
        junction = self._detect_junction(clean, w)
        fork = self._detect_fork(clean, w)
        # 개구부 fk 소실 규칙: merge_zone 중 fork 서명 프레임 = 소실 취급 — 개구부
        # 부챗살의 nl=2 쐐기 추종(랩 종료 이탈 5전 5패, run31/37: 순항 오발 0% /
        # 쐐기 구간 38~63% 점화) 차단, ra_blind_bias 가 접선 헤딩으로 건너게 한다.
        # 단 SW 락 프레임은 제외 — 락은 방향 게이트를 통과한 검증된 시야.
        if (self.drive_mode == 'ROUNDABOUT' and self.merge_zone and fork
                and not sw_locked):
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
        # 합류부 재획득 단선 = 구조상 우측(바깥) 경계 고정 (run22~24 언더스티어 서명
        # 대응). 창 안 + "소실->단선 재획득"에만 적용 — RA 전체 적용은 과범위(run25).
        if (self.drive_mode == 'ROUNDABOUT' and self.merge_zone
                and self._reacq_right_active):
            return line - half, 1
        # 대신 "점선의 반대편 경계"라는 코스 구조로 좌/우를 정한다. 오른쪽 실선이
        # 차 왼쪽에 보여도 (점선이 그보다 더 왼쪽에 있으면) 오른쪽 경계로 맞게
        # 분류돼 중심을 실선의 왼쪽(-half)에 둔다.
        if self._yellow_dash_cx is not None:
            if line < self._yellow_dash_cx:
                return line + half, 1   # 실선이 점선보다 왼쪽 = 왼쪽 경계
            # RA 순환 개구부 유출 차단 (07-13 run105): 개구부 점선 호 '너머'
            # (오른쪽)의 단독 실선은 링 밖 접선 도로 실선이다 — 우측 경계로
            # 해석해 반차폭 왼쪽을 지켜도 선 자체가 밖으로 뻗어 유출된다(실측).
            # 프레임 기각 -> 무검출 관성으로 개구부 통과 (개구부에서 아무것도
            # 안 보일 때와 같은 검증된 경로). 점선 질량/여유 하한으로 잡음 배제.
            if (self.drive_mode == 'ROUNDABOUT'
                    and int(getattr(self, '_yellow_dash_px', 0)) >= 150
                    and line - self._yellow_dash_cx > 25.0):
                return None, 0
            return line - half, 1       # 실선이 점선보다 오른쪽 = 오른쪽 경계
        # 시간 연속성 분류: 직전 중심에 더 가까운 해석을 고른다 — 화면 반쪽 기준은
        # 단선이 중앙을 넘나드는 커브에서 좌/우가 매 프레임 뒤집혀 이탈 (07-11 첫
        # 자율주행 2연속 실패 원인).
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

    # ---------- 슬라이딩 윈도우(SW) 코리도 추적 (RA 진입/탈출 전이 창 전용) ----------
    def _sw_corridor_bands(self, src, w, roi_h, band_h):
        """상자 체인 + 2차 피팅으로 코리도 경계를 추적, 밴드 리스트를 합성.

        반환 (bands, near_lanes, debug_boxes) 또는 None(락 실패 = 호출측 폴백).
        bands 는 기존 [(cx, weight, i)] 형식 — 하류 무변경 인계.
        단계: ① 정지선 행 제거 ② 시드(직전 피팅 교점 + 히스토그램 피크)
        ③ 상자 체인(이동 상한 = 누운 실선/반대 가지 차단, 빈 상자 = 추세 유지)
        ④ 2차 피팅 + 게이트(적중 상자/픽셀/잔차/방향/개방기욺 상한)
        ⑤ 페어(차폭 간격) -> 중점 nl=2, 단선 -> _combine_lr 식 좌/우 분류
        ⑥ 관측 y범위 밖 외삽 금지(클램프).
        """
        self._sw_fit_dbg = []
        m = src.copy()
        # (1) 정지선(가로줄) 행 제거
        row_on = np.count_nonzero(m, axis=1)
        m[row_on > int(w * float(self.sw_cross_row_frac)), :] = 0
        nb = max(3, int(self.sw_num_boxes))
        bh = max(2, roi_h // nb)
        margin = max(4, int(self.sw_box_margin))
        half = ((self._lane_width / 2.0) if self._lane_width > 0
                else (w * self.single_line_offset))

        # (2) 시드 수집: 직전 피팅 우선, 다음 하단 히스토그램 피크(질량 순)
        seeds = []
        if self._sw_prev_fit is not None:
            pa, pb, pc = self._sw_prev_fit
            yb = float(roi_h - 1)
            seeds.append(pa * yb * yb + pb * yb + pc)
        # 개방 구간엔 전체 높이 — 코리도가 정지선 너머(상단)에만 있는 프레임 대응
        seed_top = 0 if self._sw_open else roi_h // 2
        hist = np.count_nonzero(m[seed_top:, :], axis=0)
        col_on = np.nonzero(hist >= 2)[0]
        if col_on.size:
            splits = np.nonzero(np.diff(col_on) > 8)[0]
            peaks = []
            for g in np.split(col_on, splits + 1):
                mass = int(hist[g].sum())
                if mass >= int(self.sw_peak_min_px):
                    peaks.append((mass, float((g * hist[g]).sum()) / mass))
            peaks.sort(reverse=True)
            seeds += [cx for _, cx in peaks[:max(1, int(self.sw_max_peaks))]]
        if not seeds:
            return None

        # (3) 상자 체인
        def chain(seed_x):
            x = float(seed_x)
            trend = 0.0
            pxs, pys, hit = [], [], 0
            boxes = []
            for i in range(nb):
                y1 = roi_h - i * bh
                y0 = max(0, y1 - bh)
                if y1 - y0 < 2:
                    break
                x0 = int(max(0, x - margin))
                x1 = int(min(w, x + margin))
                if x1 - x0 < 4:
                    break
                ys_w, xs_w = np.nonzero(m[y0:y1, x0:x1])
                cx = None
                # max(1, ..) 클램프: 라이브로 0 을 넣으면 빈 상자 mean() 이 NaN 을
                # 만들어 체인이 오염된다 (07-12 검증 리뷰 — 실차에서 노드 사망 경로)
                if xs_w.size >= max(1, int(self.sw_min_box_px)):
                    raw_cx = float(xs_w.mean()) + x0
                    trend = float(np.clip(raw_cx - x, -float(self.sw_max_shift),
                                          float(self.sw_max_shift)))
                    x += trend
                    cx = x
                    pxs.append(xs_w.astype(np.float64) + x0)
                    pys.append(ys_w.astype(np.float64) + y0)
                    hit += 1
                else:
                    x += trend   # 점선 틈/짧은 소실: 직전 추세로 계속 위로
                boxes.append((x0, y0, x1, y1, cx))
                if x < -margin or x > w + margin:
                    break
            return pxs, pys, hit, boxes

        # (4) 피팅 + 게이트
        def fit_chain(pxs, pys, hit):
            # max(1, ..) + 빈 리스트 가드: 라이브로 sw_min_boxes=0 을 넣으면 빈
            # 체인이 게이트를 통과해 np.concatenate([]) ValueError 로 노드가
            # 죽는다 (07-12 검증 리뷰 — 실차 주행 중 인지 상실 경로).
            if hit < max(1, int(self.sw_min_boxes)) or not pxs:
                return None
            xs = np.concatenate(pxs)
            ys = np.concatenate(pys)
            if xs.size < int(self.sw_min_pixels):
                return None
            y_lo, y_hi = float(ys.min()), float(ys.max())
            if (y_hi - y_lo) < 2.0 * bh:
                return None
            try:
                a, b, c = np.polyfit(ys, xs, 2)
            except Exception:
                return None
            pred = a * ys * ys + b * ys + c
            resid = float(np.mean(np.abs(pred - xs)))
            if resid > float(self.sw_max_resid_px):
                return None
            x_top = a * y_lo * y_lo + b * y_lo + c
            x_bot = a * y_hi * y_hi + b * y_hi + c
            lean = float(x_top - x_bot)   # 음수 = 좌향(위로 갈수록 왼쪽)
            # 방향 게이트: 기대 '반대' 방향 기욺만 기각한다 (직선은 통과 —
            # 진입 초반의 곧은 접근로 선, 탈출 후반의 펴진 탈출로 선이 대상).
            if lean * float(self._sw_gate_dir) < -float(self.sw_wrongdir_px):
                return None
            # 개방 구간: 방향 게이트 대신 절대 기욺 상한 — BEV 를 비스듬히
            # 가로지르는 정지선(실측 |lean| 102~148)을 기각, 코리도는 통과
            if self._sw_open and abs(lean) > float(self.sw_open_max_lean_px):
                return None
            return {'abc': (float(a), float(b), float(c)),
                    'y_lo': y_lo, 'y_hi': y_hi, 'resid': resid,
                    'lean': lean, 'x_bot': float(x_bot)}

        fits = []
        for s in seeds:
            pxs, pys, hit, boxes = chain(s)
            f = fit_chain(pxs, pys, hit)
            if f is None:
                continue
            # 중복 제거: 하단 교점이 기존 채택과 사실상 같은 선이면 잔차 좋은 쪽만
            dup = False
            for prev in fits:
                if abs(prev['x_bot'] - f['x_bot']) < 0.3 * half:
                    dup = True
                    if f['resid'] < prev['resid']:
                        prev.update(f)
                        prev['boxes'] = boxes
                    break
            if not dup:
                f['boxes'] = boxes
                fits.append(f)
        if not fits:
            return None

        # 곡률 부호 가드 (run87 실측): 링 순환(진입 창) 중 B 개구부의 우곡률 가지
        # (a=+0.006 대역)를 물면 링 밖으로 끌려간다. 우곡률 상한 초과 피팅은 버리고
        # 시드 연속성(직전 피팅)으로 관성 통과한다. race_dir=left 전제.
        if self._sw_dir < 0 and float(getattr(self, 'sw_curv_max_a', 0.0)) > 0.0:
            fits = [f for f in fits if f['abc'][0] <= float(self.sw_curv_max_a)]
            if not fits:
                return None

        def ev(f, y):
            a, b, c = f['abc']
            yc = min(max(y, f['y_lo']), f['y_hi'])   # (6) 외삽 금지: 스팬 클램프
            return a * yc * yc + b * yc + c

        # (5) 코리도 해석: 두 경계(pair) 우선, 아니면 단선 + 연속성 분류
        fits.sort(key=lambda f: f['x_bot'])
        pair = None
        if (getattr(self, '_sw_kind', '') == 'approach'
                and self._prev_center is not None):
            # 접근 창 (07-14): 좌측 합류부 차선이 우리 왼선과 '차폭 간격 가짜 페어'를
            # 이룰 수 있다 — 첫 유효 페어(최좌측) 대신 직전 중심에 가장 가까운 페어.
            best_d = None
            for i in range(len(fits) - 1):
                for j in range(i + 1, len(fits)):
                    sep = fits[j]['x_bot'] - fits[i]['x_bot']
                    if 0.7 * (2.0 * half) <= sep <= 1.4 * (2.0 * half):
                        mid = (fits[i]['x_bot'] + fits[j]['x_bot']) / 2.0
                        d = abs(mid - self._prev_center)
                        if best_d is None or d < best_d:
                            best_d, pair = d, (fits[i], fits[j])
        else:
            for i in range(len(fits) - 1):
                for j in range(i + 1, len(fits)):
                    sep = fits[j]['x_bot'] - fits[i]['x_bot']
                    if 0.7 * (2.0 * half) <= sep <= 1.4 * (2.0 * half):
                        pair = (fits[i], fits[j])
                        break
                if pair:
                    break

        bands = []
        chosen = []
        if pair:
            fl, fr = pair
            chosen = [fl, fr]
            y_min = min(fl['y_lo'], fr['y_lo'])
            y_max = max(fl['y_hi'], fr['y_hi'])
            for i in range(self.num_bands):
                yc = roi_h - (i + 0.5) * band_h
                if yc < y_min - band_h or yc > y_max + band_h:
                    continue
                bands.append(((ev(fl, yc) + ev(fr, yc)) / 2.0,
                              float(self.num_bands - i), i))
            near_lanes = 2
            best = fl if fl['resid'] <= fr['resid'] else fr
        else:
            best = min(fits, key=lambda f: f['resid'])
            chosen = [best]
            # 좌/우 분류 우선순위 = _combine_lr 과 일치 (07-12 검증 리뷰):
            # 재획득 우측고정 > 점선 힌트(이격 조건) > 직전 중심 연속성 > 기본값.
            # 연속성만 쓰면 run22~24 오분류 서명이 SW 프레임에서 재발.
            if self._sw_dir > 0:
                # 항목1 (07-14): 탈출 단선의 점선/실선 판별 (Y래치 상태 무관).
                #  · 점선이면 → 무조건 바깥(우측) 경계 고정 (병합 굽이는 좌향, 남은
                #    점선은 바깥선 — 인접 진입 커넥터 점선 오물림 방지, run83/84/95).
                #  · 실선이면 → SW 락 포기(None). 탈출 중 보이는 노란 실선은 두고 떠나는
                #    링 본선/커넥터 안쪽 선일 확률이 높아 '바깥 경계' 가정이 틀림 —
                #    일반 파이프라인(점선힌트/연속성/run105 유출차단, 76~80 검증)에 위임.
                occ = self._curve_occupancy(m, best['abc'], best['y_lo'], best['y_hi'])
                if occ > float(self.sw_exit_dash_occupancy_max):
                    return None                      # 실선 → 락 실패 계약 = 일반 폴백
                side = float(self.sw_side_default)   # 점선 → 바깥(우측) 경계
            elif (self.drive_mode == 'ROUNDABOUT' and self.merge_zone
                    and self._reacq_right_active):
                side = -1.0                      # 재획득 선 = 구조상 우측(바깥) 경계
            elif (self._yellow_dash_cx is not None
                    and abs(best['x_bot'] - self._yellow_dash_cx) > 0.5 * half):
                side = (1.0 if best['x_bot'] < self._yellow_dash_cx else -1.0)
            elif self._prev_center is not None:
                cand_l = best['x_bot'] + half    # 좌측 경계 해석의 중심
                cand_r = best['x_bot'] - half    # 우측 경계 해석의 중심
                side = (1.0 if abs(cand_l - self._prev_center)
                        <= abs(cand_r - self._prev_center) else -1.0)
            else:
                side = float(self.sw_side_default)   # -1 = 추적선이 우측 경계
            for i in range(self.num_bands):
                yc = roi_h - (i + 0.5) * band_h
                if yc < best['y_lo'] - band_h or yc > best['y_hi'] + band_h:
                    continue
                bands.append((ev(best, yc) + side * half,
                              float(self.num_bands - i), i))
            near_lanes = 1
        if not bands:
            return None

        # 상태/디버그 갱신 (락 성공시에만 — 실패 프레임은 시드 연속성 유지)
        self._sw_last_side = float(side) if near_lanes == 1 else 0.0
        if pair:
            self._sw_interior = ('pair', fl['abc'], fr['abc'], 0.0)
        else:
            self._sw_interior = ('single', best['abc'], None, float(side))
        self._sw_prev_fit = best['abc']
        self._sw_last_lean = best['lean']
        boxes_dbg = []
        for f in chosen:
            boxes_dbg += f.get('boxes', [])
            self._sw_fit_dbg.append(
                [(int(round(ev(f, y))), int(y))
                 for y in range(int(f['y_lo']), int(f['y_hi']) + 1, 4)])
        return bands, near_lanes, boxes_dbg

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
        """디버그 이미지 (/perception/lane/debug). 하단 = BEV ROI + mask(시안) +
        탐색 창(마젠타) + 차선 중심(빨강)/이미지 중심(초록) + 상태 텍스트.
        상단 = 원본 ROI + BEV 소스 사각형(노랑, src_ratio 튜닝용)."""
        dbg = roi.copy()
        dbg[mask > 0] = (255, 255, 0)                                     # HSV 로 선택된 차선 픽셀 (시안)
        for x0, y0, x1, y1, bcx in band_windows:                          # guided 탐색 창
            cv2.rectangle(dbg, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (255, 0, 255), 1)
            if bcx is not None:
                cv2.circle(dbg, (int(bcx), int((y0 + y1) / 2)), 2, (0, 165, 255), -1)
        for poly in self._sw_fit_dbg:                                     # SW 코리도 피팅 곡선 (주황)
            if len(poly) >= 2:
                cv2.polylines(dbg, [np.int32(poly)], False, (0, 165, 255), 2)
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
            if self._sw_left > 0 or self._sw_locked_dbg:
                # SW 코리도 창 활성: +는 이번 프레임 락(밴드 대체), 없으면 폴백 통과
                follow_tag += '[SW+]' if self._sw_locked_dbg else '[SW]'
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
