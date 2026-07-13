"""lane_node: OpenCV 기반 차선 추종.

카메라 이미지를 구독해 LaneDetector 를 돌리고, LaneState(정규화된 offset)를
publish 한다. 옵션으로 디버그 이미지를 다시 publish 해서 키트의 모니터
대시보드에서 검출 과정을 지켜볼 수 있다.
"""

import json

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from rcl_interfaces.msg import SetParametersResult

from perception_msgs.msg import LaneState
from perception.lane_detector import LaneDetector


class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('crossline_debug', True)   # 임시: 정지선 후보 진단 토픽 발행
        # 정지선 직교성 게이트: 차선 대비 90°에서 이 각도(도) 이내면 정지선.
        # 0 = 비활성(측정 모드). 곡선 위에서 비스듬히 접근해도 통과한다.
        self.declare_parameter('crossline_perp_tol_deg', 20.0)
        # BEV 가로/세로 스케일비(sx/sy). 07-10 실측 r=1.91. dst_ratio 변경 시 재측정.
        self.declare_parameter('crossline_bev_aspect', 1.91)
        self.declare_parameter('lane_heading_alpha', 0.2)
        self.declare_parameter('crossline_exclude_px', 6.0)  # 후보 선분 배제 반경(px)
        # HoughLinesP 파라미터 (정지선 선분 검출).
        self.declare_parameter('crossline_hough_thresh', 25)    # 누적 투표 임계
        self.declare_parameter('crossline_hough_max_gap', 10)   # 선분 이어붙이기 최대 간격(px). 8 이하면 진짜 정지선도 못 잇는다
        self.declare_parameter('crossline_min_solidity', 0.80)  # 선분 위 픽셀 충실도 하한 (이어붙인 점선 배제)
        self.declare_parameter('crossline_sw_gate', 1)           # SW 코리도 교차 게이트 (run76~80 실전 검증: B 개구부 페인트 기각)
        self.declare_parameter('sw_curv_max_a', 0.003)          # 진입 창 우곡률 상한 (B 가지 오물림 방지, 0=off)
        self.declare_parameter('stopline_mode', 0)               # 관통+정면 정지선 분류기 (0=레거시)
        self.declare_parameter('stopline_ang_max', 15.0)
        self.declare_parameter('stopline_cov_min', 0.35)
        self.declare_parameter('stopline_sol_min', 0.55)
        self.declare_parameter('w_align_dash_fallback', 0)
        self.declare_parameter('follow_yellow_blind_release_frames', 12)
        self.declare_parameter('crossline_sw_margin', 40.0)      # 교차 판정 여유(px)
        self.declare_parameter('crossline_debug_all', False)    # 채택 후에도 전 선분 진단
        self.declare_parameter('debug_topic', '/perception/lane/debug')
        self.declare_parameter('jpeg_quality', 80)
        # debug_hz: 디버그 그리기+JPEG 인코딩 주기 제한 (주행 로직은 20Hz 유지).
        # 0 이하 = 매 프레임. 라이브 조절 가능.
        self.declare_parameter('debug_hz', 5.0)   # 07-09: 10->5 전력 다이어트 (그리기+JPEG 인코딩)
        self.declare_parameter('roi_top_ratio', 0.35)   # 상단 35% crop (아래 65% 사용)
        self.declare_parameter('bright_thresh', 160)
        self.declare_parameter('min_pixels', 40)
        # --- 강건한 차선 추종 (멀티밴드 look-ahead, 곡률, 스무딩) ---
        self.declare_parameter('num_bands', 4)        # 커브용 look-ahead 밴드 개수
        self.declare_parameter('morph_kernel', 3)     # 노이즈 정리(반사광/점선); 0=끔
        self.declare_parameter('width_ema', 0.1)      # 차선 폭 기억값 적응 속도
        self.declare_parameter('smooth_alpha', 0.5)   # offset 스무딩 (낮을수록 부드러움)
        self.declare_parameter('race_dir', 'left')                 # 'left'/'right' 마스터; junction_side 를 여기서 도출
        self.declare_parameter('junction_side', 'right')           # race_dir 이 left/right 가 아닐 때만 사용
        self.declare_parameter('junction_dash_transitions', 3)     # 세로 방향 on/off 변화 횟수 이상 => 점선 판정
        self.declare_parameter('junction_min_row_pixels', 2)       # 이 값 초과 픽셀이면 해당 row 를 '라인'으로 간주
        self.declare_parameter('junction_gap_rows', 2)             # 완전 개방(빈 구간) 폴백
        # --- 색상 마스킹 (흰색 + 노란색 차선) ---
        self.declare_parameter('mask_mode', 'hsv')            # 'hsv' (white|yellow) 또는 'gray'
        self.declare_parameter('use_white', True)
        self.declare_parameter('use_yellow', True)            # 노란색 회전교차로 차선
        self.declare_parameter('white_hsv_lo', [0, 0, 180])   # H,S,V 하한 (OpenCV H 0-179)
        self.declare_parameter('white_hsv_hi', [179, 60, 255])
        self.declare_parameter('yellow_hsv_lo', [18, 45, 110])   # 옅은/바랜 노란 선을 위해 S 하한을 낮춤
        self.declare_parameter('yellow_hsv_hi', [40, 255, 255])
        # 빨간 노면(ArUco 장애물 구간). 빨강은 H 양끝에 걸쳐 두 구간을 합쳐 잡는다.
        # red_ratio 만 계산하고 차선 mask 에는 넣지 않으므로 조향에 영향 없다.
        self.declare_parameter('use_red', True)
        self.declare_parameter('red_hsv_lo1', [0, 80, 60])
        self.declare_parameter('red_hsv_hi1', [10, 255, 255])
        self.declare_parameter('red_hsv_lo2', [170, 80, 60])
        self.declare_parameter('red_hsv_hi2', [179, 255, 255])
        # --- bird-eye view (기본 ON) ---
        # [x1,y1,...,x4,y4] (TL,TR,BR,BL), ROI 대비 0..1. 마운트 변경 시 재캘리 필수:
        # 직선 구간 흰선 2개를 직선 피팅, y=0.342h/0.965h 의 차선 x 를 src 꼭짓점으로
        # (워프 후 0.20w/0.80w). 합격: 기둥 slope ~0, 폭 192±5px @x=64/256.
        self.declare_parameter('use_birdeye', True)
        # 07-11 재캘리 (카메라 재조정 후): 워프 검증 좌 +0.003/우 +0.000, x=64/256 정확.
        self.declare_parameter('birdeye_src_ratio', [0.2598, 0.342, 0.7320, 0.342, 0.9264, 0.965, 0.0376, 0.965])
        self.declare_parameter('birdeye_dst_ratio', [0.20, 0.00, 0.80, 0.00, 0.80, 1.00, 0.20, 1.00])
        # --- 가이드 밴드 탐색 (기본 ON) ---
        self.declare_parameter('use_guided_band', True)
        self.declare_parameter('guide_margin_px', 60)         # 이전 밴드 중심 주변 ± px 범위만 탐색
        self.declare_parameter('guide_margin_growth_px', 10)  # 먼 밴드로 갈수록 margin += i*growth
        self.declare_parameter('guide_min_pixels', 20)        # 좁은 창이므로 min_pixels 보다 낮게
        self.declare_parameter('guide_use_previous_frame', True)
        self.declare_parameter('guide_max_jump_px', 80)       # 밴드 간 중심 점프를 이 값으로 제한
        # --- look-ahead 조향 블렌딩 (부분적 pure-pursuit; 기본 ON) ---
        self.declare_parameter('use_lookahead_control', True)
        self.declare_parameter('near_weight', 0.7)            # 가장 가까운 밴드 가중치
        self.declare_parameter('lookahead_weight', 0.3)       # 먼 밴드 가중치
        self.declare_parameter('lookahead_band_index', -1)    # -1 = 검출된 가장 먼 밴드
        self.declare_parameter('adaptive_lookahead', False)   # 급커브에서 lookahead 가중치 강화
        self.declare_parameter('curve_lookahead_weight', 0.4)
        self.declare_parameter('curve_lookahead_thresh', 0.25)
        # --- yellow crossline (노란 가로선; 회전교차로 진입/탈출 위치 신호) ---
        self.declare_parameter('crossline_roi_top_ratio', 0.40)     # 스캔 창 상단 (대각선 위해 넓힘)
        self.declare_parameter('crossline_roi_bottom_ratio', 0.95)  # 스캔 창 하단
        self.declare_parameter('crossline_min_width_ratio', 0.20)   # 성분 "주축 길이" 최소 비율 (w 기준)
        self.declare_parameter('crossline_min_area_px', 60)         # 성분 최소 픽셀 (dash/노이즈 필터)
        # --- 좌/우 갈림길 (fork) 감지 + 브랜치 선택 ---
        self.declare_parameter('fork_topic', '/decision/fork_dir')  # decision 이 확정 방향 publish
        self.declare_parameter('state_topic', '/decision/state')    # decision 주행 모드 -> BEV 표시
        self.declare_parameter('merge_zone_topic', '/decision/merge_zone')  # 합류부 창 플래그 (단선=우측 규칙 스코프)
        self.declare_parameter('fork_scan_top_ratio', 0.0)          # 분기 스캔밴드 상단(BEV far)
        self.declare_parameter('fork_scan_bottom_ratio', 0.5)       # 분기 스캔밴드 하단
        self.declare_parameter('fork_col_min_ratio', 0.15)          # 컬럼=라인 판정 세로픽셀 비율
        self.declare_parameter('fork_min_groups', 3)                # 라인 군집 개수 이상 => 분기
        self.declare_parameter('fork_span_ratio', 0.0)              # 바깥라인 간격 폴백 (0=비활성, 07-10 실측상 무용)
        self.declare_parameter('fork_seed_px', 90)                  # 브랜치 선택 시드 이동량(px)
        # --- 노란색 우선 추종 (In 코스: 노란 진입 커브/회전교차로 링) ---
        self.declare_parameter('follow_yellow', True)               # In 코스 색상 추종 상태머신 on/off
        # 흰 구간 노이즈 바닥 0.0005 (07-10, 232f). 0.005/0.01 은 원거리 노란 마킹
        # 조기 래치 이탈(07-11, 3회) -> 0.03. 07-12: 분기 yr 피크 런별 0.026~0.095
        # 실측 -> 라이브 0.02 운용 (aux.sh).
        self.declare_parameter('follow_yellow_ratio', 0.03)         # 이 노란비율 이상 -> YELLOW 모드 진입
                                                                    # (노란선이 점선이라 yr 이 낮음)
        self.declare_parameter('follow_yellow_exit_white_ratio', 1.0)  # 흰픽셀 > 이 배율*노란픽셀 (해제 조건 일부)
        self.declare_parameter('follow_yellow_exit_yellow_frac', 0.5)  # 해제 노랑문턱 = 진입문턱*이 비율 (노랑 보이면 유지)
        self.declare_parameter('follow_yellow_exit_frames', 10)        # 해제 조건 연속 프레임 (플리커 방지)
        self.declare_parameter('follow_yellow_exit_frames_exit', 4)    # 탈출(SW)창 내 해제 연속 프레임 (0=전역값)
        # 흰 인계 창 (run59 병합 관통 대응) — 상세는 lane_detector 주석
        self.declare_parameter('w_align_frames', 60)    # 창 길이 (~3s). 0=off
        self.declare_parameter('w_align_gain', 0.4)     # 헤딩 정렬 게인 (0=보정 off)
        self.declare_parameter('w_align_min_px', 80)    # 점선 필터 폴백 문턱
        self.declare_parameter('filter_yellow_dashes', True)           # Y추종 중 점선/정지선 track 제외 (실선만)
        self.declare_parameter('yellow_solid_min_h_ratio', 0.30)       # "실선" 판정 최소 세로 비율
        self.declare_parameter('yellow_dash_fallback_px', 120)         # 실선 픽셀 이 미만이면 점선 포함 폴백
        self.declare_parameter('dash_fallback_exit_frames', 30)        # 폴백 해제(실선 복귀)에 필요한 연속 프레임 (1s)
        self.declare_parameter('yellow_heading_gain', 0.0)   # a_h 널뜀으로 OFF (07-11, detector 주석)
        # 연속성 가드/1L 계열 — 근거는 lane_detector.__init__ 주석 참고.
        self.declare_parameter('center_jump_max_ratio', 0.10)   # 0.15 는 쐐기 당김 통과 -> 0.10 확정
        self.declare_parameter('entry_oneline_frames', 80)   # ~4s (run17: 2s 는 저속 미완 만료)
        self.declare_parameter('oneline_near_bands', 2)      # 1L 하단 밴드 수 (run40)
        self.declare_parameter('ra_entry_oneline_frames', 0)     # 0=off. 재활성 시 160(~8s)
        self.declare_parameter('ra_exit_oneline_frames', 0)      # 0=off. 재활성 시 60(~3s)
        # --- SW 코리도 추적 — 상세는 lane_detector.__init__ sw_* 주석. 락 프레임만
        # 밴드 대체(실패 = 폴백). 전부 라이브: ros2 param set /lane_node sw_entry_frames 440
        self.declare_parameter('sw_entry_frames', 1200)   # RA 진입 창(프레임). 0=off. run79: 저전압 랩 지연으로 440(22s) 만료 후 개구부 이탈 -> 60s
        self.declare_parameter('sw_exit_frames', 0)       # RA 탈출 창. 0=off. 권장 60(~3s)
        self.declare_parameter('sw_entry_input', 'solid') # 진입 입력: solid(병합 사선 제거) | raw
        self.declare_parameter('sw_exit_input', 'raw')    # 탈출 입력: 좌측 경계가 점선 -> raw 필수
        self.declare_parameter('sw_num_boxes', 9)         # 상자 개수
        self.declare_parameter('sw_box_margin', 30)       # 상자 반폭(px)
        self.declare_parameter('sw_max_shift', 20)        # 상자당 이동 상한(px)
        self.declare_parameter('sw_min_box_px', 8)        # 상자 적중 최소 픽셀
        self.declare_parameter('sw_min_boxes', 3)         # 유효 피팅 최소 적중 상자
        self.declare_parameter('sw_min_pixels', 50)       # 유효 피팅 최소 총 픽셀
        self.declare_parameter('sw_wrongdir_px', 8.0)     # 기대 반대 방향 기욺 기각 문턱(px)
        self.declare_parameter('sw_max_resid_px', 12.0)   # 피팅 평균 잔차 상한(px)
        self.declare_parameter('sw_peak_min_px', 40)      # 시드 피크 최소 질량
        self.declare_parameter('sw_max_peaks', 3)         # 프레임당 피크 시드 수
        self.declare_parameter('sw_cross_row_frac', 0.45) # 정지선 행 제거 문턱(가로 점유율)
        self.declare_parameter('sw_side_default', -1)     # 단선 분류 폴백(-1=우측 경계)
        self.declare_parameter('sw_exit_straight_k', 5)   # 탈출 조기해제 직선 연속 프레임
        self.declare_parameter('sw_exit_wsteady_k', 20)   # 흰 인계 종결자 연속 프레임 (0=off)
        self.declare_parameter('sw_exit_gate_frames', 40)  # 탈출 방향게이트 적용 프레임 (이후 해제)
        self.declare_parameter('sw_exit_open_frames', 30)  # 탈출 개방 구간(발화 직후 좌향 코리도 허용). 0=off
        self.declare_parameter('sw_open_max_lean_px', 90.0)  # 개방 구간 |기욺| 상한 (정지선 대각선 기각)
        self.declare_parameter('sw_exit_straight_px', 8.0)  # 직선 판정 |기욺| 문턱(px)
        self.declare_parameter('course', 'in')                      # 'in' 일 때만 색상 추종 활성 (launch 전달)
        # 차선 폭 초기값(px, BEV 워프 기준). 0=학습대기. EMA 학습은 그대로 계속 미세보정.
        # 192 = 실측 차선폭 350mm 를 BEV 캘리로 변환한 값((0.80-0.20)*320px). 단선 구간
        # 콜드스타트 해결용. 카메라 마운트/BEV 재캘리 시 다시 확인.
        self.declare_parameter('lane_width_init', 192.0)

        sub_topic = str(self.get_parameter('subscribe_topic').value)
        self.lane_topic = str(self.get_parameter('lane_topic').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.crossline_debug = bool(self.get_parameter('crossline_debug').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_period = 0.0
        dbg_hz = float(self.get_parameter('debug_hz').value)
        if dbg_hz > 0.0:
            self.debug_period = 1.0 / dbg_hz
        self._last_debug_t = None   # 마지막 디버그 발행 시각(초); rate-limit용

        gp = self.get_parameter
        # race_dir 마스터: 점선 junction 이 어느 쪽에 나타나는지를 여기서 도출한다.
        # 'left'(반시계) -> junction 은 오른쪽; 'right'(시계) -> junction 은 왼쪽.
        race_dir = str(gp('race_dir').value).lower()
        if race_dir == 'right':
            junction_side = 'left'
        elif race_dir == 'left':
            junction_side = 'right'
        else:
            junction_side = str(gp('junction_side').value)
        self.detector = LaneDetector(
            roi_top_ratio=float(gp('roi_top_ratio').value),
            bright_thresh=int(gp('bright_thresh').value),
            min_pixels=int(gp('min_pixels').value),
            junction_side=junction_side,
            junction_dash_transitions=int(gp('junction_dash_transitions').value),
            junction_min_row_pixels=int(gp('junction_min_row_pixels').value),
            junction_gap_rows=int(gp('junction_gap_rows').value),
            mask_mode=str(gp('mask_mode').value),
            use_white=bool(gp('use_white').value),
            use_yellow=bool(gp('use_yellow').value),
            white_hsv_lo=[int(x) for x in gp('white_hsv_lo').value],
            white_hsv_hi=[int(x) for x in gp('white_hsv_hi').value],
            yellow_hsv_lo=[int(x) for x in gp('yellow_hsv_lo').value],
            yellow_hsv_hi=[int(x) for x in gp('yellow_hsv_hi').value],
            use_red=bool(gp('use_red').value),
            red_hsv_lo1=[int(x) for x in gp('red_hsv_lo1').value],
            red_hsv_hi1=[int(x) for x in gp('red_hsv_hi1').value],
            red_hsv_lo2=[int(x) for x in gp('red_hsv_lo2').value],
            red_hsv_hi2=[int(x) for x in gp('red_hsv_hi2').value],
            num_bands=int(gp('num_bands').value),
            morph_kernel=int(gp('morph_kernel').value),
            width_ema=float(gp('width_ema').value),
            smooth_alpha=float(gp('smooth_alpha').value),
            use_birdeye=bool(gp('use_birdeye').value),
            birdeye_src_ratio=[float(x) for x in gp('birdeye_src_ratio').value],
            birdeye_dst_ratio=[float(x) for x in gp('birdeye_dst_ratio').value],
            use_guided_band=bool(gp('use_guided_band').value),
            guide_margin_px=int(gp('guide_margin_px').value),
            guide_margin_growth_px=int(gp('guide_margin_growth_px').value),
            guide_min_pixels=int(gp('guide_min_pixels').value),
            guide_use_previous_frame=bool(gp('guide_use_previous_frame').value),
            guide_max_jump_px=int(gp('guide_max_jump_px').value),
            use_lookahead_control=bool(gp('use_lookahead_control').value),
            near_weight=float(gp('near_weight').value),
            lookahead_weight=float(gp('lookahead_weight').value),
            lookahead_band_index=int(gp('lookahead_band_index').value),
            adaptive_lookahead=bool(gp('adaptive_lookahead').value),
            curve_lookahead_weight=float(gp('curve_lookahead_weight').value),
            curve_lookahead_thresh=float(gp('curve_lookahead_thresh').value),
            crossline_roi_top_ratio=float(gp('crossline_roi_top_ratio').value),
            crossline_roi_bottom_ratio=float(gp('crossline_roi_bottom_ratio').value),
            crossline_min_width_ratio=float(gp('crossline_min_width_ratio').value),
            fork_scan_top_ratio=float(gp('fork_scan_top_ratio').value),
            fork_scan_bottom_ratio=float(gp('fork_scan_bottom_ratio').value),
            fork_col_min_ratio=float(gp('fork_col_min_ratio').value),
            fork_min_groups=int(gp('fork_min_groups').value),
            fork_span_ratio=float(gp('fork_span_ratio').value),
            fork_seed_px=int(gp('fork_seed_px').value),
            follow_yellow=bool(gp('follow_yellow').value),
            follow_yellow_ratio=float(gp('follow_yellow_ratio').value),
            follow_yellow_exit_white_ratio=float(gp('follow_yellow_exit_white_ratio').value),
            course=str(gp('course').value).lower(),
            lane_width_init=float(gp('lane_width_init').value),
        )
        # 생성자 인자에 없는 튜닝들은 속성으로 직접 주입 (라이브 변경도 동일 경로)
        self.detector.follow_yellow_exit_yellow_frac = float(gp('follow_yellow_exit_yellow_frac').value)
        self.detector.follow_yellow_exit_frames = int(gp('follow_yellow_exit_frames').value)
        self.detector.follow_yellow_exit_frames_exit = int(gp('follow_yellow_exit_frames_exit').value)
        self.detector.w_align_frames = int(gp('w_align_frames').value)
        self.detector.w_align_gain = float(gp('w_align_gain').value)
        self.detector.w_align_min_px = int(gp('w_align_min_px').value)
        self.detector.crossline_min_area_px = int(gp('crossline_min_area_px').value)
        self.detector.crossline_perp_tol_deg = float(gp('crossline_perp_tol_deg').value)
        self.detector.crossline_bev_aspect = float(gp('crossline_bev_aspect').value)
        self.detector.lane_heading_alpha = float(gp('lane_heading_alpha').value)
        self.detector.crossline_exclude_px = float(gp('crossline_exclude_px').value)
        self.detector.crossline_hough_thresh = int(gp('crossline_hough_thresh').value)
        self.detector.crossline_hough_max_gap = int(gp('crossline_hough_max_gap').value)
        self.detector.crossline_min_solidity = float(gp('crossline_min_solidity').value)
        self.detector.crossline_sw_gate = int(gp('crossline_sw_gate').value)
        self.detector.sw_curv_max_a = float(gp('sw_curv_max_a').value)
        self.detector.stopline_mode = int(gp('stopline_mode').value)
        self.detector.stopline_ang_max = float(gp('stopline_ang_max').value)
        self.detector.stopline_cov_min = float(gp('stopline_cov_min').value)
        self.detector.stopline_sol_min = float(gp('stopline_sol_min').value)
        self.detector.w_align_dash_fallback = int(gp('w_align_dash_fallback').value)
        self.detector.follow_yellow_blind_release_frames = int(gp('follow_yellow_blind_release_frames').value)
        self.detector.crossline_sw_margin = float(gp('crossline_sw_margin').value)
        self.detector.crossline_debug_all = bool(gp('crossline_debug_all').value)
        self.detector.filter_yellow_dashes = bool(gp('filter_yellow_dashes').value)
        self.detector.yellow_solid_min_h_ratio = float(gp('yellow_solid_min_h_ratio').value)
        self.detector.yellow_dash_fallback_px = int(gp('yellow_dash_fallback_px').value)
        self.detector.dash_fallback_exit_frames = int(gp('dash_fallback_exit_frames').value)
        self.detector.yellow_heading_gain = float(gp('yellow_heading_gain').value)
        self.detector.center_jump_max_ratio = float(gp('center_jump_max_ratio').value)
        self.detector.entry_oneline_frames = int(gp('entry_oneline_frames').value)
        self.detector.oneline_near_bands = int(gp('oneline_near_bands').value)
        self.detector.ra_entry_oneline_frames = int(gp('ra_entry_oneline_frames').value)
        self.detector.ra_exit_oneline_frames = int(gp('ra_exit_oneline_frames').value)
        self.detector.sw_entry_frames = int(gp('sw_entry_frames').value)
        self.detector.sw_exit_frames = int(gp('sw_exit_frames').value)
        self.detector.sw_entry_input = str(gp('sw_entry_input').value)
        self.detector.sw_exit_input = str(gp('sw_exit_input').value)
        self.detector.sw_num_boxes = int(gp('sw_num_boxes').value)
        self.detector.sw_box_margin = int(gp('sw_box_margin').value)
        self.detector.sw_max_shift = int(gp('sw_max_shift').value)
        self.detector.sw_min_box_px = int(gp('sw_min_box_px').value)
        self.detector.sw_min_boxes = int(gp('sw_min_boxes').value)
        self.detector.sw_min_pixels = int(gp('sw_min_pixels').value)
        self.detector.sw_wrongdir_px = float(gp('sw_wrongdir_px').value)
        self.detector.sw_max_resid_px = float(gp('sw_max_resid_px').value)
        self.detector.sw_peak_min_px = int(gp('sw_peak_min_px').value)
        self.detector.sw_max_peaks = int(gp('sw_max_peaks').value)
        self.detector.sw_cross_row_frac = float(gp('sw_cross_row_frac').value)
        self.detector.sw_side_default = int(gp('sw_side_default').value)
        self.detector.sw_exit_straight_k = int(gp('sw_exit_straight_k').value)
        self.detector.sw_exit_wsteady_k = int(gp('sw_exit_wsteady_k').value)
        self.detector.sw_exit_gate_frames = int(gp('sw_exit_gate_frames').value)
        self.detector.sw_exit_open_frames = int(gp('sw_exit_open_frames').value)
        self.detector.sw_open_max_lean_px = float(gp('sw_open_max_lean_px').value)
        self.detector.sw_exit_straight_px = float(gp('sw_exit_straight_px').value)

        self.pub = self.create_publisher(LaneState, self.lane_topic, 10)
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(CompressedImage, self.debug_topic, 10)
        # 임시 진단 토픽: 정지선 후보의 (부호 있는 slope, 각도, 축길이, 인라이어, 판정).
        # 부호/길이 임계를 실측으로 정하기 위한 것. 확정되면 crossline_debug=False.
        self.crossline_dbg_pub = (
            self.create_publisher(String, '/perception/lane/crossline_dbg', 10)
            if self.crossline_debug else None)

        self.create_subscription(CompressedImage, sub_topic, self.on_image, 10)
        # decision 이 표결로 확정한 갈림길 방향('left'/'right'/'')을 받아 브랜치 선택에 쓴다.
        self.create_subscription(String, str(gp('fork_topic').value), self.on_fork_dir, 10)
        # decision 주행 모드(상태머신 상태)를 받아 BEV 디버그 화면에 표시한다.
        self.create_subscription(String, str(gp('state_topic').value), self.on_state, 10)
        # 합류부 창 활성 플래그 (decision 의 yaw 위치창) -> "단선=우측" 규칙 스코프
        self.create_subscription(Bool, str(gp('merge_zone_topic').value), self.on_merge_zone, 10)
        # 라이브 튜닝: `ros2 param set /lane_node yellow_hsv_lo "[18,45,110]"` 등을 실행하면
        # detector 가 즉시 갱신되어 재시작 없이 반영된다. 튜닝하면서 대시보드로
        # HSV 마스크를 바로 확인할 수 있다.
        self.add_on_set_parameters_callback(self._on_set_params)
        self.get_logger().info(f'lane_node up. in={sub_topic} out={self.lane_topic}')

    def _on_set_params(self, params):
        hsv_keys = {'white_hsv_lo', 'white_hsv_hi', 'yellow_hsv_lo', 'yellow_hsv_hi',
                    'red_hsv_lo1', 'red_hsv_hi1', 'red_hsv_lo2', 'red_hsv_hi2'}
        for p in params:
            n, v = p.name, p.value
            if n == 'debug_hz':
                self.debug_period = (1.0 / float(v)) if float(v) > 0.0 else 0.0
                self.get_logger().info(f'param live-updated: debug_hz = {v}')
                continue
            if n == 'lane_width_init':   # detector 속성명은 _lane_width
                self.detector._lane_width = float(v)
                self.get_logger().info(f'param live-updated: lane_width_init = {v}')
                continue
            if n in hsv_keys:
                setattr(self.detector, n, tuple(int(x) for x in v))
            elif n in ('birdeye_src_ratio', 'birdeye_dst_ratio'):
                setattr(self.detector, n, [float(x) for x in v])
                self.detector.invalidate_birdeye_cache()   # 변환 행렬을 다시 만들어야 함
            elif n in ('use_white', 'use_yellow', 'use_birdeye', 'use_guided_band',
                       'guide_use_previous_frame', 'use_lookahead_control',
                       'adaptive_lookahead', 'follow_yellow'):
                setattr(self.detector, n, bool(v))
            elif hasattr(self.detector, n):
                setattr(self.detector, n, v)
            self.get_logger().info(f'param live-updated: {n} = {v}')
        return SetParametersResult(successful=True)

    def on_fork_dir(self, msg: String):
        d = msg.data.strip().lower()
        self.detector.fork_dir = d if d in ('left', 'right') else None

    def on_state(self, msg: String):
        self.detector.drive_mode = str(msg.data)

    def on_merge_zone(self, msg: Bool):
        self.detector.merge_zone = bool(msg.data)

    def on_image(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 디버그 이미지를 이번 프레임에 만들지 결정(rate-limit). 만들지 않으면 detector가
        # 그리기 자체를 건너뛰어 인코딩+그리기 부하를 모두 던다.
        want_debug = False
        if self.debug_pub is not None:
            now = self.get_clock().now().nanoseconds / 1e9
            if (self._last_debug_t is None
                    or (now - self._last_debug_t) >= self.debug_period):
                want_debug = True
                self._last_debug_t = now

        (lane_found, offset, num_lanes, junction,
         yellow_ratio, yellow_offset, curvature,
         yellow_crossline, fork, red_ratio, debug) = self.detector.process(
            frame, draw_debug=want_debug)

        out = LaneState()
        out.header = msg.header
        out.lane_found = bool(lane_found)
        out.offset = float(offset)
        out.curvature = float(curvature)
        out.num_lanes = int(num_lanes)
        out.junction = bool(junction)
        out.yellow_ratio = float(yellow_ratio)
        out.yellow_offset = float(yellow_offset)
        out.yellow_crossline = bool(yellow_crossline)
        out.fork = bool(fork)
        out.red_ratio = float(red_ratio)
        self.pub.publish(out)

        if self.crossline_dbg_pub is not None:
            cands = getattr(self.detector, 'last_crossline_cands', [])
            if cands:
                self.crossline_dbg_pub.publish(String(data=json.dumps({
                    'xl': bool(yellow_crossline),
                    'yr': round(float(yellow_ratio), 4),
                    'cv': round(float(curvature), 3),
                    'c': cands,
                })))

        if want_debug and debug is not None:
            ok, enc = cv2.imencode('.jpg', debug, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                dbg = CompressedImage()
                dbg.header = msg.header
                dbg.format = 'jpeg'
                dbg.data = enc.tobytes()
                self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
