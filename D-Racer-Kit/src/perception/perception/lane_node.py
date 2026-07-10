"""lane_node: OpenCV 기반 차선 추종.

카메라 이미지를 구독해 LaneDetector 를 돌리고, LaneState(정규화된 offset)를
publish 한다. 옵션으로 디버그 이미지를 다시 publish 해서 키트의 모니터
대시보드에서 검출 과정을 지켜볼 수 있다.
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from rcl_interfaces.msg import SetParametersResult

from perception_msgs.msg import LaneState
from perception.lane_detector import LaneDetector


class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('debug_topic', '/perception/lane/debug')
        self.declare_parameter('jpeg_quality', 80)
        # debug_hz: 디버그 이미지 그리기+JPEG 인코딩을 이 주기로 제한한다. 주행 로직은
        # 카메라 풀레이트(20Hz) 그대로 돌지만, 대시보드는 ~7Hz로만 프레임을 가져가므로
        # 20Hz 인코딩은 대부분 버려진다. 저주기면 보드 CPU를 아끼면서 대시보드 체감은 동일.
        # (0 이하 = 매 프레임; 라이브로 ros2 param set /lane_node debug_hz 5.0 조절 가능)
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
        # 평탄화한 비율 리스트 [x1,y1, x2,y2, x3,y3, x4,y4] (TL,TR,BR,BL), ROI 크기 대비 0..1
        # 카메라 높이/각도를 다시 바꾸면 재캘리브레이션 필요.
        # 2026-07-08: 카메라 높이 28cm 재마운트 후 실측 프레임 재캘리브레이션.
        # 직선 구간에서 좌/우 흰선을 직선 피팅해 y=0.342h/0.965h 에서의 차선 x 를
        # src 꼭짓점으로 사용 (차선이 워프 후 정확히 0.20w/0.80w 에 오도록 구성).
        # 검증: 워프 후 slope L-0.021/R+0.002 (거의 수직), 차선폭 top 191/bottom 193px
        # (목표 192px = 실차선 350mm). 이전(07-07, 낮은 마운트) 값:
        # [0.226, 0.342, 0.745, 0.342, 0.998, 0.965, -0.006, 0.965]
        self.declare_parameter('use_birdeye', True)
        self.declare_parameter('birdeye_src_ratio', [0.288, 0.342, 0.713, 0.342, 0.857, 0.965, 0.150, 0.965])
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
        self.declare_parameter('crossline_min_rows', 4)             # (미사용; 하위호환)
        self.declare_parameter('crossline_max_angle_deg', 50.0)     # 수평 기준 허용 기울기 (대각선 정지선)
        self.declare_parameter('crossline_min_area_px', 60)         # 성분 최소 픽셀 (dash/노이즈 필터)
        self.declare_parameter('crossline_max_resid_px', 2.5)       # 직선성: 피팅 잔차 허용 px
        self.declare_parameter('crossline_min_inlier_frac', 0.75)   # 직선성: 인라이어 컬럼 비율
        # --- 좌/우 갈림길 (fork) 감지 + 브랜치 선택 ---
        self.declare_parameter('fork_topic', '/decision/fork_dir')  # decision 이 확정 방향 publish
        self.declare_parameter('state_topic', '/decision/state')    # decision 주행 모드 -> BEV 표시
        self.declare_parameter('fork_scan_top_ratio', 0.0)          # 분기 스캔밴드 상단(BEV far)
        self.declare_parameter('fork_scan_bottom_ratio', 0.5)       # 분기 스캔밴드 하단
        self.declare_parameter('fork_col_min_ratio', 0.15)          # 컬럼=라인 판정 세로픽셀 비율
        self.declare_parameter('fork_min_groups', 3)                # 라인 군집 개수 이상 => 분기
        self.declare_parameter('fork_span_ratio', 0.65)             # 바깥라인 간격 폭 이상 => 분기
        self.declare_parameter('fork_seed_px', 90)                  # 브랜치 선택 시드 이동량(px)
        # --- 노란색 우선 추종 (In 코스: 노란 진입 커브/회전교차로 링) ---
        self.declare_parameter('follow_yellow', True)               # In 코스 색상 추종 상태머신 on/off
        self.declare_parameter('follow_yellow_ratio', 0.03)         # 이 노란비율 이상 -> YELLOW 모드 진입
                                                                    # (노란선이 점선이라 yr 이 낮음 -> 0.03)
        self.declare_parameter('follow_yellow_exit_white_ratio', 1.0)  # 흰픽셀 > 이 배율*노란픽셀 (해제 조건 일부)
        self.declare_parameter('follow_yellow_exit_yellow_frac', 0.5)  # 해제 노랑문턱 = 진입문턱*이 비율 (노랑 보이면 유지)
        self.declare_parameter('follow_yellow_exit_frames', 10)        # 해제 조건 연속 프레임 (플리커 방지)
        self.declare_parameter('filter_yellow_dashes', True)           # Y추종 중 점선/정지선 track 제외 (실선만)
        self.declare_parameter('yellow_solid_min_h_ratio', 0.30)       # "실선" 판정 최소 세로 비율
        self.declare_parameter('yellow_dash_fallback_px', 120)         # 실선 픽셀 이 미만이면 점선 포함 폴백
        self.declare_parameter('dash_fallback_exit_frames', 30)        # 폴백 해제(실선 복귀)에 필요한 연속 프레임 (1s)
        self.declare_parameter('yellow_heading_gain', 0.4)             # 헤어핀 주축 기울기 헤딩 보정 게인 (0=off)
        self.declare_parameter('course', 'in')                      # 'in' 일 때만 색상 추종 활성 (launch 전달)
        # 차선 폭 초기값(px, BEV 워프 기준). 0=학습대기. EMA 학습은 그대로 계속 미세보정.
        # 192 = 실측 차선폭 350mm 를 BEV 캘리로 변환한 값((0.80-0.20)*320px). 단선 구간
        # 콜드스타트 해결용. 카메라 마운트/BEV 재캘리 시 다시 확인.
        self.declare_parameter('lane_width_init', 192.0)

        sub_topic = str(self.get_parameter('subscribe_topic').value)
        self.lane_topic = str(self.get_parameter('lane_topic').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
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
            crossline_min_rows=int(gp('crossline_min_rows').value),
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
        self.detector.crossline_max_angle_deg = float(gp('crossline_max_angle_deg').value)
        self.detector.crossline_min_area_px = int(gp('crossline_min_area_px').value)
        self.detector.crossline_max_resid_px = float(gp('crossline_max_resid_px').value)
        self.detector.crossline_min_inlier_frac = float(gp('crossline_min_inlier_frac').value)
        self.detector.filter_yellow_dashes = bool(gp('filter_yellow_dashes').value)
        self.detector.yellow_solid_min_h_ratio = float(gp('yellow_solid_min_h_ratio').value)
        self.detector.yellow_dash_fallback_px = int(gp('yellow_dash_fallback_px').value)
        self.detector.dash_fallback_exit_frames = int(gp('dash_fallback_exit_frames').value)
        self.detector.yellow_heading_gain = float(gp('yellow_heading_gain').value)

        self.pub = self.create_publisher(LaneState, self.lane_topic, 10)
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(CompressedImage, self.debug_topic, 10)

        self.create_subscription(CompressedImage, sub_topic, self.on_image, 10)
        # decision 이 표결로 확정한 갈림길 방향('left'/'right'/'')을 받아 브랜치 선택에 쓴다.
        self.create_subscription(String, str(gp('fork_topic').value), self.on_fork_dir, 10)
        # decision 주행 모드(상태머신 상태)를 받아 BEV 디버그 화면에 표시한다.
        self.create_subscription(String, str(gp('state_topic').value), self.on_state, 10)
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
