"""lane_node: OpenCV lane following.

Subscribes to the camera image, runs LaneDetector, and publishes a
LaneState (normalised offset). Optionally republishes a debug image so
you can watch the detection in the kit's monitor dashboard.
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

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
        self.declare_parameter('roi_top_ratio', 0.55)
        self.declare_parameter('bright_thresh', 160)
        self.declare_parameter('min_pixels', 40)
        # --- robust lane following (multi-band look-ahead, curvature, smoothing) ---
        self.declare_parameter('num_bands', 4)        # look-ahead bands for curves
        self.declare_parameter('morph_kernel', 3)     # noise cleanup (glare/dashes); 0=off
        self.declare_parameter('width_ema', 0.1)      # lane-width memory adapt rate
        self.declare_parameter('smooth_alpha', 0.5)   # offset smoothing (lower=smoother)
        self.declare_parameter('race_dir', 'left')                 # 'left'/'right' master; derives junction_side
        self.declare_parameter('junction_side', 'right')           # used only if race_dir is not left/right
        self.declare_parameter('junction_dash_transitions', 3)     # vertical on/off changes => dashed
        self.declare_parameter('junction_min_row_pixels', 2)       # row counts as 'line' above this
        self.declare_parameter('junction_gap_rows', 2)             # full-opening fallback
        # --- colour masking (white + yellow lanes) ---
        self.declare_parameter('mask_mode', 'hsv')            # 'hsv' (white|yellow) or 'gray'
        self.declare_parameter('use_white', True)
        self.declare_parameter('use_yellow', True)            # yellow roundabout lane
        self.declare_parameter('white_hsv_lo', [0, 0, 180])   # H,S,V lower (OpenCV H 0-179)
        self.declare_parameter('white_hsv_hi', [179, 60, 255])
        self.declare_parameter('yellow_hsv_lo', [18, 45, 110])   # S floor lowered for pale/faded yellow lines
        self.declare_parameter('yellow_hsv_hi', [40, 255, 255])
        # --- bird-eye view (default ON; src/dst below are GENERIC guesses -> tune on track) ---
        # flat ratio lists [x1,y1, x2,y2, x3,y3, x4,y4] (TL,TR,BR,BL), 0..1 of ROI size
        self.declare_parameter('use_birdeye', True)
        self.declare_parameter('birdeye_src_ratio', [0.25, 0.05, 0.75, 0.05, 0.95, 0.95, 0.05, 0.95])
        self.declare_parameter('birdeye_dst_ratio', [0.20, 0.00, 0.80, 0.00, 0.80, 1.00, 0.20, 1.00])
        # --- guided band search (default ON) ---
        self.declare_parameter('use_guided_band', True)
        self.declare_parameter('guide_margin_px', 60)         # search ± px around previous band centre
        self.declare_parameter('guide_margin_growth_px', 10)  # margin += i*growth toward far bands
        self.declare_parameter('guide_min_pixels', 20)        # narrow window -> lower than min_pixels
        self.declare_parameter('guide_use_previous_frame', True)
        self.declare_parameter('guide_max_jump_px', 80)       # clamp band-to-band centre jumps
        # --- look-ahead steering blend (partial pure-pursuit; default ON) ---
        self.declare_parameter('use_lookahead_control', True)
        self.declare_parameter('near_weight', 0.7)            # nearest-band weight
        self.declare_parameter('lookahead_weight', 0.3)       # far-band weight
        self.declare_parameter('lookahead_band_index', -1)    # -1 = farthest detected band
        self.declare_parameter('adaptive_lookahead', False)   # boost lookahead on sharp curves
        self.declare_parameter('curve_lookahead_weight', 0.4)
        self.declare_parameter('curve_lookahead_thresh', 0.25)

        sub_topic = str(self.get_parameter('subscribe_topic').value)
        self.lane_topic = str(self.get_parameter('lane_topic').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        gp = self.get_parameter
        # race_dir master: derive which side the dashed junction shows on.
        # 'left' (CCW) -> junction on the right; 'right' (CW) -> junction on the left.
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
        )

        self.pub = self.create_publisher(LaneState, self.lane_topic, 10)
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(CompressedImage, self.debug_topic, 10)

        self.create_subscription(CompressedImage, sub_topic, self.on_image, 10)
        # LIVE tuning: `ros2 param set /lane_node yellow_hsv_lo "[18,45,110]"` etc. applies
        # immediately by updating the detector (no restart), so you can watch the HSV mask
        # on the dashboard while tuning.
        self.add_on_set_parameters_callback(self._on_set_params)
        self.get_logger().info(f'lane_node up. in={sub_topic} out={self.lane_topic}')

    def _on_set_params(self, params):
        hsv_keys = {'white_hsv_lo', 'white_hsv_hi', 'yellow_hsv_lo', 'yellow_hsv_hi'}
        for p in params:
            n, v = p.name, p.value
            if n in hsv_keys:
                setattr(self.detector, n, tuple(int(x) for x in v))
            elif n in ('birdeye_src_ratio', 'birdeye_dst_ratio'):
                setattr(self.detector, n, [float(x) for x in v])
                self.detector.invalidate_birdeye_cache()   # matrix must be rebuilt
            elif n in ('use_white', 'use_yellow', 'use_birdeye', 'use_guided_band',
                       'guide_use_previous_frame', 'use_lookahead_control',
                       'adaptive_lookahead'):
                setattr(self.detector, n, bool(v))
            elif hasattr(self.detector, n):
                setattr(self.detector, n, v)
            self.get_logger().info(f'param live-updated: {n} = {v}')
        return SetParametersResult(successful=True)

    def on_image(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        (lane_found, offset, num_lanes, junction,
         yellow_ratio, yellow_offset, curvature, debug) = self.detector.process(frame)

        out = LaneState()
        out.header = msg.header
        out.lane_found = bool(lane_found)
        out.offset = float(offset)
        out.curvature = float(curvature)
        out.num_lanes = int(num_lanes)
        out.junction = bool(junction)
        out.yellow_ratio = float(yellow_ratio)
        out.yellow_offset = float(yellow_offset)
        self.pub.publish(out)

        if self.debug_pub is not None:
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
