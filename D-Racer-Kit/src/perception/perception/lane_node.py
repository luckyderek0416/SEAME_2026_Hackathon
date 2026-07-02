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
            elif n in ('use_white', 'use_yellow'):
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
