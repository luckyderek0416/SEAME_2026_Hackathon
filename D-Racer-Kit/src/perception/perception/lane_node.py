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

        sub_topic = str(self.get_parameter('subscribe_topic').value)
        self.lane_topic = str(self.get_parameter('lane_topic').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.detector = LaneDetector(
            roi_top_ratio=float(self.get_parameter('roi_top_ratio').value),
            bright_thresh=int(self.get_parameter('bright_thresh').value),
            min_pixels=int(self.get_parameter('min_pixels').value),
        )

        self.pub = self.create_publisher(LaneState, self.lane_topic, 10)
        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(CompressedImage, self.debug_topic, 10)

        self.create_subscription(CompressedImage, sub_topic, self.on_image, 10)
        self.get_logger().info(f'lane_node up. in={sub_topic} out={self.lane_topic}')

    def on_image(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        lane_found, offset, num_lanes, debug = self.detector.process(frame)

        out = LaneState()
        out.header = msg.header
        out.lane_found = bool(lane_found)
        out.offset = float(offset)
        out.curvature = 0.0
        out.num_lanes = int(num_lanes)
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
