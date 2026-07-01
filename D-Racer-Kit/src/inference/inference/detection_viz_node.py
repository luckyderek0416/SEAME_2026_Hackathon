"""detection_viz_node: draw YOLO detection boxes on the camera image.

Subscribes the camera image + /inference/detections, overlays each detection
as a coloured box + label + confidence, and republishes a CompressedImage you
can watch (rqt_image_view / monitor). Makes it obvious whether the traffic
light / signs are being detected, at what distance, and how confidently -
without staring at `ros2 topic echo`.

Detections carry NORMALISED centre + size (x_center, y_center, width, height in
0..1), matching inference_msgs/Detection, so the box is scaled back to pixels.
"""
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from inference_msgs.msg import Detections

# per-label colour (BGR)
COLORS = {
    'red_light': (0, 0, 255),
    'green_light': (0, 200, 0),
    'left_sign': (255, 128, 0),
    'right_sign': (255, 0, 128),
}


class DetectionVizNode(Node):
    def __init__(self):
        super().__init__('detection_viz_node')
        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('detections_topic', '/inference/detections')
        self.declare_parameter('output_topic', '/inference/viz/compressed')
        self.declare_parameter('jpeg_quality', 80)

        self.out_topic = str(self.get_parameter('output_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        self.latest_dets = []
        self.pub = self.create_publisher(CompressedImage, self.out_topic, 10)
        self.create_subscription(CompressedImage, str(self.get_parameter('image_topic').value),
                                 self.on_image, 10)
        self.create_subscription(Detections, str(self.get_parameter('detections_topic').value),
                                 self.on_dets, 10)
        self.get_logger().info(f'detection_viz up. out={self.out_topic}')

    def on_dets(self, msg):
        self.latest_dets = list(msg.detections)

    def on_image(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        h, w = frame.shape[:2]

        for d in self.latest_dets:
            color = COLORS.get(d.label, (200, 200, 200))
            x1 = int((d.x_center - d.width / 2.0) * w)
            y1 = int((d.y_center - d.height / 2.0) * h)
            x2 = int((d.x_center + d.width / 2.0) * w)
            y2 = int((d.y_center + d.height / 2.0) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{d.label} {d.confidence:.2f}'
            cv2.rectangle(frame, (x1, max(0, y1 - 14)), (x1 + 9 * len(label), y1), color, -1)
            cv2.putText(frame, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # count banner (top-left) so you can see detections even if boxes are tiny
        cv2.putText(frame, f'det: {len(self.latest_dets)}', (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            out = CompressedImage()
            out.header = msg.header
            out.format = 'jpeg'
            out.data = enc.tobytes()
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
