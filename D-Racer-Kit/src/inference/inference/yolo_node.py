"""yolo_node: the ONLY deep-learning model in the whole stack.

It detects the 4 competition objects (red_light, green_light, left_sign,
right_sign) and publishes their class + screen position. It does NOT drive
the car -- lane following is OpenCV in lane_node. decision_node turns these
detections into actions (start on green, branch on a sign, stop on red).

YOLO inference is run on a TIMER (infer_hz), not on every camera frame, so
it can run slower than lane following without blocking anything.
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from inference_msgs.msg import Detection, Detections


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('detections_topic', '/inference/detections')
        self.declare_parameter('model_path', '/home/topst/D-Racer/models/best.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('infer_hz', 10.0)
        self.declare_parameter('imgsz', 320)

        sub_topic = str(self.get_parameter('subscribe_topic').value)
        self.det_topic = str(self.get_parameter('detections_topic').value)
        self.model_path = str(self.get_parameter('model_path').value)
        self.conf = float(self.get_parameter('conf_threshold').value)
        infer_hz = float(self.get_parameter('infer_hz').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.model = self._load_model()
        self.latest = None  # (frame, header) of the most recent camera image

        self.pub = self.create_publisher(Detections, self.det_topic, 10)
        self.create_subscription(CompressedImage, sub_topic, self.on_image, 10)
        self.timer = self.create_timer(1.0 / max(infer_hz, 1.0), self.on_timer)
        self.get_logger().info(f'yolo_node up. model={self.model_path} infer_hz={infer_hz}')

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            self.get_logger().info('YOLO model loaded.')
            return model
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'Could not load YOLO model ({exc}). '
                'Publishing EMPTY detections until model_path points to a trained .pt file.'
            )
            return None

    def on_image(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self.latest = (frame, msg.header)

    def on_timer(self):
        if self.latest is None:
            return
        frame, header = self.latest

        out = Detections()
        out.header = header

        if self.model is not None:
            h, w = frame.shape[:2]
            results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
            names = self.model.names  # {class_id: 'name'}
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    d = Detection()
                    d.label = str(names[cls_id]) if cls_id in names else str(cls_id)
                    d.class_id = cls_id
                    d.confidence = float(b.conf[0])
                    d.x_center = float((x1 + x2) / 2.0 / w)
                    d.y_center = float((y1 + y2) / 2.0 / h)
                    d.width = float((x2 - x1) / w)
                    d.height = float((y2 - y1) / h)
                    out.detections.append(d)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
