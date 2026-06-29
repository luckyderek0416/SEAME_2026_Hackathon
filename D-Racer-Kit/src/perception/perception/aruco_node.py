"""aruco_node: OpenCV ArUco marker detection (the dynamic obstacle).

Pure OpenCV, no deep learning. Reports whether a marker is currently in
view and how big it looks (proximity proxy). The stop/go *decision* and
debounce live in decision_node, not here -- this node only reports facts.
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from perception_msgs.msg import ArucoState


class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('aruco_topic', '/perception/aruco')
        self.declare_parameter('dictionary', 'DICT_4X4_50')

        sub_topic = str(self.get_parameter('subscribe_topic').value)
        self.aruco_topic = str(self.get_parameter('aruco_topic').value)
        dict_name = str(self.get_parameter('dictionary').value)

        dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        # OpenCV >= 4.7 uses ArucoDetector; older versions use detectMarkers(...)
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
            self.use_new_api = True
        except AttributeError:
            self.params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        self.pub = self.create_publisher(ArucoState, self.aruco_topic, 10)
        self.create_subscription(CompressedImage, sub_topic, self.on_image, 10)
        self.get_logger().info(f'aruco_node up. in={sub_topic} out={self.aruco_topic}')

    def on_image(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        out = ArucoState()
        out.header = msg.header
        if ids is not None and len(ids) > 0:
            areas = [cv2.contourArea(c.reshape(-1, 1, 2).astype(np.float32)) for c in corners]
            idx = int(np.argmax(areas))          # nearest marker = largest on screen
            out.detected = True
            out.marker_id = int(ids[idx][0])
            out.area_ratio = float(areas[idx] / (w * h))
        else:
            out.detected = False
            out.marker_id = -1
            out.area_ratio = 0.0
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
