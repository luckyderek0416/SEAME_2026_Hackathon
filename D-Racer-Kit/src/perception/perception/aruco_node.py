"""aruco_node: OpenCV ArUco marker detection (the dynamic obstacle).

Pure OpenCV, no deep learning. Reports whether a marker is currently in
view and how big it looks (proximity proxy). The stop/go *decision* and
debounce live in decision_node, not here -- this node only reports facts.
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage

from perception_msgs.msg import ArucoState


class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('aruco_topic', '/perception/aruco')
        self.declare_parameter('dictionary', 'DICT_4X4_50')   # MUST match the printed marker
        self.declare_parameter('inverted', False)             # True if marker is white-on-black
        # detectMarkers 는 매 프레임 돌릴 필요가 없다(마커는 여러 프레임에 걸쳐 접근함).
        # 카메라 30Hz 전부 대신 이 주기로만 검출해 보드 CPU 부하를 던다(YOLO 노드와 동일 패턴).
        # 0 이하 = 카메라 프레임마다(구버전 동작). 라이브: ros2 param set /aruco_node aruco_hz 15.0
        self.declare_parameter('aruco_hz', 12.0)

        sub_topic = str(self.get_parameter('subscribe_topic').value)
        self.aruco_topic = str(self.get_parameter('aruco_topic').value)
        dict_name = str(self.get_parameter('dictionary').value)
        inverted = bool(self.get_parameter('inverted').value)

        dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        # OpenCV >= 4.7 uses ArucoDetector; older versions use detectMarkers(...)
        try:
            params = cv2.aruco.DetectorParameters()
            if inverted and hasattr(params, 'detectInvertedMarker'):
                params.detectInvertedMarker = True
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
            self.use_new_api = True
        except AttributeError:
            self.params = cv2.aruco.DetectorParameters_create()
            if inverted and hasattr(self.params, 'detectInvertedMarker'):
                self.params.detectInvertedMarker = True
            self.use_new_api = False
        self.get_logger().info(f'aruco dictionary={dict_name} inverted={inverted}')

        self.pub = self.create_publisher(ArucoState, self.aruco_topic, 10)
        # on_image 은 최신 프레임만 저장하고(가벼움), 무거운 detectMarkers 는 타이머가
        # aruco_hz 로만 돌린다. 접근하는 마커는 여러 프레임에 걸쳐 커지므로 저속이어도 안 놓친다.
        self.latest = None
        self.create_subscription(CompressedImage, sub_topic, self.on_image, 10)
        self._timer = None
        self._make_timer(float(self.get_parameter('aruco_hz').value))
        self.add_on_set_parameters_callback(self._on_set_params)
        self.get_logger().info(f'aruco_node up. in={sub_topic} out={self.aruco_topic}')

    def _make_timer(self, aruco_hz):
        """(재)생성: aruco_hz>0 이면 그 주기 타이머, 아니면 매 프레임 검출로 폴백."""
        if self._timer is not None:
            self.destroy_timer(self._timer)
            self._timer = None
        if aruco_hz > 0.0:
            self._timer = self.create_timer(1.0 / aruco_hz, self.on_timer)

    def _on_set_params(self, params):
        for p in params:
            if p.name == 'aruco_hz':
                try:
                    hz = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason='aruco_hz 는 숫자여야 합니다')
                self._make_timer(hz)
                self.get_logger().info(f'[live] aruco_hz -> {hz:g}')
        return SetParametersResult(successful=True)

    def on_image(self, msg: CompressedImage):
        # 타이머 모드: 압축 바이트만 저장하고 디코드+검출은 on_timer 에 맡긴다
        # (JPEG 디코드도 30Hz -> aruco_hz 로 줄어 부하를 더 던다).
        if self._timer is not None:
            self.latest = (msg.data, msg.header)
            return
        self._detect_and_publish(msg.data, msg.header)   # aruco_hz<=0 폴백: 매 프레임

    def on_timer(self):
        if self.latest is None:
            return
        data, header = self.latest
        self._detect_and_publish(data, header)

    def _detect_and_publish(self, data, header):
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        out = ArucoState()
        out.header = header
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
