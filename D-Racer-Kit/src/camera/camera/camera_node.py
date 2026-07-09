import os
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
import yaml


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # ROS 파라미터
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('publish_topic', 'camera/image/compressed')
        self.declare_parameter('publish_hz', 30.0)
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('usb_camera_device', '/dev/video1')
        self.declare_parameter('mipi_camera_device', '/dev/video0')
        self.declare_parameter('flip_method', 'rotate-180')
        self.declare_parameter('jpeg_quality', 90)
        # debug_log: 프레임마다 'Published frame' 로그를 찍는다. 30Hz로 SSH에 텍스트를
        # 쏟아내는 I/O 부하가 커서 기본 OFF. 진단 시에만 -p debug_log:=true 로 켠다.
        self.declare_parameter('debug_log', False)

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        publish_topic = str(self.get_parameter('publish_topic').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')
        default_camera_device = str(self.get_parameter('camera_device').value)
        usb_camera_device = str(self.get_parameter('usb_camera_device').value)
        mipi_camera_device = str(self.get_parameter('mipi_camera_device').value)
        flip_method = str(self.get_parameter('flip_method').value)
        jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.publish_hz = publish_hz
        self.jpeg_quality = jpeg_quality

        self.image_width, self.image_height = self.load_image_size()
        self.usb_cam_enabled, self.mipi_cam_enabled = self.load_camera_source_flags()
        usb_camera_device, mipi_camera_device = self.load_camera_device_overrides(
            usb_camera_device,
            mipi_camera_device,
        )
        if self.usb_cam_enabled:
            self.camera_source = 'usb'
            camera_device = usb_camera_device or default_camera_device
        else:
            self.camera_source = 'mipi'
            camera_device = mipi_camera_device or default_camera_device

        self.camera_device = camera_device
        self.flip_method = flip_method

        # web_video_server 와 monitor 구독자들과 호환되는 QoS.
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher_ = self.create_publisher(CompressedImage, publish_topic, self.image_qos)
        self.cap = None
        self.pipeline = None
        if not self.open_capture():
            raise RuntimeError(
                'Failed to open camera with GStreamer pipeline '
                f'(source={self.camera_source}, device={camera_device}, '
                f'width={self.image_width}, height={self.image_height})'
            )

        self.timer = self.create_timer(1.0 / self.publish_hz, self.timer_callback)
        self.get_logger().info('\n'
            f'[Camera Node] : topic={publish_topic} \n'
            f'[camera source] : {self.camera_source} \n'
            f'[width] : {self.image_width}, [height] : {self.image_height} \n'
            f'[camera_device] : {camera_device} \n'
            f'[flip_method] : {flip_method} \n'
            f'[jpeg_quality] : {self.jpeg_quality} \n'
            f'[vehicle_config_file] : {self.vehicle_config_file} \n'
            f'[debug_log] : {self.debug_log} \n'
        )

    def load_image_size(self):
        default_size = (640, 480)
        if not os.path.exists(self.vehicle_config_file):
            return default_size

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_size

        image_width = int(config_data.get('IMAGE_WIDTH', default_size[0]))
        image_height = int(config_data.get('IMAGE_HEIGHT', default_size[1]))
        return image_width, image_height

    def load_camera_source_flags(self):
        # 하위 호환 기본값: MIPI 활성화.
        default_usb_cam = False
        default_mipi_cam = True

        if not os.path.exists(self.vehicle_config_file):
            return default_usb_cam, default_mipi_cam

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_usb_cam, default_mipi_cam

        usb_cam = bool(config_data.get('USB_CAM', default_usb_cam))
        mipi_cam = bool(config_data.get('MIPI_CAM', default_mipi_cam))

        if usb_cam and mipi_cam:
            raise ValueError('Only one of USB_CAM or MIPI_CAM can be true.')
        if not usb_cam and not mipi_cam:
            raise ValueError('One of USB_CAM or MIPI_CAM must be true.')

        return usb_cam, mipi_cam

    def build_candidate_pipelines(self, camera_device, flip_method):
        if self.usb_cam_enabled:
            # 많은 USB 웹캠이 기본적으로 MJPG 포맷을 제공한다.
            mjpg_pipeline = (
                f"v4l2src device={camera_device} io-mode=2 ! "
                "image/jpeg,framerate=30/1 ! jpegdec ! "
                "videoconvert ! videoscale ! "
                f"video/x-raw,format=BGR,width={self.image_width},height={self.image_height},framerate=30/1 ! "
                "appsink sync=false drop=true max-buffers=1"
            )
            # raw 모드 USB 카메라를 위한 폴백.
            raw_pipeline = (
                f"v4l2src device={camera_device} io-mode=2 ! "
                "videoconvert ! videoscale ! "
                f"video/x-raw,format=BGR,width={self.image_width},height={self.image_height},framerate=30/1 ! "
                "appsink sync=false drop=true max-buffers=1"
            )
            return [mjpg_pipeline, raw_pipeline]

        mipi_pipeline = (
            f"v4l2src device={camera_device} io-mode=2 ! "
            f"video/x-raw,format=NV12,width={self.image_width},height={self.image_height},framerate=30/1 ! "
            f"videoconvert ! videoflip method={flip_method} ! "
            "video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1"
        )
        return [mipi_pipeline]

    def open_capture(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

        for candidate_pipeline in self.build_candidate_pipelines(self.camera_device, self.flip_method):
            cap = cv2.VideoCapture(candidate_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self.cap = cap
                self.pipeline = candidate_pipeline
                self.get_logger().info(f'Camera capture opened with pipeline: {candidate_pipeline}')
                return True

            cap.release()
            self.get_logger().warning(f'Failed to open candidate pipeline: {candidate_pipeline}')

        # GStreamer 미지원 OpenCV 빌드(pip wheel 등) 폴백: USB 카메라를 V4L2 로
        # 직접 열고, 프레임은 timer_callback 에서 목표 크기로 resize 한다.
        # 원래 GStreamer 경로도 videoscale(비율 무시 스트레치)이므로 결과 동일.
        if self.usb_cam_enabled:
            cap = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, _ = cap.read()
                if ok:
                    self.cap = cap
                    self.pipeline = f'v4l2-direct({self.camera_device})'
                    self.get_logger().warning(
                        'GStreamer 사용 불가 -> V4L2 직접 캡처 + resize 폴백으로 동작')
                    return True
                cap.release()

        self.cap = None
        self.pipeline = None
        return False

    def load_camera_device_overrides(self, default_usb_camera_device, default_mipi_camera_device):
        if not os.path.exists(self.vehicle_config_file):
            return default_usb_camera_device, default_mipi_camera_device

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return default_usb_camera_device, default_mipi_camera_device

        usb_camera_device = str(
            config_data.get('USB_CAM_DEVICE', default_usb_camera_device)
        ).strip()
        mipi_camera_device = str(
            config_data.get('MIPI_CAM_DEVICE', default_mipi_camera_device)
        ).strip()
        return usb_camera_device, mipi_camera_device

    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warning('Camera capture is not opened')
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warning('Failed to read frame')
            return

        # V4L2 직접 폴백 경로: 네이티브 해상도로 들어오므로 목표 크기로 맞춘다
        # (GStreamer videoscale 과 동일하게 비율 무시 스트레치).
        if frame.shape[1] != self.image_width or frame.shape[0] != self.image_height:
            frame = cv2.resize(frame, (self.image_width, self.image_height),
                               interpolation=cv2.INTER_AREA)

        success, encoded = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            self.get_logger().warning('Failed to encode frame as JPEG')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()

        self.publisher_.publish(msg)
        if self.debug_log:
            self.get_logger().info(f'Published frame: {len(msg.data)} bytes')

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
