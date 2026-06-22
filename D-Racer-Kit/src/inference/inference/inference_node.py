import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from control_msgs.msg import Control

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('camera_topic', 'camera/image/compressed')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('model_path', '')
        self.declare_parameter('debug_log', False)
        self.declare_parameter('throttle_speed', 0.2)

        self.camera_topic = str(self.get_parameter('camera_topic').value)
        self.control_topic = str(self.get_parameter('control_topic').value)
        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.throttle_speed = float(self.get_parameter('throttle_speed').value)

        self.control_pub = self.create_publisher(Control, self.control_topic, 10)
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(
            f'Inference node initialized. camera_topic={self.camera_topic}, '
            f'control_topic={self.control_topic}, model_path={self.model_path}'
        )

        self.model = self.load_model(self.model_path)

    def load_model(self, model_path: str):
        if load_model is None:
            self.get_logger().warning(
                'TensorFlow is not installed. Using fallback dummy inference.'
            )
            return None

        if model_path and os.path.exists(model_path):
            try:
                model = load_model(model_path, compile=False)
                self.get_logger().info(f'Loaded model from {model_path}')
                return model
            except Exception as exc:
                self.get_logger().error(
                    f'Failed to load model: {exc}. Using fallback dummy inference.'
                )
                return None

        self.get_logger().warning(
            'Model file not found or model_path is empty. Using fallback dummy inference.'
        )
        return None

    def image_callback(self, msg: CompressedImage):
        frame = self.decode_image(msg)
        if frame is None:
            self.get_logger().warning('Failed to decode camera frame')
            return

        steering, throttle = self.predict_control(frame)

        control_msg = Control()
        control_msg.steering = float(steering)
        control_msg.throttle = float(throttle)
        self.control_pub.publish(control_msg)

        if self.debug_log:
            self.get_logger().info(
                f'Published control: steering={steering:.3f}, throttle={throttle:.3f}'
            )

    @staticmethod
    def decode_image(msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    @staticmethod
    def preprocess_image(image):
        if image is None:
            return None

        h = image.shape[0]
        cropped = image[int(h * 0.35):int(h * 0.95), :, :]
        resized = cv2.resize(cropped, (200, 66))
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        normalized = yuv.astype(np.float32) / 255.0
        return normalized

    def predict_control(self, frame):
        steering = 0.0
        if self.model is not None:
            preprocessed = self.preprocess_image(frame)
            if preprocessed is not None:
                inp = np.expand_dims(preprocessed, axis=0)
                try:
                    prediction = self.model.predict(inp, verbose=0)
                    steering = float(np.asarray(prediction).flatten()[0])
                except Exception as exc:
                    self.get_logger().warning(
                        f'Model prediction failed: {exc}. Using fallback steering=0.0.'
                    )

        throttle = self.throttle_speed
        return steering, throttle


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
