import os
import csv
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from joystick_msgs.msg import Joystick


class MakeDBNode(Node):
    def __init__(self):
        super().__init__('makedb_node')

        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('joystick_topic', '/joystick')
        self.declare_parameter('output_dir', 'dataset')
        self.declare_parameter('save_every_n_frames', 1)
        self.declare_parameter('require_recording_button', False)
        self.declare_parameter('min_abs_throttle', 0.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.joystick_topic = self.get_parameter('joystick_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.save_every_n_frames = int(self.get_parameter('save_every_n_frames').value)
        self.require_recording_button = bool(self.get_parameter('require_recording_button').value)
        self.min_abs_throttle = float(self.get_parameter('min_abs_throttle').value)

        self.images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

        self.csv_path = os.path.join(self.output_dir, 'labels.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['filename', 'steering', 'throttle', 'timestamp'])

        self.latest_steering = 0.0
        self.latest_throttle = 0.0
        self.latest_is_recording = True

        self.image_count = 0
        self.saved_count = 0

        self.joystick_sub = self.create_subscription(
            Joystick,
            self.joystick_topic,
            self.joystick_callback,
            10
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info('MakeDB node started')
        self.get_logger().info(f'image_topic: {self.image_topic}')
        self.get_logger().info(f'joystick_topic: {self.joystick_topic}')
        self.get_logger().info(f'output_dir: {self.output_dir}')

    def joystick_callback(self, msg):
        self.latest_steering = float(msg.control_msg.steering)
        self.latest_throttle = float(msg.control_msg.throttle)
        self.latest_is_recording = bool(msg.is_recording)

    def image_callback(self, msg):
        self.image_count += 1

        if self.image_count % self.save_every_n_frames != 0:
            return

        if self.require_recording_button and not self.latest_is_recording:
            return

        if abs(self.latest_throttle) < self.min_abs_throttle:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warning('Failed to decode image')
            return

        filename = f'{self.saved_count:06d}.jpg'
        image_path = os.path.join(self.images_dir, filename)

        cv2.imwrite(image_path, frame)

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.csv_writer.writerow([
            filename,
            self.latest_steering,
            self.latest_throttle,
            timestamp
        ])
        self.csv_file.flush()

        self.saved_count += 1

        if self.saved_count % 50 == 0:
            self.get_logger().info(
                f'saved {self.saved_count} images | '
                f'steering={self.latest_steering:.3f}, '
                f'throttle={self.latest_throttle:.3f}'
            )

    def destroy_node(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MakeDBNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. stopping makedb.')
    finally:
        node.get_logger().info(f'total saved images: {node.saved_count}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
