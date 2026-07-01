#!/usr/bin/env python3
"""카메라 토픽(/camera/image/compressed)을 구독해 mp4로 녹화하는 도구.

초록불 -> 출발 장면처럼 카메라가 본 영상을 딥러닝 팀에 공유할 때 사용.
Ctrl+C 로 종료하면 mp4 파일을 저장하고 실제 수신 fps를 함께 알려준다.

실행:
    cd D-Racer-Kit
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 tools/record_camera_mp4.py            # 기본 30fps, recordings/ 에 저장
    python3 tools/record_camera_mp4.py --fps 30 --outdir recordings
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class CameraRecorder(Node):
    def __init__(self, topic, fps, output_path):
        super().__init__('camera_mp4_recorder')
        self.fps = float(fps)
        self.output_path = str(output_path)
        self.writer = None
        self.frame_size = None
        self.frame_count = 0
        self.first_stamp = None
        self.last_stamp = None

        # 카메라 발행 QoS와 동일하게 맞춰야 프레임을 수신한다.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(CompressedImage, topic, self.on_image, qos)
        self.get_logger().info(f'구독 시작: {topic} (fps={self.fps:g}) -> {self.output_path}')
        self.get_logger().info('녹화 중... 종료하려면 Ctrl+C')

    def on_image(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('프레임 디코드 실패 (건너뜀)')
            return

        if self.writer is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f'VideoWriter 열기 실패: {self.output_path}')
            self.get_logger().info(f'첫 프레임 수신: 해상도 {w}x{h}, 녹화 시작')

        self.writer.write(frame)
        self.frame_count += 1
        now = time.monotonic()
        if self.first_stamp is None:
            self.first_stamp = now
        self.last_stamp = now
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'{self.frame_count} 프레임 기록됨')

    def finish(self):
        if self.writer is not None:
            self.writer.release()
        if self.frame_count == 0:
            self.get_logger().warn(
                '수신한 프레임이 없습니다. 카메라 노드가 실행 중인지, 토픽 이름이 맞는지 확인하세요.'
            )
            return
        actual_fps = None
        if self.first_stamp is not None and self.last_stamp and self.last_stamp > self.first_stamp:
            actual_fps = (self.frame_count - 1) / (self.last_stamp - self.first_stamp)
        self.get_logger().info(f'저장 완료: {self.output_path}')
        self.get_logger().info(f'총 {self.frame_count} 프레임, 저장 fps={self.fps:g}')
        if actual_fps is not None:
            self.get_logger().info(f'실제 수신 fps ~ {actual_fps:.1f}')
            if abs(actual_fps - self.fps) > 3:
                self.get_logger().warn(
                    f'실제 수신 fps({actual_fps:.1f})가 저장 fps({self.fps:g})와 차이가 큽니다. '
                    f'재생 속도가 어색하면 --fps {actual_fps:.0f} 로 다시 녹화하세요.'
                )


def main():
    parser = argparse.ArgumentParser(description='카메라 토픽을 mp4로 녹화')
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--output', default=None, help='출력 mp4 경로 (지정 시 --outdir 무시)')
    parser.add_argument('--outdir', default='recordings', help='출력 폴더 (기본: recordings)')
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        output_path = outdir / f'camera_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
    output_path = output_path.resolve()

    rclpy.init()
    node = CameraRecorder(args.topic, args.fps, output_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
