"""미션 주행 노드 — NCNN YOLO 인지 + 라인추종 + 행동(신호등/갈림길).

상태머신:
  WAIT_START : 정지(throttle 0). green_light 보이면 -> DRIVING(출발)
  DRIVING    : 라인추종 주행. throttle 0.20 에서 시작해 서서히 증가.
               left_sign  -> 잠깐 좌회전 강제
               right_sign -> 잠깐 우회전 강제
               red_light  -> STOPPED(정지)
  STOPPED    : 정지

인지는 어제 학습한 best_ncnn_model(NCNN) 을 그대로 사용.
조향 기본값은 라인추종(LaneFollower+PD), 갈림길에서만 강제 조향으로 덮어씀.
값(throttle/조향/임계)은 실제 주행으로 튜닝.
"""

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from control_msgs.msg import Control

from lane_follow import LaneFollower
from pd_controller import PDController
from yolo_detector import YoloDetector


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')
        self.bridge = CvBridge()

        # [파라미터]
        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('model_path', os.path.expanduser(
            '~/ros2_ws/src/auto_driving_pkg/training/best_ncnn_model'))
        self.declare_parameter('conf', 0.5)            # YOLO 신뢰도 임계
        self.declare_parameter('min_area', 0.01)       # 이만큼 커야(가까워야) 반응
        self.declare_parameter('detect_every', 3)      # N프레임마다 YOLO(부하↓)
        self.declare_parameter('start_throttle', 0.20)  # 출발 속도(=20%)
        self.declare_parameter('cruise_throttle', 0.40)  # 최고 순항 속도
        self.declare_parameter('ramp_rate', 0.05)      # 초당 throttle 증가량
        self.declare_parameter('fork_steer', 0.5)      # 갈림길 강제 조향 크기
        self.declare_parameter('fork_duration', 2.0)   # 갈림길 강제 시간(초)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('kd', 0.1)

        g = lambda n: self.get_parameter(n).value
        image_topic = str(g('image_topic'))
        self.min_area = float(g('min_area'))
        self.detect_every = int(g('detect_every'))
        self.start_throttle = float(g('start_throttle'))
        self.cruise_throttle = float(g('cruise_throttle'))
        self.ramp_rate = float(g('ramp_rate'))
        self.fork_steer = float(g('fork_steer'))
        self.fork_duration = float(g('fork_duration'))

        self.lane = LaneFollower()
        self.pd = PDController(kp=float(g('kp')), kd=float(g('kd')))
        self.get_logger().info(f"NCNN 모델 로드: {g('model_path')}")
        self.detector = YoloDetector(str(g('model_path')),
                                     conf=float(g('conf')), imgsz=320)

        # 상태
        self.state = 'WAIT_START'
        self.start_time = 0.0
        self.fork_dir = 0          # -1 좌 / +1 우 / 0 없음
        self.fork_until = 0.0
        self.frame_i = 0
        self.last_dets = []

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.sub = self.create_subscription(CompressedImage, image_topic,
                                            self.image_callback, qos)
        self.pub = self.create_publisher(Control, str(g('control_topic')), 10)
        self.get_logger().info('=== 미션 주행 노드 시작 (WAIT_START) ===')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            t = self.now()
            self.frame_i += 1

            # 1) YOLO 인지 (N프레임마다, 가까운 것만)
            if self.frame_i % self.detect_every == 0:
                self.last_dets = [d for d in self.detector.detect(frame)
                                  if d[3] >= self.min_area]
            seen = {d[0] for d in self.last_dets}     # 보이는 클래스 집합

            # 2) 라인추종 기본 조향
            angle, _ = self.lane.compute_angle(frame)
            steering = self.pd.step(angle / self.lane.max_angle)
            throttle = 0.0

            # 3) 상태머신
            if self.state == 'WAIT_START':
                steering, throttle = 0.0, 0.0
                if 'green_light' in seen:
                    self.state = 'DRIVING'
                    self.start_time = t
                    self.get_logger().info('🟢 초록불 → 출발')

            elif self.state == 'DRIVING':
                # throttle: 20%에서 시작해 서서히 순항속도까지
                throttle = min(self.cruise_throttle,
                               self.start_throttle + self.ramp_rate * (t - self.start_time))
                # 갈림길: 표지판 보이면 일정시간 그 방향 강제
                if 'left_sign' in seen:
                    self.fork_dir, self.fork_until = -1, t + self.fork_duration
                    self.get_logger().info('⬅️ 좌회전 표지판')
                elif 'right_sign' in seen:
                    self.fork_dir, self.fork_until = +1, t + self.fork_duration
                    self.get_logger().info('➡️ 우회전 표지판')
                if t < self.fork_until:
                    steering = self.fork_dir * self.fork_steer   # 강제 조향
                # 도착: 빨간불 정지
                if 'red_light' in seen:
                    self.state = 'STOPPED'
                    self.get_logger().info('🔴 빨간불 → 정지')

            if self.state == 'STOPPED':
                steering, throttle = 0.0, 0.0

            # 4) 발행
            out = Control()
            out.header.stamp = self.get_clock().now().to_msg()
            out.steering = float(max(-1.0, min(1.0, steering)))
            out.throttle = float(throttle)
            self.pub.publish(out)
        except Exception as e:
            self.get_logger().error(f'미션 처리 오류: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드를 종료합니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
