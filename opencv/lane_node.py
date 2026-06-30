"""라인 추종 주행 ROS 노드 — 카메라 → 비전(각도) → PD → /control.

흐름:
    /camera/image/compressed
        -> LaneFollower.compute_angle()   목표 각도(도)
        -> 각도 정규화(error)
        -> PDController.step()             부드러운 조향값(-1~1)
        -> 속도 정책                       throttle(커브면 감속)
        -> control_msgs/Control 로 /control 발행 -> D-Racer control_node -> 차

파라미터(Kp/Kd/속도/HSV 등)는 ros2 param 으로 주행 중 튜닝.
※ 팀 PID를 쓰면 use_pd:=false 로 두고 각도(정규화)를 그대로 steering 에 실음.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

from control_msgs.msg import Control

from lane_follow import LaneFollower
from pd_controller import PDController


class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_node')
        self.bridge = CvBridge()
        self.get_logger().info('=== 라인 추종 주행 노드 시작 ===')

        # [파라미터]
        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('mode', 'white')          # white/yellow
        self.declare_parameter('use_pd', True)           # PD 사용 여부
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('kd', 0.1)
        self.declare_parameter('base_throttle', 0.15)    # 기본 전진 속도
        self.declare_parameter('min_throttle', 0.10)     # 급커브 시 최저 속도
        self.declare_parameter('curve_slowdown', 0.5)    # |steering|에 따른 감속량

        image_topic = str(self.get_parameter('image_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        self.use_pd = bool(self.get_parameter('use_pd').value)
        self.base_throttle = float(self.get_parameter('base_throttle').value)
        self.min_throttle = float(self.get_parameter('min_throttle').value)
        self.curve_slowdown = float(self.get_parameter('curve_slowdown').value)

        self.lane = LaneFollower()
        self.lane.mode = str(self.get_parameter('mode').value)
        self.pd = PDController(
            kp=float(self.get_parameter('kp').value),
            kd=float(self.get_parameter('kd').value),
        )

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.sub = self.create_subscription(CompressedImage, image_topic,
                                            self.image_callback, qos)
        self.pub = self.create_publisher(Control, control_topic, 10)

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')

            # 1) 비전: 목표 각도
            angle, _ = self.lane.compute_angle(frame)
            error = angle / self.lane.max_angle          # -1 ~ +1 정규화

            # 2) 조향: PD 사용 or 각도 그대로(팀 PID에 맡김)
            steering = self.pd.step(error) if self.use_pd else error

            # 3) 속도: 급커브(조향 큼)일수록 감속
            throttle = self.base_throttle - self.curve_slowdown * abs(steering)
            throttle = max(self.min_throttle, throttle)

            # 4) 발행
            out = Control()
            out.header.stamp = self.get_clock().now().to_msg()
            out.steering = float(steering)
            out.throttle = float(throttle)
            self.pub.publish(out)

            self.get_logger().info(
                f'[주행] angle={angle:+.1f} steer={steering:+.2f} thr={throttle:.2f}')
        except Exception as e:
            self.get_logger().error(f'주행 처리 오류: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드를 종료합니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
