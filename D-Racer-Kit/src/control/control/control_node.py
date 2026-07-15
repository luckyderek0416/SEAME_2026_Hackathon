import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml

from control_msgs.msg import Control
from joystick_msgs.msg import Joystick
from topst_utils.d3racer import D3Racer


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # ROS 파라미터
        self.declare_parameter('i2c_bus', 3)
        self.declare_parameter('pca9685_addr', 0x40)
        self.declare_parameter('steering_channel', 0)
        self.declare_parameter('throttle_channel', 1)
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('use_joystick_control', False)
        self.declare_parameter('joystick_topic', 'joystick')
        self.declare_parameter('control_topic', '/control')
        # command_hz: PCA9685(서보·ESC)에 실제로 값을 쓰는 액추에이션 주기.
        # decision_node 는 30Hz로 /control 을 발행하는데 이게 10Hz면 최대 100ms 다운샘플
        # 지연이 생겨 고속에서 조향 보정이 늦는다. 20Hz로 올려 지연을 절반으로 줄인다.
        # (I2C 쓰기 빈도 2배지만 apply_actuation 이 오류를 방어처리하므로 안전.
        #  decision 과 완전히 맞추려면 30.0 으로. 오버슈트의 주원인은 아니고 지연 개선용.)
        self.declare_parameter('command_hz', 20.0)
        # ESC 는 시동(arming)을 위해 시작 시 몇 초간 중립 throttle 신호를 유지해야 한다.
        # 그 전까지 들어오는 throttle 은 무시(중립 유지)하며, 그렇지 않으면 ESC 가 arm 되지 않는다.
        self.declare_parameter('esc_arm_sec', 3.0)
        # 명령 스테일 워치독 (07-15): decision_node 가 죽으면 이 노드는 마지막 /control
        # (예: 링 좌조향 + slow)을 영원히 반복 출력해 차가 폭주한다. 자율 모드에서
        # 명령이 이 시간 이상 끊기면 throttle 을 중립으로 강제한다 (조향은 유지 —
        # 급격한 조향 복귀가 더 위험). 명령이 다시 오면 자동 복귀. 0 = off.
        self.declare_parameter('cmd_stale_s', 0.5)

        i2c_bus = int(self.get_parameter('i2c_bus').value)
        pca9685_addr = int(self.get_parameter('pca9685_addr').value)
        steering_channel = int(self.get_parameter('steering_channel').value)
        throttle_channel = int(self.get_parameter('throttle_channel').value)
        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.use_joystick_control = bool(self.get_parameter('use_joystick_control').value)
        joystick_topic = str(self.get_parameter('joystick_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        command_hz = float(self.get_parameter('command_hz').value)
        if command_hz <= 0.0:
            raise ValueError('command_hz must be greater than 0')

        self.command_hz = command_hz
        self.esc_arm_sec = float(self.get_parameter('esc_arm_sec').value)
        self.cmd_stale_s = float(self.get_parameter('cmd_stale_s').value)
        self._last_cmd_time = None   # 마지막 /control 수신 시각 (워치독용)
        self._stale_active = False   # 워치독 발동 중 (로그 1회용)
        self.steer_trim = self.load_steer_trim()

        # PCA9685 초기화. respawn 재시작 시 직전 인스턴스가 아직 I2C fd 를 놓지 않아
        # OSError [Errno 16] Device or resource busy 가 날 수 있으므로 backoff 재시도한다.
        self.d3_racer = None
        for attempt in range(1, 11):
            try:
                self.d3_racer = D3Racer(
                    i2c_bus=i2c_bus,
                    pca9685_addr=pca9685_addr,
                    steering_channel=steering_channel,
                    throttle_channel=throttle_channel,
                )
                break
            except OSError as exc:
                self.get_logger().warning(
                    f'PCA9685(0x{pca9685_addr:02X}) init 실패 ({attempt}/10): {exc}. 0.5s 후 재시도'
                )
                time.sleep(0.5)
        if self.d3_racer is None:
            raise RuntimeError(
                f'PCA9685(0x{pca9685_addr:02X}) 를 여러 번 시도 후에도 열지 못함 (I2C busy). '
                '다른 control_node/launch 가 동시에 도는지, 중복 실행이 없는지 확인하세요.'
            )

        self.get_logger().info(
            'd3_racer configured:\n'
            f'  i2c_bus={i2c_bus}\n'
            f'  pca9685_addr=0x{pca9685_addr:02X}\n'
            f'  steering_channel={steering_channel}\n'
            f'  throttle_channel={throttle_channel}\n'
            f'  steer_trim={self.steer_trim}\n'
            f'  use_joystick_control={self.use_joystick_control}\n'
            f'  joystick_topic={joystick_topic}\n'
            f'  control_topic={control_topic}\n'
            f'  command_hz={self.command_hz}\n'
            f'  vehicle_config_file={self.vehicle_config_file}'
        )

        self.throttle = 0.0
        self.steering = self.steer_trim
        self.e_stop_active = False
        self._io_err_streak = 0  # 연속 I2C 쓰기 실패 카운트 (로그 rate-limit 용)

        # ESC arming: 명령을 받아들이기 전에 esc_arm_sec 동안 중립 throttle 을 유지한다.
        self.arming = self.esc_arm_sec > 0.0
        self._arm_start = self.get_clock().now()

        # 제어 입력
        self.create_subscription(
            Joystick,
            joystick_topic,
            self.joystick_callback,
            10,
        )
        self.create_subscription(
            Control,
            control_topic,
            self.control_callback,
            10,
        )

        # 명령 출력 루프
        self.timer = self.create_timer(1.0 / self.command_hz, self.timer_callback)

    def timer_callback(self):
        if self.e_stop_active:
            self.apply_actuation(self.steering, 0.0)
            return

        if self.arming:
            elapsed = (self.get_clock().now() - self._arm_start).nanoseconds / 1e9
            if elapsed < self.esc_arm_sec:
                # 조향은 자유롭게 움직여도 되지만, ESC 가 arm 되도록 throttle 은 중립을 유지한다.
                self.apply_actuation(self.steering, 0.0)
                return
            self.arming = False
            self.get_logger().info(f'ESC arming complete ({self.esc_arm_sec:g}s neutral). Throttle enabled.')

        # 워치독: 자율 모드에서 /control 이 cmd_stale_s 이상 끊기면 throttle 중립.
        # 첫 명령 수신 전에는 미적용 (throttle 초기값이 이미 0.0 이라 무해).
        if (self.cmd_stale_s > 0.0 and not self.use_joystick_control
                and self._last_cmd_time is not None):
            age = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
            if age >= self.cmd_stale_s:
                if not self._stale_active:
                    self._stale_active = True
                    self.get_logger().warning(
                        f'/control 스테일 {age:.2f}s >= {self.cmd_stale_s:g}s — '
                        'throttle 중립 강제 (decision 사망 의심)')
                self.apply_actuation(self.steering, 0.0)
                return
            if self._stale_active:
                self._stale_active = False
                self.get_logger().info('/control 재개 — throttle 복귀')

        self.apply_actuation(self.steering, self.throttle)

    def apply_actuation(self, steering, throttle):
        # 주행 중 진동/전압 강하로 인한 순간 I2C 오류(OSError 등)가 timer_callback 을 통해
        # 노드 전체를 죽이지 않도록 방어한다. 예외 시 로그만 남기고 계속 구동한다.
        try:
            self.d3_racer.set_steering_percent(float(steering))
            self.d3_racer.set_throttle_percent(float(throttle))
            self._io_err_streak = 0
        except Exception as exc:
            self._io_err_streak += 1
            # 연속 실패 시 10Hz 로 스팸나지 않게 첫 실패 + 50회마다만 로그.
            if self._io_err_streak == 1 or self._io_err_streak % 50 == 0:
                self.get_logger().warning(
                    f'I2C actuation write failed ({self._io_err_streak}회 연속): {exc}'
                )

    def joystick_callback(self, msg: Joystick):
        if bool(msg.e_stop_en):
            self.engage_e_stop()
            return

        if self.e_stop_active or not self.use_joystick_control:
            return

        self.steering = float(msg.control_msg.steering)
        self.throttle = float(msg.control_msg.throttle)

    def control_callback(self, msg: Control):
        if self.e_stop_active or self.use_joystick_control:
            return

        self._last_cmd_time = self.get_clock().now()
        self.steering = float(msg.steering)
        self.throttle = float(msg.throttle)

    def engage_e_stop(self):
        if self.e_stop_active:
            return

        self.e_stop_active = True
        self.throttle = 0.0
        self.apply_actuation(self.steering, 0.0)
        self.get_logger().warning('E-STOP engaged. Ignoring incoming throttle commands.')

    def load_steer_trim(self):
        if not os.path.exists(self.vehicle_config_file):
            return 0.0

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as config_stream:
                config_data = yaml.safe_load(config_stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return 0.0

        return float(config_data.get('STEER_TRIM', 0.0))

    def destroy_node(self):
        try:
            if hasattr(self, 'd3_racer') and self.d3_racer is not None:
                self.apply_actuation(self.steer_trim, 0.0)
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
