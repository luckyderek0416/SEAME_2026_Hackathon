import time

from smbus2 import SMBus

import rclpy
from rclpy.node import Node

from battery_msgs.msg import Battery
from topst_utils.ina219 import INA219, INA_ADDR, I2C_BUS


class BatteryPublisher(Node):
    def __init__(self):
        super().__init__('battery_node')

        # ROS 파라미터
        self.declare_parameter('publish_topic', 'battery_status')
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('i2c_bus', I2C_BUS)
        self.declare_parameter('ina_addr', INA_ADDR)
        self.declare_parameter('r_shunt_ohm', 0.1)
        self.declare_parameter('max_current_a', 2.0)
        self.declare_parameter('min_voltage', 6.4)
        self.declare_parameter('max_voltage', 8.4)
        self.declare_parameter('use_load_voltage', True)
        self.declare_parameter('debug_log', False)

        publish_topic = str(self.get_parameter('publish_topic').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')

        i2c_bus = int(self.get_parameter('i2c_bus').value)
        ina_addr = int(self.get_parameter('ina_addr').value)
        r_shunt_ohm = float(self.get_parameter('r_shunt_ohm').value)
        max_current_a = float(self.get_parameter('max_current_a').value)
        self.min_voltage = float(self.get_parameter('min_voltage').value)
        self.max_voltage = float(self.get_parameter('max_voltage').value)
        if self.max_voltage <= self.min_voltage:
            raise ValueError('max_voltage must be greater than min_voltage')

        self.use_load_voltage = bool(self.get_parameter('use_load_voltage').value)
        self.debug_log = bool(self.get_parameter('debug_log').value)
        self.publish_hz = publish_hz

        self.publisher_ = self.create_publisher(Battery, publish_topic, 10)
        self.bus = SMBus(i2c_bus)
        # control_node 와 동시에 시작하면 같은 I2C 버스를 동시에 초기화하다 순간 충돌(EBUSY 등)이
        # 날 수 있다. 한 번의 글리치로 노드가 죽지 않도록 backoff 재시도한다.
        self.ina219 = None
        for attempt in range(1, 11):
            try:
                self.ina219 = INA219(
                    self.bus,
                    ina_addr,
                    r_shunt_ohm=r_shunt_ohm,
                    max_current_a=max_current_a,
                )
                break
            except OSError as exc:
                self.get_logger().warning(
                    f'INA219(0x{ina_addr:02X}) init 실패 ({attempt}/10): {exc}. 0.5s 후 재시도'
                )
                time.sleep(0.5)
        if self.ina219 is None:
            raise RuntimeError(
                f'INA219(0x{ina_addr:02X}) 를 여러 번 시도 후에도 열지 못함 (I2C busy).'
            )
        self.timer = self.create_timer(1.0 / self.publish_hz, self.timer_callback)

        self.get_logger().info(
            f'[Battery Publisher] : topic={publish_topic} \n'
            f'[publish_hz] : {self.publish_hz} \n'
            f'[i2c_bus] : {i2c_bus} \n'
            f'[ina_addr] : 0x{ina_addr:02X} \n'
            f'[min_voltage] : {self.min_voltage} \n'
            f'[max_voltage] : {self.max_voltage} \n'
            f'[use_load_voltage] : {self.use_load_voltage} \n'
            f'[debug_log] : {self.debug_log} \n'
        )

    def voltage_to_percentage(self, voltage):
        normalized = ((voltage - self.min_voltage) / (self.max_voltage - self.min_voltage)) * 100.0
        return max(0.0, min(100.0, normalized))

    def timer_callback(self):
        try:
            bus_voltage = self.ina219.bus_voltage
            shunt_voltage = self.ina219.shunt_voltage
            current_ma = self.ina219.current

            load_voltage = bus_voltage + shunt_voltage if self.use_load_voltage else bus_voltage
            battery_percentage = self.voltage_to_percentage(load_voltage)

            msg = Battery()
            msg.battery_status = float(battery_percentage)
            self.publisher_.publish(msg)

            if self.debug_log:
                self.get_logger().info('\n'
                    f'[Battery Status] : bus={bus_voltage:.3f}V \n'
                    f'[shunt] : {shunt_voltage:.5f}V \n'
                    f'[load] : {load_voltage:.3f}V \n'
                    f'[current] : {current_ma:.1f}mA  \n'
                    f'[status] : {battery_percentage:.1f}% \n'
                )
        except Exception as exc:
            self.get_logger().error(f'Failed to read battery status: {exc}')

    def destroy_node(self):
        try:
            if hasattr(self, 'bus') and self.bus is not None:
                self.bus.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BatteryPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
