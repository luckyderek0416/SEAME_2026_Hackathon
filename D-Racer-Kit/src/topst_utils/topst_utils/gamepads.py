# Copyright (C) 2022 twyleg
import array
import struct
import os
from typing import Optional, Tuple

axis_attributes = {
    0x00: 'L_STICK_X',
    0x01: 'L_STICK_Y',
    0x02: 'L2_TRIGGER',
    0x03: 'R_STICK_X',
    0x04: 'R_STICK_Y',
    0x05: 'R2_TRIGGER',
    0x10: 'DPAD_X',
    0x11: 'DPAD_Y' 
}

button_attributes = {
    0x130: 'A',
    0x131: 'B',
    0x133: 'X',
    0x134: 'Y',
    0x136: 'L1',
    0x137: 'R1',
    0x13a: 'SELECT',
    0x13b: 'START',
    0x13c: 'HOME'
}

class Vector3f:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class ShanWanGamepadInput:
    def __init__(self) -> None:
        self.analog_stick_left = Vector3f()
        self.analog_stick_right = Vector3f()
        self.dpad_up = None
        self.dpad_down = None
        self.dpad_left = None
        self.dpad_right = None
        self.button_L1 = None
        self.button_L2 = None
        self.button_R1 = None
        self.button_R2 = None
        self.button_x = None
        self.button_a = None
        self.button_b = None
        self.button_y = None
        self.button_select = None
        self.button_start = None
        self.button_home = None


class Joystick(object):
    '''
    donkeycar 프로젝트 기반
    MIT License
    https://github.com/autorope/donkeycar/blob/4.3.6.2/donkeycar/parts/controller.py

    물리 조이스틱에 대한 인터페이스.
    조이스틱은 사용 가능한 버튼과 축의 이름·값을 모두 보관하며,
    폴링(poll)하여 상태 변화를 확인할 수 있다.
    '''
    def __init__(self, dev_fn='/dev/input/js0') -> None:
        self.axis_states = {}
        self.button_states = {}
        self.axis_names = {}
        self.button_names = {}
        self.axis_map = []
        self.button_map = []
        self.jsdev = None
        self.dev_fn = dev_fn


    def init(self) -> None:
        """
        리눅스 디바이스 트리 경로를 받아
        사용 가능한 버튼과 축을 조회한다.
        """
        try:
            from fcntl import ioctl
        except ModuleNotFoundError:
            self.num_axes = 0
            self.num_buttons = 0
            return False

        if not os.path.exists(self.dev_fn):
            return False

        '''
        디바이스 연결 설정과 버튼 매핑을 위해 한 번만 호출
        '''
        # 조이스틱 디바이스를 연다.
        self.jsdev = open(self.dev_fn, 'rb')

        # 디바이스 이름을 가져온다.
        buf = array.array('B', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf) # JSIOCGNAME(len)
        self.js_name = buf.tobytes().decode('utf-8')

        # 축과 버튼의 개수를 가져온다.
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf) # JSIOCGAXES
        self.num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf) # JSIOCGBUTTONS
        self.num_buttons = buf[0]

        # 축 맵을 가져온다.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf) # JSIOCGAXMAP

        for axis in buf[:self.num_axes]:
            # axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
            axis_name = axis_attributes.get(axis, f'unknown(0x{axis:02x})')
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # 버튼 맵을 가져온다.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf) # JSIOCGBTNMAP

        for btn in buf[:self.num_buttons]:
            btn_name = button_attributes.get(btn, f'unknown(0x{btn:03x})')
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0

        return True


    def show_map(self) -> None:
        '''
        이 조이스틱에서 발견된 버튼과 축을 나열
        '''
        print('axis found')
        for i in range(self.num_axes):
            print(f'{i}: {self.axis_map[i]}, state={self.axis_states[self.axis_map[i]]:.2f}')
        print('buttons found')
        for i in range(self.num_buttons):
            print(f'{i}: {self.button_map[i]}, state={self.button_states[self.button_map[i]]}')

    def poll(self) -> Tuple[Optional[str], Optional[int], Optional[bool], Optional[str], Optional[int], Optional[float]]:
        '''
        조이스틱 상태를 조회한다. 눌린 버튼(있다면)과 움직인 축(있다면)을 반환한다.
        button_state 는 변화 없음/눌림/떼짐에 따라 각각 None, 1, 0 이다.
        axis_val 은 -1 ~ +1 범위의 float 이다. button 과 axis 는 init 의
        축 맵에서 결정된 문자열 라벨이다.
        '''
        button_name: Optional[str] = None
        button_number: Optional[int] = None
        button_state: Optional[bool] = None
        axis_name: Optional[str] = None
        axis_number: Optional[int] = None
        axis_val: Optional[float] = None

        if self.jsdev is None:
            return button_name, button_number, button_state, axis_name, axis_number, axis_val

        # 메인 이벤트 루프
        evbuf = self.jsdev.read(8)

        if evbuf:
            tval, value, typev, number = struct.unpack('IhBB', evbuf)

            if typev & 0x80:
                # 초기화 이벤트는 무시
                return button_name, button_number, button_state, axis_name, axis_number, axis_val

            if typev & 0x01:
                button_name = self.button_map[number]
                if button_name:
                    self.button_states[button_name] = value
                    button_number = number
                    button_state = value

            if typev & 0x02:
                axis_name = self.axis_map[number]
                if axis_name:
                    fvalue = value / 32767.0
                    self.axis_states[axis_name] = fvalue
                    axis_number = number
                    axis_val = fvalue

        return button_name, button_number, button_state, axis_name, axis_number, axis_val


class ShanWanGamepad(Joystick):

    def __init__(self)  -> None:
        super(ShanWanGamepad, self).__init__()
        super(ShanWanGamepad, self).init()
        self.gamepad_input = ShanWanGamepadInput()

    def read_data(self) -> ShanWanGamepadInput:
        button_name, button_number, button_state, axis_name, axis_number, axis_val = super(ShanWanGamepad, self).poll()

        # 조이스틱
        if axis_name == 'L_STICK_Y':        # 가속 제어: 앞으로 밀면 음수, 뒤로 당기면 양수
            self.gamepad_input.analog_stick_left.y  = -axis_val
        elif axis_name == 'R_STICK_X':      # 조향 제어: 왼쪽이 음수, 오른쪽이 양수
            self.gamepad_input.analog_stick_right.x = -axis_val
        # DPAD (십자키)
        elif axis_name == 'DPAD_X':         # DPAD 좌/우: 왼쪽 -1, 오른쪽 +1
            self.gamepad_input.dpad_left = axis_val if axis_val < 0 else None
            self.gamepad_input.dpad_right = axis_val if axis_val > 0 else None
        elif axis_name == 'DPAD_Y':         # DPAD 상/하: 위 -1, 아래 +1
            self.gamepad_input.dpad_up = axis_val if axis_val < 0 else None
            self.gamepad_input.dpad_down = axis_val if axis_val > 0 else None

        # 버튼
        if button_name == 'L1':             # 가속 비율 증가
            self.gamepad_input.button_L1 = button_state
        elif button_name == 'R1':           # 가속 비율 감소
            self.gamepad_input.button_R1 = button_state
        elif button_name == 'X':            
            self.gamepad_input.button_x = button_state
        elif button_name == 'A':
            self.gamepad_input.button_a = button_state
        elif button_name == 'B':            # 조향 트림 보정 증가
            self.gamepad_input.button_b = button_state
        elif button_name == 'Y':            # 조향 트림 보정 감소
            self.gamepad_input.button_y = button_state
        elif button_name == 'SELECT':
            self.gamepad_input.button_select = button_state
        elif button_name == 'START':        # 데이터 기록 토글
            self.gamepad_input.button_start = button_state
        elif button_name == 'HOME':
            self.gamepad_input.button_home = button_state
        
        return self.gamepad_input



# 조이스틱 테스트 코드
# js = Joystick()
# js.init()

# while True:
#     js.poll()  # 이벤트 올 때까지 블로킹될 수 있음
#     js.show_map()
