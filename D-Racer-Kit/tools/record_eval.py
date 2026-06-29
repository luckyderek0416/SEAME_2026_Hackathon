#!/usr/bin/env python3
"""카메라 높이/흔들림 평가용 녹화 스크립트.

자율주행 스택이 떠 있는 상태(camera_node + lane_node)에서 실행하면,
한 번의 실행마다 영상 2개와 인식 품질 수치를 만든다.

  - rec_<label>_raw.mp4   원본 카메라  → 카메라 흔들림(진동) 눈으로 확인
  - rec_<label>_lane.mp4  차선 디버그   → 차선이 잘 잡히는지(빨간 중심선) 확인
  - eval_summary.csv       label, 인식률(%), offset 평균/표준편차 한 줄 누적

offset 표준편차(std)가 작을수록 = 차선 중심 추정이 덜 떨림(= 흔들림 적고 안정적).
인식률(lane_found %)이 높을수록 = 그 높이에서 차선이 잘 잡힘.

사용 (보드에서):
    cd ~/D-Racer-Kit
    source install/setup.bash          # perception_msgs(LaneState) 임포트용
    # 다른 터미널에서 스택 실행 중이어야 함:
    #   ros2 launch decision auto_race.launch.py course:=out
    python3 tools/record_eval.py --label 20cm --seconds 30

높이를 바꿔가며 --label 만 다르게 해서 여러 번 실행 → eval_summary.csv 로 비교.
종료는 --seconds 경과 시 자동, 또는 Ctrl+C.
"""
import argparse
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# LaneState(인식률/offset 수치)는 워크스페이스가 source 돼 있어야 임포트됨.
try:
    from perception_msgs.msg import LaneState
    HAVE_LANESTATE = True
except Exception:
    HAVE_LANESTATE = False


class Recorder(Node):
    def __init__(self, args):
        super().__init__('record_eval')
        self.label = args.label
        self.fps = float(args.fps)
        self.seconds = float(args.seconds)
        self.outdir = args.outdir
        os.makedirs(self.outdir, exist_ok=True)

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.raw_writer = None
        self.lane_writer = None

        # 인식 품질 통계
        self.n_lane_msgs = 0
        self.n_found = 0
        self.offsets = []

        self.start_t = None  # 첫 프레임 수신 시각(ROS time, 초)

        self.create_subscription(
            CompressedImage, '/camera/image/compressed', self.on_raw, 10)
        self.create_subscription(
            CompressedImage, '/perception/lane/debug', self.on_lane_img, 10)
        if HAVE_LANESTATE:
            self.create_subscription(
                LaneState, '/perception/lane', self.on_lane_state, 10)
        else:
            self.get_logger().warn(
                "perception_msgs 미임포트 → 인식률/offset 수치는 건너뜀. "
                "'source install/setup.bash' 후 다시 실행하면 수치도 기록됩니다.")

        self.get_logger().info(
            f"녹화 시작: label={self.label}, fps={self.fps}, "
            f"{'무제한(Ctrl+C로 종료)' if self.seconds <= 0 else str(self.seconds)+'초'}")

    # --- 시간/종료 관리 ---
    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _tick(self):
        t = self._now()
        if self.start_t is None:
            self.start_t = t
        if self.seconds > 0 and (t - self.start_t) >= self.seconds:
            raise KeyboardInterrupt  # spin 루프 빠져나가 정상 종료

    def _decode(self, msg):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

    # --- 콜백 ---
    def on_raw(self, msg):
        self._tick()
        frame = self._decode(msg)
        if frame is None:
            return
        if self.raw_writer is None:
            h, w = frame.shape[:2]
            path = os.path.join(self.outdir, f'rec_{self.label}_raw.mp4')
            self.raw_writer = cv2.VideoWriter(path, self.fourcc, self.fps, (w, h))
            self.get_logger().info(f'raw → {path} ({w}x{h})')
        self.raw_writer.write(frame)

    def on_lane_img(self, msg):
        self._tick()
        frame = self._decode(msg)
        if frame is None:
            return
        rate = (100.0 * self.n_found / self.n_lane_msgs) if self.n_lane_msgs else 0.0
        elapsed = (self._now() - self.start_t) if self.start_t else 0.0
        txt = f'{self.label}  t={elapsed:4.1f}s  found={rate:5.1f}%'
        cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)
        if self.lane_writer is None:
            h, w = frame.shape[:2]
            path = os.path.join(self.outdir, f'rec_{self.label}_lane.mp4')
            self.lane_writer = cv2.VideoWriter(path, self.fourcc, self.fps, (w, h))
            self.get_logger().info(f'lane → {path} ({w}x{h})')
        self.lane_writer.write(frame)

    def on_lane_state(self, msg):
        self.n_lane_msgs += 1
        if msg.lane_found:
            self.n_found += 1
            self.offsets.append(float(msg.offset))

    # --- 종료 시 마무리 ---
    def finish(self):
        if self.raw_writer:
            self.raw_writer.release()
        if self.lane_writer:
            self.lane_writer.release()

        rate = (100.0 * self.n_found / self.n_lane_msgs) if self.n_lane_msgs else 0.0
        if self.offsets:
            arr = np.array(self.offsets)
            off_mean, off_std = float(arr.mean()), float(arr.std())
        else:
            off_mean = off_std = float('nan')

        line = (f'{self.label},{rate:.1f},{off_mean:.4f},{off_std:.4f},'
                f'{self.n_found}/{self.n_lane_msgs}')
        summary = os.path.join(self.outdir, 'eval_summary.csv')
        new = not os.path.exists(summary)
        with open(summary, 'a') as f:
            if new:
                f.write('label,found_rate_pct,offset_mean,offset_std,found/total\n')
            f.write(line + '\n')

        self.get_logger().info(
            f'\n==== {self.label} 결과 ====\n'
            f'  차선 인식률 : {rate:.1f}%  (높을수록 좋음)\n'
            f'  offset std  : {off_std:.4f} (작을수록 안정/덜 흔들림)\n'
            f'  요약 누적   : {summary}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='run', help='이번 측정 이름 (예: 15cm, 20cm)')
    ap.add_argument('--seconds', default=30, type=float, help='녹화 길이(초). 0=Ctrl+C까지')
    ap.add_argument('--fps', default=20, type=float, help='저장 영상 fps')
    ap.add_argument('--outdir', default='recordings', help='저장 폴더')
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
