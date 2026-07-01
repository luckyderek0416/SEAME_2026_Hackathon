#!/usr/bin/env python3
"""Sample the lane colour (HSV) off the real track to set lane_node's HSV ranges.

Two ways to grab a frame:
  # live, straight from the camera topic (run on the board with camera_node up):
  python3 hsv_sampler.py --topic /camera/image/compressed
  # from a saved image:
  python3 hsv_sampler.py --image frame.jpg

It looks at the bottom ROI (the road in front of the car), separates likely
WHITE and YELLOW lane pixels, and prints suggested inRange lo/hi triples you can
paste into lane_node params (white_hsv_lo/hi, yellow_hsv_lo/hi). OpenCV HSV:
H 0-179, S 0-255, V 0-255.

Tip: position the car so a lane line is in the lower-centre of the view, then run.
"""
import argparse
import sys

import cv2
import numpy as np


def pct(a, p):
    return int(np.percentile(a, p)) if a.size else 0


def report(name, hsv_pixels):
    if hsv_pixels.shape[0] < 30:
        print(f'  {name}: too few pixels ({hsv_pixels.shape[0]}) — not enough to suggest a range')
        return
    H, S, V = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    lo = (max(0, pct(H, 5) - 5), max(0, pct(S, 5) - 20), max(0, pct(V, 5) - 20))
    hi = (min(179, pct(H, 95) + 5), min(255, pct(S, 95) + 20), min(255, pct(V, 95) + 20))
    print(f'  {name}: {hsv_pixels.shape[0]} px | H[{H.min()}-{H.max()}] '
          f'S[{S.min()}-{S.max()}] V[{V.min()}-{V.max()}]')
    print(f'    suggested lo={list(lo)}  hi={list(hi)}')


def analyze(bgr, roi_top=0.55):
    h, w = bgr.shape[:2]
    roi = bgr[int(h * roi_top):h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    Hh, Ss, Vv = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    # white = bright + low saturation
    white = hsv[(Vv >= 170) & (Ss <= 70)]
    # yellow = yellow hue + saturated + bright
    yellow = hsv[(Hh >= 15) & (Hh <= 45) & (Ss >= 70) & (Vv >= 70)]

    print(f'ROI {roi.shape[1]}x{roi.shape[0]} (bottom {int((1-roi_top)*100)}%):')
    report('WHITE ', white)
    report('YELLOW', yellow)
    print('\nPaste into lane_node, e.g.:')
    print('  -p mask_mode:=hsv -p use_white:=true -p use_yellow:=true \\')
    print('  -p white_hsv_lo:="[..]" -p white_hsv_hi:="[..]" \\')
    print('  -p yellow_hsv_lo:="[..]" -p yellow_hsv_hi:="[..]"')


def grab_from_topic(topic):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    box = {}

    class Grab(Node):
        def __init__(self):
            super().__init__('hsv_sampler')
            self.create_subscription(CompressedImage, topic, self.cb, 10)

        def cb(self, msg):
            box['frame'] = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

    rclpy.init()
    n = Grab()
    for _ in range(200):
        rclpy.spin_once(n, timeout_sec=0.05)
        if 'frame' in box:
            break
    n.destroy_node()
    rclpy.shutdown()
    return box.get('frame')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image')
    ap.add_argument('--topic')
    ap.add_argument('--roi-top', type=float, default=0.55)
    a = ap.parse_args()
    if a.image:
        bgr = cv2.imread(a.image)
    elif a.topic:
        bgr = grab_from_topic(a.topic)
    else:
        print('give --image <file> or --topic <camera topic>')
        sys.exit(1)
    if bgr is None:
        print('no frame (bad path, or no camera publishing on the topic)')
        sys.exit(1)
    analyze(bgr, a.roi_top)


if __name__ == '__main__':
    main()
