#!/usr/bin/env python3
"""Identify which ArUco dictionary a marker image belongs to.

Usage:
    python3 identify_aruco.py <marker_image.png|jpg>

Tries every predefined dictionary (normal + inverted) and reports which one(s)
detect a marker, with the marker ID. Use the reported dict for the
'aruco_dict' / 'aruco_inverted' parameters of mission_control.
"""
import sys
import cv2
import numpy as np

DICTS = [
    '4X4_50', '4X4_100', '4X4_250', '4X4_1000',
    '5X5_50', '5X5_100', '5X5_250', '5X5_1000',
    '6X6_50', '6X6_100', '6X6_250', '6X6_1000',
    '7X7_50', '7X7_100', '7X7_250', '7X7_1000',
    'ARUCO_ORIGINAL',
    'APRILTAG_16h5', 'APRILTAG_25h9', 'APRILTAG_36h10', 'APRILTAG_36h11',
]


def make_detector(dictionary, inverted):
    a = cv2.aruco
    if hasattr(a, 'ArucoDetector'):              # OpenCV >= 4.7
        p = a.DetectorParameters()
        if inverted and hasattr(p, 'detectInvertedMarker'):
            p.detectInvertedMarker = True
        det = a.ArucoDetector(dictionary, p)
        return lambda g: det.detectMarkers(g)
    p = a.DetectorParameters_create()            # OpenCV 4.6
    if inverted and hasattr(p, 'detectInvertedMarker'):
        p.detectInvertedMarker = True
    return lambda g: a.detectMarkers(g, dictionary, parameters=p)


def main():
    if len(sys.argv) < 2:
        print('usage: python3 identify_aruco.py <marker_image>')
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f'cannot read image: {sys.argv[1]}')
        sys.exit(1)
    # add a white border in case the printout/screenshot has no quiet zone
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = cv2.aruco

    hits = []
    for name in DICTS:
        try:
            d = a.getPredefinedDictionary(getattr(a, 'DICT_' + name))
        except Exception:
            continue
        for inverted in (False, True):
            try:
                corners, ids, _ = make_detector(d, inverted)(gray)
            except Exception:
                continue
            if ids is not None and len(ids) > 0:
                hits.append((name, inverted, [int(i) for i in ids.flatten()]))

    if not hits:
        print('No marker detected with any dictionary.')
        print('Tips: ensure good contrast, a white margin around the marker, and a flat photo.')
    else:
        print('=== Detected ===')
        for name, inverted, ids in hits:
            print(f'  dict={name}  inverted={inverted}  ids={ids}')
        print('\nSet mission_control params:')
        n, inv, _ = hits[0]
        print(f'  -p aruco_dict:={n} -p aruco_inverted:={str(inv).lower()}')


if __name__ == '__main__':
    main()
