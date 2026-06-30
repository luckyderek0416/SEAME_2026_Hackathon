"""HSV 색상 필터링 — 흰 선과 노란(주황) 선을 색으로 분리.

회색조로는 흰선/노란선이 둘 다 밝아서 구분 안 됨 -> HSV 로 색 기준 분리.
  - 흰색: 채도(S) 낮고, 명도(V) 높음  (색이 거의 없음)
  - 노랑: 색상(H) ~노랑대역, 채도(S) 높음

트랙바로 영상 보며 두 색 범위를 맞춘 뒤, 그 값을 라인 추종에 쓴다.
'w' = 흰색 튜닝 / 'y' = 노랑 튜닝 / 'p' 누르면 현재 HSV 값 출력 / 'q' 종료.

사용 예:
    python hsv_filter.py --source track.mp4
"""

import argparse

import cv2
import numpy as np

# 기본 HSV 범위 (트랙 보며 조절) — [H, S, V]
RANGES = {
    'white':  {'low': [0, 0, 180],  'high': [180, 40, 255]},
    'yellow': {'low': [10, 80, 80], 'high': [35, 255, 255]},
}


def mask_for(hsv, color):
    r = RANGES[color]
    return cv2.inRange(hsv, np.array(r['low']), np.array(r['high']))


def run(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    is_image = not source.isdigit() and source.lower().endswith(
        ('.jpg', '.jpeg', '.png', '.bmp'))

    win = 'HSV filter  (w=흰색 y=노랑 p=값출력 q=종료)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    target = 'white'   # 지금 트랙바가 조절하는 색

    def load_trackbars(color):
        r = RANGES[color]
        for i, ch in enumerate('HSV'):
            cv2.setTrackbarPos(f'{ch}min', win, r['low'][i])
            cv2.setTrackbarPos(f'{ch}max', win, r['high'][i])

    for ch, mx in zip('HSV', (180, 255, 255)):
        cv2.createTrackbar(f'{ch}min', win, 0, mx, lambda v: None)
        cv2.createTrackbar(f'{ch}max', win, mx, mx, lambda v: None)
    load_trackbars(target)

    frame = None
    while True:
        if not is_image or frame is None:
            ok, frame = cap.read()
            if not ok:
                if is_image:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # 트랙바 값 -> 현재 target 색 범위에 반영
        for i, ch in enumerate('HSV'):
            RANGES[target]['low'][i] = cv2.getTrackbarPos(f'{ch}min', win)
            RANGES[target]['high'][i] = cv2.getTrackbarPos(f'{ch}max', win)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = mask_for(hsv, 'white')
        yellow = mask_for(hsv, 'yellow')

        # 시각화: 원본 | 흰 마스크 | 노란 마스크 (가로로)
        def lbl(m, t):
            b = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            cv2.putText(b, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return b
        h = 360
        scale = h / frame.shape[0]
        sz = (int(frame.shape[1] * scale), h)
        combo = np.hstack([
            cv2.resize(frame, sz),
            cv2.resize(lbl(white, 'WHITE'), sz),
            cv2.resize(lbl(yellow, 'YELLOW'), sz),
        ])
        cv2.putText(combo, f'editing: {target.upper()}', (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, combo)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('w'):
            target = 'white'; load_trackbars(target)
        elif k == ord('y'):
            target = 'yellow'; load_trackbars(target)
        elif k == ord('p'):
            print(f"\n[{target}]  low={RANGES[target]['low']}  high={RANGES[target]['high']}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='영상/이미지 또는 카메라(0)')
    run(ap.parse_args().source)


if __name__ == '__main__':
    main()
