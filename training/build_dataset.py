"""표지판 영상 -> YOLO 데이터셋 자동 생성 (프레임 추출 + 자동 라벨링 + 분할).

각 영상은 통째로 한 클래스(좌/우)이고, 표지판이 선명한 파란 원이라
OpenCV로 "파란 원"을 찾아 박스를 자동 생성한다(손 라벨링 불필요).
배경의 파란 모니터 화면은 '원형도(circularity)' 필터로 걸러낸다.
확실한 원형 표지판이 보이는 프레임만 채택하고 나머지는 버린다.

출력 (data.yaml 과 일치):
    dataset/
      images/train/*.jpg   labels/train/*.txt
      images/val/*.jpg     labels/val/*.txt

클래스 id 는 data.yaml 기준: left_sign=2, right_sign=3
(신호등 0,1 은 나중에 추가 — id 순서 유지)

사용 예:
    python build_dataset.py --src ~/Downloads --every 10 --val 0.2
"""

import argparse
import os

import cv2
import numpy as np

# 영상 파일명 -> 클래스 id (data.yaml: left_sign=2, right_sign=3)
# 새 영상이 생기면 여기에 추가.
VIDEO_CLASS = {
    'TalkFile_WIN_20260617_14_29_20_Pro.mp4.mp4': 3,  # right_sign
    'TalkFile_WIN_20260617_14_30_35_Pro.mp4.mp4': 3,  # right_sign
    'TalkFile_WIN_20260617_14_34_30_Pro.mp4.mp4': 2,  # left_sign
    'TalkFile_WIN_20260617_14_35_28_Pro.mp4.mp4': 2,  # left_sign
}


def detect_sign(img):
    """파란 원형 표지판의 (x,y,w,h) 반환. 확실치 않으면 None (그 프레임은 버림)."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 110, 40), (135, 255, 255))  # 진한 파랑만
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.03 * H * W:           # 너무 작으면 무시
            continue
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        circ = 4 * np.pi * area / (per * per)        # 1.0 = 완전한 원
        (_, _), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r + 1e-6)         # 원을 얼마나 꽉 채우나
        score = circ * fill
        if circ > 0.6 and fill > 0.6 and score > best_score:  # 화면(사각형) 제외
            best_score, best = score, cv2.boundingRect(c)
    return best


def to_yolo_line(cls, box, W, H):
    x, y, w, h = box
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    return f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}\n"


def main():
    parser = argparse.ArgumentParser(description='표지판 영상 -> YOLO 데이터셋')
    parser.add_argument('--src', default=os.path.expanduser('~/Downloads'),
                        help='영상이 있는 폴더')
    parser.add_argument('--out', default='dataset', help='출력 데이터셋 폴더')
    parser.add_argument('--every', type=int, default=10, help='N프레임마다 검사')
    parser.add_argument('--val', type=float, default=0.2, help='검증 비율')
    args = parser.parse_args()

    for split in ('train', 'val'):
        os.makedirs(os.path.join(args.out, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.out, 'labels', split), exist_ok=True)

    counts = {2: 0, 3: 0}
    idx = 0
    for vid, cls in VIDEO_CLASS.items():
        path = os.path.join(args.src, vid)
        if not os.path.exists(path):
            print(f"[건너뜀] 없음: {path}")
            continue
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        acc = 0
        for i in range(0, n, args.every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            box = detect_sign(frame)
            if box is None:
                continue
            H, W = frame.shape[:2]
            # val/train 분할 (idx 기반으로 결정적)
            split = 'val' if (idx % int(1 / args.val) == 0) else 'train'
            stem = f"{vid[18:27]}_{i:06d}"
            cv2.imwrite(os.path.join(args.out, 'images', split, stem + '.jpg'), frame)
            with open(os.path.join(args.out, 'labels', split, stem + '.txt'), 'w') as f:
                f.write(to_yolo_line(cls, box, W, H))
            counts[cls] += 1
            acc += 1
            idx += 1
        cap.release()
        name = 'left_sign' if cls == 2 else 'right_sign'
        print(f"{vid[8:30]} [{name}]: {acc}장 채택")

    print(f"\n완료 -> {args.out}/")
    print(f"  left_sign(2): {counts[2]}장   right_sign(3): {counts[3]}장")
    print(f"  총 {counts[2] + counts[3]}장")


if __name__ == '__main__':
    main()
