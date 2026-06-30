"""주최측 제공 영상을 프레임(.jpg)으로 쪼개는 스크립트.

라벨링 전 단계: 영상 → 일정 간격으로 프레임 추출 → frames/ 에 저장.
추출한 프레임을 Roboflow/labelImg 등으로 4클래스 박스 라벨링한 뒤
dataset/ 구조로 정리한다.

사용 예:
    python extract_frames.py --video red_light.mp4 --out frames --every 5
    python extract_frames.py --video videos/ --out frames --every 5   # 폴더 통째로
"""

import argparse
import os

import cv2


def iter_videos(path):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                yield os.path.join(path, name)
    else:
        yield path


def extract(video_path, out_dir, every):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[건너뜀] 열 수 없음: {video_path}")
        return 0

    stem = os.path.splitext(os.path.basename(video_path))[0]
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            fname = f"{stem}_{idx:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"  {video_path}: {saved}장 추출")
    return saved


def main():
    parser = argparse.ArgumentParser(description='영상 → 프레임 추출')
    parser.add_argument('--video', required=True, help='영상 파일 또는 폴더')
    parser.add_argument('--out', default='frames', help='출력 폴더 (기본: frames)')
    parser.add_argument('--every', type=int, default=5,
                        help='N프레임마다 1장 저장 (기본: 5)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for v in iter_videos(args.video):
        total += extract(v, args.out, args.every)
    print(f"완료: 총 {total}장 → {args.out}/")


if __name__ == '__main__':
    main()
