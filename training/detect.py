"""학습한 YOLO 가중치로 추론 테스트 (이미지/영상/폴더).

차에 올리기 전, PC 에서 best.pt 가 4객체를 잘 잡는지 확인하는 용도.
실제 차량 추론 노드(ROS)는 이 로직을 그대로 옮겨 쓰면 된다.

사용 예:
    python detect.py --weights runs/detect/train/weights/best.pt --source test.jpg
    python detect.py --weights best.pt --source test.mp4 --conf 0.4 --show
"""

import argparse

from ultralytics import YOLO

CLASS_NAMES = ['red_light', 'green_light', 'left_sign', 'right_sign']


def main():
    parser = argparse.ArgumentParser(description='YOLO 추론 테스트')
    parser.add_argument('--weights', required=True, help='학습된 가중치 (best.pt)')
    parser.add_argument('--source', required=True, help='이미지/영상/폴더 경로')
    parser.add_argument('--conf', type=float, default=0.4, help='신뢰도 임계값')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--show', action='store_true', help='결과 창 띄우기')
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        show=args.show,
    )

    # 탐지된 클래스 요약 출력
    for r in results:
        names = r.names
        detected = [names[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
        print(f"{r.path}: {detected if detected else '탐지 없음'}")


if __name__ == '__main__':
    main()
