"""YOLO26n 학습 스크립트 (ultralytics).

라벨링까지 끝난 dataset/ 과 data.yaml 로 YOLO 를 학습시켜
best.pt(가중치)를 만든다. 학습은 PC(GPU)에서 수행.

사용 예:
    python train.py                                   # 기본값
    python train.py --model yolo26n.pt --epochs 100 --imgsz 320 --batch 16

엣지(D3-G) 배포를 고려해 입력 크기는 작게(320) 두는 것을 추천.
yolo26n.pt 가 없으면 yolo11n.pt / yolov8n.pt 로 바꿔도 된다(모두 nano).
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='YOLO 학습')
    parser.add_argument('--model', default='yolo26n.pt',
                        help='사전학습 가중치 (기본: yolo26n.pt)')
    parser.add_argument('--data', default='data.yaml', help='데이터 설정 yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=320, help='입력 크기 (엣지용 작게)')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    # 결과(best.pt)는 runs/detect/train/weights/best.pt 에 생성된다.
    print("학습 완료 → runs/detect/train/weights/best.pt")


if __name__ == '__main__':
    main()
