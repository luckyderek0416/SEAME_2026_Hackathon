"""yolo_ncnn_node: YOLOv8 object detection via NCNN (ARM-friendly, no PyTorch).

Drop-in replacement for yolo_node that runs the NCNN-exported model
(best_ncnn_model/model.ncnn.param + .bin) instead of ultralytics .pt.
Publishes the SAME inference_msgs/Detections, so decision_node is unchanged.

Why NCNN: on the D3-G (ARM CPU) it is far faster and lighter than
ultralytics+PyTorch, with no heavy dependency (pip install ncnn).

Model output (this export): out0 = [4+nc, 2100]
  rows 0..3  = box cx,cy,w,h in letterboxed 320px space
  rows 4..   = per-class scores (already sigmoid-applied in the graph)

Decode lives in `decode_yolov8()` so it can be unit-tested without ncnn.
"""
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from inference_msgs.msg import Detection, Detections


def letterbox(img, new=320, color=(114, 114, 114)):
    """Resize keeping aspect ratio + pad to new x new. Returns (out, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(new / w, new / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new, new, 3), color, dtype=np.uint8)
    px, py = (new - nw) // 2, (new - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, px, py


def decode_yolov8(out, num_classes, img_w, img_h, imgsz, scale, pad_x, pad_y,
                  conf_thresh, nms_thresh):
    """Decode raw NCNN YOLOv8 output -> list of dicts (label_id, conf, normalized xywh).

    out: np.ndarray, either (4+nc, N) or (N, 4+nc). Boxes in letterboxed imgsz px.
    Returns detections with x_center/y_center/width/height normalised to [0,1]
    of the ORIGINAL image.
    """
    feat = 4 + num_classes
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(feat, -1)
    if arr.shape[0] != feat and arr.shape[1] == feat:   # (N, feat) -> (feat, N)
        arr = arr.T
    if arr.shape[0] != feat:
        return []

    boxes_xywh = arr[:4, :]               # cx,cy,w,h (letterboxed px)
    cls_scores = arr[4:4 + num_classes, :]
    class_ids = np.argmax(cls_scores, axis=0)
    confs = cls_scores[class_ids, np.arange(cls_scores.shape[1])]

    keep = confs >= conf_thresh
    if not np.any(keep):
        return []
    boxes_xywh = boxes_xywh[:, keep]
    class_ids = class_ids[keep]
    confs = confs[keep]

    # letterboxed center-xywh -> original-image top-left xywh
    cx, cy, bw, bh = boxes_xywh
    x1 = (cx - bw / 2.0 - pad_x) / scale
    y1 = (cy - bh / 2.0 - pad_y) / scale
    ww = bw / scale
    hh = bh / scale

    rects = np.stack([x1, y1, ww, hh], axis=1).tolist()
    scores = confs.tolist()
    idxs = cv2.dnn.NMSBoxes(rects, scores, conf_thresh, nms_thresh)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()

    dets = []
    for i in idxs:
        bx, by, bwid, bhei = rects[i]
        xc = (bx + bwid / 2.0) / img_w
        yc = (by + bhei / 2.0) / img_h
        dets.append({
            'class_id': int(class_ids[i]),
            'confidence': float(scores[i]),
            'x_center': float(min(1.0, max(0.0, xc))),
            'y_center': float(min(1.0, max(0.0, yc))),
            'width': float(min(1.0, max(0.0, bwid / img_w))),
            'height': float(min(1.0, max(0.0, bhei / img_h))),
        })
    return dets


class YoloNcnnNode(Node):
    def __init__(self):
        super().__init__('yolo_ncnn_node')
        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        self.declare_parameter('detections_topic', '/inference/detections')
        self.declare_parameter('model_param', '/home/topst/D-Racer/models/model.ncnn.param')
        self.declare_parameter('model_bin', '/home/topst/D-Racer/models/model.ncnn.bin')
        self.declare_parameter('class_names', ['red_light', 'green_light', 'left_sign', 'right_sign'])
        self.declare_parameter('input_name', 'in0')
        self.declare_parameter('output_name', 'out0')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('infer_hz', 10.0)
        self.declare_parameter('num_threads', 4)

        g = self.get_parameter
        self.sub_topic = str(g('subscribe_topic').value)
        self.det_topic = str(g('detections_topic').value)
        self.model_param = str(g('model_param').value)
        self.model_bin = str(g('model_bin').value)
        self.class_names = [str(x) for x in g('class_names').value]
        self.input_name = str(g('input_name').value)
        self.output_name = str(g('output_name').value)
        self.conf = float(g('conf_threshold').value)
        self.nms = float(g('nms_threshold').value)
        self.imgsz = int(g('imgsz').value)
        self.num_threads = int(g('num_threads').value)
        infer_hz = float(g('infer_hz').value)

        self.net = self._load_net()
        self.latest = None

        self.pub = self.create_publisher(Detections, self.det_topic, 10)
        self.create_subscription(CompressedImage, self.sub_topic, self.on_image, 10)
        self.create_timer(1.0 / max(infer_hz, 1.0), self.on_timer)
        self.get_logger().info(
            f'yolo_ncnn_node up. param={self.model_param} classes={self.class_names}')

    def _load_net(self):
        try:
            import ncnn
            net = ncnn.Net()
            net.opt.num_threads = self.num_threads
            if net.load_param(self.model_param) != 0:
                raise RuntimeError(f'load_param failed: {self.model_param}')
            if net.load_model(self.model_bin) != 0:
                raise RuntimeError(f'load_model failed: {self.model_bin}')
            self._ncnn = ncnn
            self.get_logger().info('NCNN model loaded.')
            return net
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'Could not load NCNN model ({exc}). Publishing EMPTY detections. '
                'Install with "pip install ncnn" and point model_param/model_bin to the export.')
            self._ncnn = None
            return None

    def on_image(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self.latest = (frame, msg.header)

    def on_timer(self):
        if self.latest is None:
            return
        frame, header = self.latest
        out = Detections()
        out.header = header
        if self.net is not None:
            try:
                for d in self._infer(frame):
                    det = Detection()
                    cid = d['class_id']
                    det.label = self.class_names[cid] if cid < len(self.class_names) else str(cid)
                    det.class_id = cid
                    det.confidence = d['confidence']
                    det.x_center = d['x_center']
                    det.y_center = d['y_center']
                    det.width = d['width']
                    det.height = d['height']
                    out.detections.append(det)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f'inference failed: {exc}')
        self.pub.publish(out)

    def _infer(self, frame):
        h, w = frame.shape[:2]
        lb, scale, px, py = letterbox(frame, self.imgsz)
        mat_in = self._ncnn.Mat.from_pixels(
            lb, self._ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.imgsz, self.imgsz)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat_in)
        ret, mat_out = ex.extract(self.output_name)
        out_np = np.array(mat_out)
        return decode_yolov8(out_np, len(self.class_names), w, h, self.imgsz,
                             scale, px, py, self.conf, self.nms)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNcnnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
