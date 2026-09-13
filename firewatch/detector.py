"""YOLO11 fire/smoke detector on ONNX Runtime (CPU) with optional CLAHE pre-pass.

Own implementation: letterbox + class-aware NMS, no Ultralytics runtime needed at
serving time (keeps AGPL code out of the serving path).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = {0: "Smoke", 1: "Fire"}


@dataclass
class Detection:
    cls_id: int
    name: str
    conf: float
    xyxy: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["conf"] = round(self.conf, 3)
        return d


def letterbox(img: np.ndarray, imgsz: int, pad_value: int = 114):
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class Detector:
    def __init__(self, model_path: str | Path, imgsz: int = 480, conf: float = 0.35,
                 iou: float = 0.45, threads: int = 2, use_clahe: bool = True):
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.use_clahe = use_clahe
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(self.model_path, opts,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ---------- preprocessing ----------
    def _enhance(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge((self._clahe.apply(l), a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def preprocess(self, bgr: np.ndarray):
        if self.use_clahe:
            bgr = self._enhance(bgr)
        lb, r, dx, dy = letterbox(bgr, self.imgsz)
        x = lb[:, :, ::-1].astype(np.float32) / 255.0        # BGR -> RGB, /255
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # NCHW
        return x, r, dx, dy

    # ---------- inference ----------
    def detect(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        x, r, dx, dy = self.preprocess(bgr)
        out = self.sess.run(None, {self.input_name: x})[0]   # (1, 4+nc, N)
        pred = np.asarray(out)[0].T                          # (N, 4+nc)
        if pred.size == 0:
            return []
        boxes_cxcywh = pred[:, :4]
        scores = pred[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(cls_ids)), cls_ids]

        keep = confs >= self.conf
        if not keep.any():
            return []
        boxes_cxcywh, cls_ids, confs = boxes_cxcywh[keep], cls_ids[keep], confs[keep]

        # cxcywh (letterbox space) -> xyxy (original image space)
        cx, cy, bw, bh = (boxes_cxcywh[:, i] for i in range(4))
        x1 = (cx - bw / 2 - dx) / r
        y1 = (cy - bh / 2 - dy) / r
        x2 = (cx + bw / 2 - dx) / r
        y2 = (cy + bh / 2 - dy) / r
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        # class-aware NMS (cv2 wants xywh)
        xywh = np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0],
                         xyxy[:, 3] - xyxy[:, 1]], axis=1)
        kept: list[int] = []
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            sub = cv2.dnn.NMSBoxes(xywh[idx].tolist(), confs[idx].tolist(),
                                   self.conf, self.iou)
            if len(sub):
                kept.extend(idx[np.asarray(sub).reshape(-1)].tolist())

        dets = [Detection(int(cls_ids[i]), CLASS_NAMES.get(int(cls_ids[i]), str(int(cls_ids[i]))),
                          float(confs[i]), tuple(int(v) for v in xyxy[i])) for i in sorted(kept)]
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets

    # ---------- rendering ----------
    @staticmethod
    def annotate(bgr: np.ndarray, dets: list[Detection]) -> np.ndarray:
        img = bgr.copy()
        h, w = img.shape[:2]
        thickness = max(1, int(round(max(h, w) / 400)))
        for d in dets:
            color = (60, 60, 235) if d.name == "Fire" else (200, 160, 40)   # BGR
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{d.name} {d.conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.6 * thickness, max(1, thickness - 1))
            cv2.rectangle(img, (x1, max(0, y1 - th - bl - 4)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, max(th, y1 - bl - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thickness, (255, 255, 255),
                        max(1, thickness - 1), cv2.LINE_AA)
        return img
