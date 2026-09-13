#!/usr/bin/env python3
"""Run the fire/smoke detector on a single image.

    python examples/detect_image.py --model m.onnx --image street.jpg --out out.jpg

Writes the annotated image and, optionally, a JSON report. Prints what it found
plus the inference time, so it doubles as a smoke test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from firewatch.detector import Detector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a YOLO-style .onnx model")
    ap.add_argument("--image", required=True, help="image to inspect")
    ap.add_argument("--out", help="write the annotated image here")
    ap.add_argument("--json", help="write detections as JSON here")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="inference size (default: taken from the file name, else 480)")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--threads", type=int, default=2, help="ONNX Runtime threads")
    ap.add_argument("--no-clahe", action="store_true",
                    help="disable the local-contrast pre-pass")
    args = ap.parse_args()

    imgsz = args.imgsz or _guess_imgsz(args.model)
    det = Detector(args.model, imgsz=imgsz, conf=args.conf, iou=args.iou,
                   threads=args.threads, use_clahe=not args.no_clahe)

    img = cv2.imread(args.image)
    if img is None:
        print(f"could not read {args.image}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    dets = det.detect(img)
    ms = (time.perf_counter() - t0) * 1000

    counts: dict[str, int] = {}
    for d in dets:
        counts[d.name] = counts.get(d.name, 0) + 1

    print(f"{Path(args.image).name}: {img.shape[1]}x{img.shape[0]}  "
          f"model={Path(args.model).name} imgsz={imgsz}  {ms:.0f} ms")
    print(f"  {len(dets)} detection(s) {counts or ''}")
    for d in dets:
        print(f"    {d.name:<6} {d.conf:.2f}  {d.xyxy}")

    if args.out:
        cv2.imwrite(args.out, Detector.annotate(img, dets))
        print("  annotated ->", args.out)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"image": args.image, "imgsz": imgsz, "latency_ms": round(ms, 1),
             "detections": [d.to_dict() for d in dets]}, indent=2))
        print("  json ->", args.json)
    return 0


def _guess_imgsz(model: str) -> int:
    """`best_480.onnx` -> 480, otherwise the common default."""
    import re
    m = re.search(r"(\d{3})", Path(model).stem)
    return int(m.group(1)) if m else 480


if __name__ == "__main__":
    raise SystemExit(main())
