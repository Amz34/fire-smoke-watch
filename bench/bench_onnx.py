#!/usr/bin/env python3
"""Latency / throughput harness for the ONNX detector.

    python bench/bench_onnx.py --model best_480.onnx                 # single model
    python bench/bench_onnx.py --model best_480.onnx --image f.jpg
    python bench/bench_onnx.py --pattern "best_{imgsz}.onnx" --sizes 320,480,640

The inference size is read from the ONNX file itself (or from the file name as a
fallback), so a size can never be paired with the wrong model. The point of the
harness is to answer, for a specific machine, "how many frames per second can
this afford?" before wiring a detector into a camera loop. Warm-up runs are
discarded and p50/p95 are reported next to the mean because one slow frame is
what actually breaks an alert pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from firewatch.detector import Detector  # noqa: E402


def percentiles(samples: list[float]) -> tuple[float, float]:
    s = sorted(samples)
    return s[int(0.50 * (len(s) - 1))], s[int(0.95 * (len(s) - 1))]


def model_input_size(path: str) -> int | None:
    """Real input size from the ONNX graph; None if it cannot be determined."""
    try:
        import onnxruntime as ort
        shape = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).get_inputs()[0].shape
        dim = shape[-1]
        return int(dim) if isinstance(dim, int) else None
    except Exception:
        return None


def resolve(args) -> list[tuple[int, str]]:
    if args.pattern:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
        if not sizes:
            sys.exit("--pattern needs --sizes, e.g. --sizes 320,480,640")
        pairs = [(s, args.pattern.format(imgsz=s)) for s in sizes]
    elif args.model:
        pairs = [(0, args.model)]
    else:
        sys.exit("give --model (single file) or --pattern with --sizes")

    out = []
    for size, path in pairs:
        real = model_input_size(path)
        if real is None:
            m = re.search(r"(\d{3})", Path(path).stem)
            real = m and int(m.group(1))
        if real is None:
            sys.exit(f"cannot determine the input size of {path}; pass a file named e.g. best_480.onnx")
        if size and real != size:
            print(f"  note: {Path(path).name} is {real}px, ignoring requested {size}", file=sys.stderr)
        out.append((real, path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a single .onnx model file")
    ap.add_argument("--pattern", help='one model per size, e.g. "best_{imgsz}.onnx"')
    ap.add_argument("--sizes", default="", help="comma list used with --pattern")
    ap.add_argument("--image", help="image to use as the frame (else a synthetic one)")
    ap.add_argument("--runs", type=int, default=20, help="timed runs per model")
    ap.add_argument("--warmup", type=int, default=3, help="discarded warm-up runs")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--json", help="write the measurements as JSON here")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"could not read {args.image}")
        source = Path(args.image).name
    else:
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        cv2.circle(frame, (320, 240), 90, (90, 170, 255), -1)
        source = "synthetic 640x480"

    models = resolve(args)
    rows = []
    for size, path in models:
        if not Path(path).exists():
            sys.exit(f"missing model: {path}")
        det = Detector(path, imgsz=size, threads=args.threads,
                       use_clahe=not args.no_clahe)
        for _ in range(args.warmup):
            det.detect(frame)
        samples, ndet = [], 0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            ndet = len(det.detect(frame))
            samples.append((time.perf_counter() - t0) * 1000)
        mean = statistics.fmean(samples)
        p50, p95 = percentiles(samples)
        rows.append({"model": Path(path).name, "imgsz": size, "mean_ms": round(mean, 1),
                     "p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
                     "fps": round(1000 / mean, 2), "detections": ndet})
        print(f"  {Path(path).name:<18} mean {mean:6.1f} ms  p50 {p50:6.1f}  "
              f"p95 {p95:6.1f}   {1000 / mean:5.2f} FPS   ({ndet} detections)")

    print(f"\n  frame: {source}   runs/model: {args.runs}   threads: {args.threads}"
          f"   clahe: {not args.no_clahe}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"frame": source, "runs": args.runs, "threads": args.threads,
             "clahe": not args.no_clahe, "results": rows}, indent=2))
        print("  json ->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
