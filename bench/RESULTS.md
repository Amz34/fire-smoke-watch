# Measured performance

Harness: [`bench_onnx.py`](bench_onnx.py) · raw output: [`results.json`](results.json)

**Machine:** Oracle Cloud Always Free (Ampere A1), `aarch64`, 2 vCPU of
Neoverse-N1, no GPU. ONNX Runtime `CPUExecutionProvider`, 2 intra-op threads,
local-contrast (CLAHE) pre-pass enabled, 3 warm-up runs discarded, 20 timed runs
per model. Frame: a 640x480 photograph containing real smoke and flame.

| model | mean | p50 | p95 | throughput |
|---|---|---|---|---|
| `best_320.onnx` | 47.0 ms | 45.1 ms | 51.0 ms | 21.3 FPS |
| `best_480.onnx` | 97.3 ms | 94.5 ms | 114.0 ms | 10.3 FPS |
| `best_640.onnx` | 166.4 ms | 156.7 ms | 187.6 ms | 6.0 FPS |

Both models returned the same 2 detections (1 smoke, 1 fire) on this frame, so
the smaller sizes are not paying for accuracy *on this particular frame* — that
is a latency comparison, not an accuracy benchmark. Accuracy has to be measured
on a held-out dataset (see the training notebook).

Reproduce:

```bash
python bench/bench_onnx.py --pattern "best_{imgsz}.onnx" --sizes 320,480,640 \
    --image frame.jpg --runs 20 --json bench/results.json
```

## Reading the numbers

- **Sampling headroom.** A watch loop that inspects one frame per second costs
  ~5–10% of one core at 480px on this machine, so the box can watch several
  cameras at 1 FPS without a GPU. A 2-core box running the model at 480px has
  room for roughly 3–4 concurrent 1 FPS streams before it is CPU-bound.
- **p95, not mean, decides the design.** The slow tail (114 ms vs a 94 ms
  median at 480px) is what a cooldown window has to absorb.
- **Model size is a knob, not a constant.** 320px is ~2x faster than 640px; ride
  the smallest input size that still detects the hazard sizes you care about.
- **These numbers are CPU-only and single-process.** Hard acceleration (x86 AVX2,
  a GPU, or an edge NPU) shifts them substantially — rerun the harness on your
  target hardware instead of trusting this table.

## What is *not* measured here

- Accuracy on a dataset (only latency on one frame).
- Real camera pipelines: decode, resize, network transfer and alert delivery all
  sit outside this harness.
- Night / IR footage, which behaves differently from the daylight frame used.
