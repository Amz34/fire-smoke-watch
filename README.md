# Fire & Smoke Watch — detector quickstart

Fire and smoke detection you can run on a CPU, in a browser tab or on a small
edge box: **ONNX Runtime + OpenCV + NumPy**, no deep-learning framework in the
serving path.

This repository is the detection layer and the training recipe. It is the part
of the system that answers one question as cheaply as possible:

> *Is there visible fire or smoke in this frame, and where?*

## What this is — and what it is not

**It is** a supplementary early-warning detector. Point a camera at an area and
it will flag visible flames or smoke and report the bounding boxes.

**It is not** a certified fire alarm system. It is not a replacement for
code-compliant detection and alarm infrastructure (NFPA 72, EN 54, or the local
civil-defence requirements that apply to you), and it must not be the only means
of protection in any building. Never remove, bypass or downgrade an approved
alarm system because this detector exists.

Accuracy depends on the camera, the distance, the lighting, and what else is in
the air. Steam, dust, welding sparks and sunsets can be misread as smoke or fire.
Treat a detection as *"a human should look at this right now"*, not as a verdict.

## Contents

| Path | What it does |
|---|---|
| `firewatch/detector.py` | Letterbox, ONNX Runtime inference, class-aware NMS, optional CLAHE pre-pass, annotation helper |
| `examples/detect_image.py` | Run the detector on an image, write the annotated copy plus JSON |
| `bench/bench_onnx.py` | Latency / throughput harness across input sizes |
| `bench/RESULTS.md` | Numbers measured on a 2-vCPU ARM server |
| `kaggle/train_firewatch.ipynb` | End-to-end training recipe (see `LICENSES.md` first) |
| `demo/` | 9:16 demo clip and a console screenshot. The hazard imagery is a public fire/smoke still used to exercise the model — swap in your own footage before publishing anything. |

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python examples/detect_image.py \
    --model /path/to/model.onnx \
    --image street.jpg \
    --out annotated.jpg --json detections.json
```

### Output

`--json` writes the envelope below (real run, 640x480 photo, `imgsz=480`). Boxes are
`[x1, y1, x2, y2]` pixels in the **original** image, so they can be drawn directly.

```json
{
  "image": "street.jpg",
  "imgsz": 480,
  "latency_ms": 229.4,
  "detections": [
    {"cls_id": 0, "name": "Smoke", "conf": 0.793, "xyxy": [247, 4, 640, 293]},
    {"cls_id": 1, "name": "Fire",  "conf": 0.531, "xyxy": [320, 201, 349, 215]}
  ]
}
```

## Model interface

Any checkpoint exported to this contract works; `--model` is the only coupling.

* **Input** — `float32`, shape `1x3xSxS` (`S` = 320/480/640), RGB, scaled to
  `0..1`, letterboxed to a square with grey `114` padding.
* **Output** — `(1, 4+nc, N)`: `cxcywh` in pixels of the letterboxed image,
  followed by one confidence per class.
* **Classes** — `0 = Smoke`, `1 = Fire`.

## Weights

No weights ship with this repository, and that is deliberate — see
[`LICENSES.md`](LICENSES.md). Supply your own export, or train one with the
notebook.

## Measured performance

Measurement is on a 2-vCPU ARM server (Neoverse-N1, no GPU) — i.e. a worst case,
not a best case. See `bench/RESULTS.md` for the full table.

## The rest of the system

The multi-camera console — phone/browser capture, confirming-photo logic, alert
cooldowns and e-mail alerts that carry an annotated evidence photo — is a
separate product and is not part of this repository.

## Licence

Detection code: **Apache-2.0**. Training notebook: **AGPL-3.0** (it depends on
Ultralytics). Full map and the reasoning: [`LICENSES.md`](LICENSES.md).

---

Part of [my always-on agent stack](https://github.com/Amz34) · [Awesome Agent Infrastructure](https://github.com/Amz34/awesome-agent-infrastructure) (135 live-checked building blocks).
