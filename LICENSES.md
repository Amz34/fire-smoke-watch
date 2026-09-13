# Licensing map

Three different things live in this repository and they do **not** share one
licence. Read this before you reuse any part of it commercially.

## 1. Detection and tooling code — Apache-2.0

```
firewatch/          examples/          bench/          LICENSE
```

Written for this project. Apache-2.0: use it, modify it, ship it, keep your
changes closed, no obligation beyond attribution and the patent grant.

There is no deep-learning framework in this path — only ONNX Runtime, OpenCV and
NumPy — so nothing here drags a copyleft obligation into your serving code.

## 2. Training notebook — AGPL-3.0

```
kaggle/train_firewatch.ipynb
```

The notebook trains with **Ultralytics**, which is AGPL-3.0. Ultralytics' stated
position ([ultralytics.com/license](https://www.ultralytics.com/license)) is that
the licence covers not only their training code, but also the models produced
with it — including models trained on your own data, in closed products, in SaaS
and embedded in hardware.

Practical consequences:

* Publishing the notebook is fine under AGPL-3.0, which is how it is offered here.
* **Weights trained with it are covered by AGPL-3.0 as well.** If you want to ship
  a closed-source product or a paid service built on those weights, take one of
  two routes:
  1. buy an Ultralytics Enterprise Licence, or
  2. retrain on a permissively licensed detector stack (for example RT-DETR,
     YOLOX, NanoDet, or the torchvision detectors) — the dataset below is
     licence-clean, so the retrain carries no third-party claim.

Note that AGPL-3.0 does not forbid selling: you may charge for hosting,
installation or support. What it forbids is serving a modified version to users
without offering them the source.

## 3. Weights — not distributed here

No `.pt` or `.onnx` file is included in this repository.

Reason: the checkpoint used for the numbers in `bench/RESULTS.md` came from a
public source that carries **no licence grant** at all, so there is no right to
redistribute it. Rather than ship a file we cannot license, this repository stops
at the code and lets you supply your own export.

Export your own with:

```bash
# from the training notebook (AGPL-3.0 route)
yolo export model=best.pt format=onnx imgsz=480
```

or in whatever way your detector of choice provides an ONNX graph that matches
the interface documented in the README.

## 4. Dataset

The training recipe in `kaggle/train_firewatch.ipynb` targets the **D-Fire**
dataset (fire and smoke, YOLO format), released under **CC0-1.0** — public
domain, no attribution required, safe to train on and to ship models from.

Always re-check the licence of whichever dataset you point the notebook at.

## 5. Demo media

`demo/firewatch_demo_9x16.mp4` and `demo/console.png` show the detector running
in the author's own console. The footage inside them is a public hazard clip,
cropped to exclude people and to hide the operator's alert inbox address. Reuse
the stills and the clip freely alongside this project.
