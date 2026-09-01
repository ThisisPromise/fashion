# Garment segmentation and artwork placement

## What this is

A garment preview tool places artwork on garment photos. The old version
used fixed pixel coordinates for things like "left chest", so it broke
whenever the garment was a different size, pose, or crop. Artwork drifted,
landed on sleeves, or crossed seams.

This project fixes that in two parts.

1. A small model looks at a garment photo and finds the body panel and the
   sleeves.
2. A placement function uses those regions to work out where "left chest"
   or "centred" actually is on that specific photo, then places the
   artwork there. If it cannot find a good spot, it refuses instead of
   guessing.

Current model: mean IoU 0.693 (background 0.959, body 0.647, sleeve 0.473)
on the validation set. Details, splits, and known failure cases are in
`NOTE.md`.

## Getting started

Install dependencies:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Try the placement engine on your own photo, using the checkpoint already in
`outputs/checkpoint.pt`:

```
.venv\Scripts\python.exe -m src.demo --garment path\to\photo.jpg --artwork artwork\badge_circle.png --instruction "front, left chest"
```

Instruction keywords: `left`, `right`, `centre`; `chest` or `breast`;
`front` or `back`.

This needs the pose model at `models/pose_landmarker_lite.task`
(MediaPipe PoseLandmarker Lite, Apache 2.0). If it is missing:

```
curl -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Re-running the evaluation cases

```
.venv\Scripts\python.exe -m src.run_evaluation
```

Runs every case in `evaluation_cases.csv` and renders the result to
`outputs/evaluation_renders/`. Already-generated results are there too.

## Re-training from scratch

Download Fashionpedia's validation set and build the training subset:

```
curl -o data/fashionpedia/instances_attributes_val2020.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
curl -o data/fashionpedia/val_test2020.zip https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
```

Extract the zip to `data/fashionpedia/images_raw/`, then:

```
.venv\Scripts\python.exe -m src.prepare_data
.venv\Scripts\python.exe -m src.train --config configs/train_config.json
```

Training writes a checkpoint, per-class IoU, and training curves to
`outputs/`. To compare predictions against ground truth afterward:

```
.venv\Scripts\python.exe -m src.visualize_predictions
```

## Files

```
src/prepare_data.py            builds the dataset and split manifest
src/model.py                   segmentation model
src/dataset.py                 data loading and augmentation
src/train.py                   training loop
src/placement.py               placement logic
src/pose_utils.py              pose detection, used as a fallback
src/viz.py                     shared rendering helpers
src/demo.py                    try placement on one image
src/run_evaluation.py          runs evaluation_cases.csv
src/visualize_predictions.py   compare predictions to ground truth

evaluation_cases.csv           test cases: garment, artwork, instruction
outputs/checkpoint.pt          trained model
outputs/val_metrics.json       per-class IoU
outputs/training_curves.png    loss and IoU over training
outputs/evaluation_renders/    rendered result for each evaluation case
NOTE.md                        write-up: splits, metric, failure cases
```
