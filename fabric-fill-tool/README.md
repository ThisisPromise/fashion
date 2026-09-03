# Swatch Table

An interactive tool for applying fabric prints to garment images. Upload a
sketch, technical flat, or real photo of a garment; the app automatically
segments it into regions (sleeves, torso panels, collar, pockets, ...);
click a region (or several) and drag a fabric swatch onto it to fill it.

## What it does

- **Sketches and technical flats** are segmented with classical image
  processing (no model): threshold the linework, close small gaps, and
  flood-fill each enclosed shape into its own region.
- **Real photos** are segmented with two trained neural network
  checkpoints -- one for upper-body garments, one for lower-body garments
  -- run automatically together on every photo and merged into one region
  set, so a photo showing a top, trousers, or both just works.
- Regions are independently selectable and fillable. Click to select one,
  click several to select a group, and drop a fabric swatch on any of them
  to fill the whole selection at once.

## Setup

From the repository root:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r fabric-fill-tool/requirements.txt
```

This tool depends on two files from the sibling `submission/` folder in
this repository, which must be present: `submission/src/model.py` (the
segmentation model architecture) and
`submission/models/pose_landmarker_lite.task` (the pose detection model).
Both are already included in this repository.

## Running it

```
cd fabric-fill-tool/src
python server.py
```

Then open `http://127.0.0.1:5000/` in a browser. Everything runs locally;
no external services are called.

## Project layout

```
src/
  server.py                  Flask app: routes, image upload handling
  flats_segmentation.py      classical CV pipeline for sketches/flats
  photo_segmentation.py      neural segmentation pipeline for real photos
  pose_utils_extended.py     pose detection, used to sanity-check photo regions
  dataset_extended.py        dataset loader (training only)
  prepare_data_extended.py   builds training data from Fashionpedia (training only)
  train_extended.py          training entrypoint (training only)
static/
  index.html                 frontend: upload/drag-and-drop, region selection, fabric fill
configs/
  upperbody_extended.json    training config for the upper-body model
  lowerbody.json             training config for the lower-body model
outputs/
  upperbody_extended/        trained checkpoint + metrics for the upper-body model
  lowerbody/                 trained checkpoint + metrics for the lower-body model
assets/
  flats/                     sample sketch/flat images, served by the two flat presets
  fabrics/                   reference copies of the two default fabric swatches
                              (the running app embeds these directly in static/index.html)
```

## Model quality

Both models are a frozen MobileNetV3-Small backbone with a small trainable
decoder (the same architecture as `submission/src/model.py`), trained on a
subset of Fashionpedia's val2020 split. Current validation mean IoU:

**Upper-body** (background, body, neckline, collar, pocket, zipper, sleeve)

| class | IoU |
|---|---|
| background | 0.960 |
| body | 0.634 |
| sleeve | 0.490 |
| neckline | 0.159 |
| collar | 0.123 |
| pocket | 0.038 |
| zipper | 0.024 |
| **mean** | **0.347** |

**Lower-body** (background, body, pocket, zipper)

| class | IoU |
|---|---|
| background | 0.964 |
| body | 0.578 |
| pocket | 0.124 |
| zipper | 0.115 |
| **mean** | **0.445** |

Both models are well under the 2,000,000 trainable-parameter budget
(526,343 and 525,956 respectively).

Background, body, and sleeve/pants are reliable. Neckline, collar,
pocket, and zipper are weaker -- these classes cover a small fraction of
pixels in a minority of training images, which is a data limitation, not a
training bug. Retraining on a larger sample of Fashionpedia's data would
be the direct way to improve them further.

## Known limitations

- Pocket and zipper region detection is unreliable given how little
  training data represents them.
- A garment occluding another (e.g. a long coat over trousers) can
  produce predictions that bleed past the true boundary between them;
  pose-based clipping reduces but does not eliminate this.
- Pose detection (used to sanity-check photo regions) requires a
  confidently detected person with both shoulders and hips in frame;
  a tightly cropped photo falls back to un-corrected geometric splitting.
- Flat/sketch segmentation assumes clean, closed linework. Repeated
  parallel lines (e.g. pleats) can fragment into multiple thin regions
  instead of one.

## Retraining

To rebuild the training datasets and retrain either model, first download
Fashionpedia's val2020 data per `submission/README.md`, then from the
repository root:

```
python fabric-fill-tool/src/prepare_data_extended.py
python fabric-fill-tool/src/train_extended.py --config fabric-fill-tool/configs/upperbody_extended.json
python fabric-fill-tool/src/train_extended.py --config fabric-fill-tool/configs/lowerbody.json
```
