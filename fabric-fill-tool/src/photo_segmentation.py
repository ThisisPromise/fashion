"""Region segmentation for real garment photos.

Reuses trained TinyGarmentSegModel checkpoints -- no retraining logic here.
Generalized over whatever checkpoint is passed in, so the same code serves
both the original submission checkpoint (background/body/sleeve) and the
new extended one (background/body/neckline/collar/pocket/zipper/sleeve):
  - class 1 ("body") is one connected blob on a worn garment, so it's split
    into left/right halves by its own bounding-box centerline -- a rougher
    approximation than a real seam, worth being honest about.
  - every other non-background class (sleeve, collar, pocket, ...) gets
    connected-components + a minimum-area filter. Sleeves naturally come out
    as two separate blobs (left/right arm don't touch); a class like pocket
    or zipper comes out as however many blobs the model actually predicted,
    which may be zero on a given photo if that class wasn't detected there
    -- expected, not a bug, especially for pocket/zipper given how weak
    those classes are (see fabric-fill-tool/outputs/upperbody_extended/val_metrics.json).

Scope cut, deliberately: no pose-model flat-lay fallback yet (mediapipe
isn't installed in this environment). A flat-lay/no-person photo will get
whatever the segmentation model gives it, untrusted-or-not -- same
limitation the original submission's placement engine was built to guard
against, not yet reintroduced here.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(TOOL_DIR)
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")

ORIGINAL_CHECKPOINT = os.path.join(SUBMISSION_DIR, "outputs", "checkpoint.pt")
EXTENDED_CHECKPOINT = os.path.join(TOOL_DIR, "outputs", "upperbody_extended", "checkpoint.pt")
DEFAULT_CHECKPOINT = EXTENDED_CHECKPOINT

sys.path.insert(0, SUBMISSION_DIR)
from src.model import TinyGarmentSegModel  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

MIN_REGION_AREA_PX = 150  # smaller than the original 200: the new part classes are naturally tiny

_models = {}  # checkpoint_path -> (model, cfg), loaded once and cached


def _load_model(checkpoint_path):
    if checkpoint_path not in _models:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = TinyGarmentSegModel(num_classes=cfg["num_classes"], decoder_ch=cfg["decoder_channels"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _models[checkpoint_path] = (model, cfg)
    return _models[checkpoint_path]


@torch.no_grad()
def predict_labels(image_path, checkpoint_path=DEFAULT_CHECKPOINT):
    """Per-pixel class id at the photo's original resolution, plus the
    checkpoint's class_names. Same argmax-then-nearest-upsample path as
    submission/src/placement.py:predict_regions."""
    model, cfg = _load_model(checkpoint_path)
    garment_image = Image.open(image_path).convert("RGB")
    size = cfg["image_size"]
    orig_w, orig_h = garment_image.size

    resized = garment_image.resize((size, size), Image.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)

    logits = model(tensor)
    pred = logits.argmax(dim=1)[0].numpy()

    pred_img = Image.fromarray(pred.astype(np.uint8), mode="L").resize((orig_w, orig_h), Image.NEAREST)
    pred_full = np.array(pred_img)
    return pred_full, cfg["class_names"]


def _label_and_filter(mask, min_area=MIN_REGION_AREA_PX):
    labeled, n = ndi.label(mask, structure=np.ones((3, 3)))
    regions = []
    for label_id in range(1, n + 1):
        m = labeled == label_id
        if m.sum() >= min_area:
            regions.append(m)
    return regions


def _split_left_right(mask, min_area=MIN_REGION_AREA_PX):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    mid_x = (xs.min() + xs.max()) / 2
    col_idx = np.arange(mask.shape[1])[None, :]
    left = mask & (col_idx < mid_x)
    right = mask & (col_idx >= mid_x)
    return [m for m in (left, right) if m.sum() >= min_area]


# Classes that are anatomically two-sided (a garment's own left/right front
# panel, a collar's left/right point) get the same left/right centerline
# split as body -- one clean region per side, instead of exposing every
# small disconnected fragment the raw prediction happens to break into.
# Also collapses a lot of the "boundary is messy" complaint: several tiny
# scattered blobs become two coherent halves. Sleeve/pocket/zipper don't get
# this -- sleeves already separate naturally (left/right arm don't touch),
# and there's no similar "two sides" reading for a zipper or a single pocket.
BILATERAL_CLASS_NAMES = {"collar", "neckline"}


def extract_regions(image_path, checkpoint_path=DEFAULT_CHECKPOINT):
    """Returns {region_id: bool mask} -- same shape as
    flats_segmentation.extract_regions, so the frontend needs zero changes
    to handle either source. Works for any checkpoint: class 1 (body) is
    always split left/right; bilateral part classes (collar, neckline) get
    the same treatment; everything else is split by connected components."""
    pred_full, class_names = predict_labels(image_path, checkpoint_path)

    regions = {}
    region_id = 1

    body_mask = pred_full == 1
    for m in _split_left_right(body_mask):
        regions[region_id] = m
        region_id += 1

    for class_id in range(2, len(class_names)):
        class_mask = pred_full == class_id
        if class_names[class_id] in BILATERAL_CLASS_NAMES:
            pieces = _split_left_right(class_mask)
        else:
            pieces = _label_and_filter(class_mask)
        for m in pieces:
            regions[region_id] = m
            region_id += 1

    return regions


if __name__ == "__main__":
    import sys as _sys
    path = _sys.argv[1] if len(_sys.argv) > 1 else "fabric-fill-tool/sample_photos/photo1.jpg"
    ckpt = _sys.argv[2] if len(_sys.argv) > 2 else DEFAULT_CHECKPOINT
    regions = extract_regions(path, ckpt)
    _, class_names = predict_labels(path, ckpt)
    print(f"checkpoint classes: {class_names}")
    print(f"{len(regions)} regions found in {path}")
    for rid, mask in regions.items():
        ys, xs = np.where(mask)
        print(f"  region {rid}: {mask.sum()}px, center=({xs.mean():.0f},{ys.mean():.0f})")
