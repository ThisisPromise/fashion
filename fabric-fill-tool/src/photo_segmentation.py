"""Region segmentation for real garment photos.

Reuses the trained TinyGarmentSegModel from the submission project
(background/body/sleeve, frozen MobileNetV3-Small backbone + FPN decoder) --
no retraining here, this is the same checkpoint. What's new is turning its
two masks into four selectable regions:
  - sleeves are naturally two separate blobs on a worn garment (left arm and
    right arm don't touch), so connected-components splits them for free,
    the same trick that split flats by their drawn zip line.
  - torso is one connected blob, so it's split into left/right halves by its
    own bounding-box centerline -- a rougher approximation than a real seam,
    worth being honest about.

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
CHECKPOINT_PATH = os.path.join(SUBMISSION_DIR, "outputs", "checkpoint.pt")

sys.path.insert(0, SUBMISSION_DIR)
from src.model import TinyGarmentSegModel  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

MIN_REGION_AREA_PX = 200

_model = None
_cfg = None


def _load_model():
    global _model, _cfg
    if _model is None:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = TinyGarmentSegModel(num_classes=cfg["num_classes"], decoder_ch=cfg["decoder_channels"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _model, _cfg = model, cfg
    return _model, _cfg


@torch.no_grad()
def predict_body_sleeve(image_path):
    """Same prediction path as submission/src/placement.py:predict_regions --
    model runs at its training resolution, argmax first, then nearest-
    neighbor upsample back to the photo's real resolution."""
    model, cfg = _load_model()
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
    return pred_full == 1, pred_full == 2  # body_mask, sleeve_mask


def _label_and_filter(mask, min_area=MIN_REGION_AREA_PX):
    labeled, n = ndi.label(mask, structure=np.ones((3, 3)))
    regions = []
    for label_id in range(1, n + 1):
        m = labeled == label_id
        if m.sum() >= min_area:
            regions.append(m)
    return regions


def _split_left_right(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    mid_x = (xs.min() + xs.max()) / 2
    col_idx = np.arange(mask.shape[1])[None, :]
    left = mask & (col_idx < mid_x)
    right = mask & (col_idx >= mid_x)
    return [m for m in (left, right) if m.sum() >= MIN_REGION_AREA_PX]


def extract_regions(image_path):
    """Returns {region_id: bool mask} -- same shape of return value as
    flats_segmentation.extract_regions, so the frontend needs zero changes
    to handle either source."""
    body_mask, sleeve_mask = predict_body_sleeve(image_path)
    torso_mask = body_mask & ~sleeve_mask

    regions = {}
    region_id = 1
    for m in _split_left_right(torso_mask):
        regions[region_id] = m
        region_id += 1
    for m in _label_and_filter(sleeve_mask):
        regions[region_id] = m
        region_id += 1

    return regions


if __name__ == "__main__":
    import sys as _sys
    path = _sys.argv[1] if len(_sys.argv) > 1 else "fabric-fill-tool/scratch/real_photo_case01.jpg"
    regions = extract_regions(path)
    print(f"{len(regions)} regions found in {path}")
    for rid, mask in regions.items():
        ys, xs = np.where(mask)
        print(f"  region {rid}: {mask.sum()}px, center=({xs.mean():.0f},{ys.mean():.0f})")
