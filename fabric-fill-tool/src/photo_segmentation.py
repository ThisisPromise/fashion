"""Region segmentation for real garment photos.

Two trained segmentation checkpoints -- one for upper-body garments
(background/body/neckline/collar/pocket/zipper/sleeve), one for lower-body
garments (background/body/pocket/zipper) -- turned into a set of
independently selectable regions:

  - class 1 ("body") is one connected blob on a worn garment, so it's split
    into left/right halves by a centerline (the real shoulder midpoint when
    a person is detected, otherwise the mask's own bounding-box midpoint).
  - collar and neckline get the same left/right split, since they're
    anatomically two-sided the same way the torso is.
  - every other class (sleeve, pocket, zipper) is split by connected
    components. Sleeves naturally come out as two separate blobs since the
    left and right arm don't touch; pocket/zipper come out as however many
    blobs the model actually predicted, which may be zero on a given photo.

Pose landmarks (MediaPipe, see pose_utils_extended.py) are used as a
geometric sanity check on top of the segmentation output when a person is
confidently detected:
  - predictions are clipped to a padded box around the detected person,
    which removes false positives on background content.
  - the left/right split uses the real shoulder midpoint instead of a
    bounding-box guess.
  - lower-body predictions are clipped to not extend above the hip line, so
    an occluding garment (e.g. a jacket over the waistband) can't produce
    "pants" pixels above where the waist actually is.
No person detected means none of this runs; results fall back to plain
bounding-box-based behavior.
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

UPPERBODY_CHECKPOINT = os.path.join(TOOL_DIR, "outputs", "upperbody_extended", "checkpoint.pt")
LOWERBODY_CHECKPOINT = os.path.join(TOOL_DIR, "outputs", "lowerbody", "checkpoint.pt")
DEFAULT_CHECKPOINT = UPPERBODY_CHECKPOINT

sys.path.insert(0, SUBMISSION_DIR)
sys.path.insert(0, SRC_DIR)
from src.model import TinyGarmentSegModel  # noqa: E402
from pose_utils_extended import get_pose_info  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

MIN_REGION_AREA_PX = 150

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
    checkpoint's class_names.

    The model's raw per-class scores are upsampled to the photo's real
    resolution with bilinear interpolation before picking a winning class
    per pixel, rather than picking the winner at the model's native
    resolution and upsampling that decision with nearest-neighbor -- the
    latter turns each low-resolution cell into a visible block on a larger
    photo. Upsampling the scores first lets the class boundary follow a
    smooth curve instead."""
    model, cfg = _load_model(checkpoint_path)
    garment_image = Image.open(image_path).convert("RGB")
    size = cfg["image_size"]
    orig_w, orig_h = garment_image.size

    resized = garment_image.resize((size, size), Image.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)

    logits = model(tensor)
    logits_full = torch.nn.functional.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    pred_full = logits_full.argmax(dim=1)[0].numpy().astype(np.uint8)
    return pred_full, cfg["class_names"]


def _label_and_filter(mask, min_area=MIN_REGION_AREA_PX):
    labeled, n = ndi.label(mask, structure=np.ones((3, 3)))
    regions = []
    for label_id in range(1, n + 1):
        m = labeled == label_id
        if m.sum() >= min_area:
            regions.append(m)
    return regions


def _split_left_right(mask, min_area=MIN_REGION_AREA_PX, center_x=None):
    """center_x, when given, is used as the split line instead of the
    mask's own bounding-box midpoint -- pass the detected shoulder midpoint
    for a real anatomical centerline."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    mid_x = center_x if center_x is not None else (xs.min() + xs.max()) / 2
    col_idx = np.arange(mask.shape[1])[None, :]
    left = mask & (col_idx < mid_x)
    right = mask & (col_idx >= mid_x)
    return [m for m in (left, right) if m.sum() >= min_area]


def _clip_to_bbox(mask, bbox, image_shape, padding_frac=0.15):
    """Zeroes out anything outside a padded box around the detected
    person."""
    h, w = image_shape
    x0, y0, x1, y1 = bbox
    pad_x = (x1 - x0) * padding_frac
    pad_y = (y1 - y0) * padding_frac
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    clip = np.zeros_like(mask)
    clip[int(y0):int(y1), int(x0):int(x1)] = True
    return mask & clip


def _clip_above_hip(mask, hip_y, margin_px):
    """Zeroes out anything above (hip_y - margin). margin_px allows some
    slack since the real waistband usually sits a little above the hip
    landmark itself."""
    cutoff = max(0, hip_y - margin_px)
    clip = np.zeros_like(mask)
    clip[int(cutoff):, :] = True
    return mask & clip


# Anatomically two-sided part classes get the same left/right split as the
# torso. Sleeve/pocket/zipper don't: sleeves already separate naturally
# (left and right arm don't touch), and there's no "two sides" reading for
# a single pocket or zipper.
BILATERAL_CLASS_NAMES = {"collar", "neckline"}


def extract_regions(image_path, checkpoint_path=DEFAULT_CHECKPOINT, pose_info=None, clip_above_hip=False):
    """Returns {region_id: bool mask}. Class 1 (body) is always split
    left/right; bilateral part classes (collar, neckline) get the same
    treatment; everything else is split by connected components.

    pose_info (from pose_utils_extended.get_pose_info), when given,
    restricts every region to a padded box around the detected person and
    uses the shoulder midpoint as the left/right split line instead of a
    bounding-box guess. clip_above_hip additionally drops anything above
    the hip line -- meaningful only for the lower-body checkpoint."""
    pred_full, class_names = predict_labels(image_path, checkpoint_path)
    center_x = pose_info["shoulder_center_x"] if pose_info else None

    regions = {}
    region_id = 1

    body_mask = pred_full == 1
    if pose_info:
        body_mask = _clip_to_bbox(body_mask, pose_info["person_bbox"], pred_full.shape)
        if clip_above_hip:
            margin = 0.15 * (pose_info["hip_y"] - pose_info["shoulder_y"])
            body_mask = _clip_above_hip(body_mask, pose_info["hip_y"], margin)
    for m in _split_left_right(body_mask, center_x=center_x):
        regions[region_id] = m
        region_id += 1

    for class_id in range(2, len(class_names)):
        class_mask = pred_full == class_id
        if pose_info:
            class_mask = _clip_to_bbox(class_mask, pose_info["person_bbox"], pred_full.shape)
            if clip_above_hip:
                margin = 0.15 * (pose_info["hip_y"] - pose_info["shoulder_y"])
                class_mask = _clip_above_hip(class_mask, pose_info["hip_y"], margin)
        if class_names[class_id] in BILATERAL_CLASS_NAMES:
            pieces = _split_left_right(class_mask, center_x=center_x)
        else:
            pieces = _label_and_filter(class_mask)
        for m in pieces:
            regions[region_id] = m
            region_id += 1

    return regions


def extract_regions_combined(image_path):
    """Runs the upper-body and lower-body checkpoints on the same photo and
    merges their regions into one set, so a photo showing a top, trousers,
    or both is handled the same way with no manual selection needed.

    Where the two models' predictions overlap in pixel space (e.g. right at
    the waistband), the lower-body region wins in the final region map,
    since it's merged in second -- an arbitrary tie-break that only affects
    which single region a pixel in that thin overlap strip belongs to.

    Pose runs once here and is shared by both checkpoints. Any failure to
    get a pose reading (no person detected, or an infrastructure problem
    such as a missing model file) is treated the same way: proceed without
    pose, falling back to plain bounding-box-based behavior."""
    try:
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        pose_info = get_pose_info(image_rgb)
    except Exception:
        pose_info = None

    upper_regions = extract_regions(image_path, UPPERBODY_CHECKPOINT, pose_info=pose_info)
    lower_regions = extract_regions(image_path, LOWERBODY_CHECKPOINT, pose_info=pose_info, clip_above_hip=True)

    merged = {}
    region_id = 1
    for m in upper_regions.values():
        merged[region_id] = m
        region_id += 1
    for m in lower_regions.values():
        merged[region_id] = m
        region_id += 1
    return merged


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("usage: python photo_segmentation.py <image_path> [checkpoint_path]")
        _sys.exit(1)
    path = _sys.argv[1]
    ckpt = _sys.argv[2] if len(_sys.argv) > 2 else DEFAULT_CHECKPOINT
    regions = extract_regions(path, ckpt)
    _, class_names = predict_labels(path, ckpt)
    print(f"checkpoint classes: {class_names}")
    print(f"{len(regions)} regions found in {path}")
    for rid, mask in regions.items():
        ys, xs = np.where(mask)
        print(f"  region {rid}: {mask.sum()}px, center=({xs.mean():.0f},{ys.mean():.0f})")
