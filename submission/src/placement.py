"""Deterministic artwork placement from the segmentation model's predicted
regions -- no hardcoded offsets.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from src.model import TinyGarmentSegModel
from src.pose_utils import get_shoulder_y

MIN_INSCRIBED_RADIUS_PX = 6  # below this, an artwork placement would be unusably tiny
MAX_ARTWORK_FRACTION_OF_RADIUS = 1.0  # artwork half-diagonal <= this * inscribed radius
MIN_TORSO_AREA_PX = 400  # below this, treat "no garment detected" as a hard failure

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Keyword vocabulary for parse_instruction. This is plain keyword matching,
# not language understanding -- an unrecognized phrase silently falls back
# to "centre" rather than raising an error, which is a known limitation.
CHEST_SYNONYMS = ("chest", "breast")


@dataclass
class PlacementResult:
    status: str  # "ok" | "rejected"
    reason: str
    composited_image: Optional[Image.Image] = None
    predicted_body: Optional[np.ndarray] = None
    predicted_sleeve: Optional[np.ndarray] = None
    target_region: Optional[np.ndarray] = None
    anchor_xy: Optional[tuple] = None
    artwork_size_px: Optional[tuple] = None
    debug: dict = field(default_factory=dict)


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TinyGarmentSegModel(num_classes=cfg["num_classes"], decoder_ch=cfg["decoder_channels"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_regions(model, cfg, garment_image: Image.Image):
    """(body_mask, sleeve_mask) as bool arrays at the original image
    resolution -- the model runs at its training resolution and we upsample
    the prediction back so placement geometry matches the caller's pixels."""
    size = cfg["image_size"]
    orig_w, orig_h = garment_image.size

    resized = garment_image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)

    logits = model(tensor)
    pred = logits.argmax(dim=1)[0].numpy()

    pred_img = Image.fromarray(pred.astype(np.uint8), mode="L").resize((orig_w, orig_h), Image.NEAREST)
    pred_full = np.array(pred_img)
    return pred_full == 1, pred_full == 2


def parse_instruction(instruction):
    text = instruction.lower()
    if "left" in text:
        horizontal = "left"
    elif "right" in text:
        horizontal = "right"
    else:
        horizontal = "centre"

    vertical = "chest" if any(word in text for word in CHEST_SYNONYMS) else "centre"
    view = "back" if "back" in text else "front"
    return {"view": view, "horizontal": horizontal, "vertical": vertical}


def infer_flat_lay_torso(garment_silhouette, side_margin_fraction=0.28):
    """Torso estimate for flat-lay/hanger photos: treat the outer margin of
    the silhouette's bounding box as sleeve, the rest as torso. Used when
    there's no person to check the model's body/sleeve split against."""
    ys, xs = np.where(garment_silhouette)
    if len(ys) == 0:
        return np.zeros_like(garment_silhouette)

    x0, x1 = xs.min(), xs.max()
    width = x1 - x0 + 1
    left_boundary = x0 + int(width * side_margin_fraction)
    right_boundary = x1 - int(width * side_margin_fraction)

    torso = garment_silhouette.copy()
    torso[:, :left_boundary] = False
    torso[:, right_boundary:] = False
    return torso


def build_target_region(torso_mask, horizontal, vertical, shoulder_y=None, side_fraction=0.4):
    ys, xs = np.where(torso_mask)
    if len(ys) == 0:
        return np.zeros_like(torso_mask)

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    width = x1 - x0 + 1

    # Prefer the shoulder line over the mask's own bbox top: a raised arm
    # pulls the bbox top up past the shoulder toward the hand.
    if shoulder_y is not None and y0 <= shoulder_y <= y1:
        top = shoulder_y
    else:
        # No shoulder reference: bbox top is the collar, which sits above
        # where a shoulder line would be, so skip an equivalent margin.
        top = y0 + (y1 - y0) * 0.22
    height = y1 - top + 1

    vert_mask = np.zeros_like(torso_mask)
    if vertical == "chest":
        vert_mask[int(top):int(top + height * 0.45), :] = True
    else:  # "centre": middle band of the torso, avoiding the top/bottom hem
        vert_mask[int(top + height * 0.2):int(top + height * 0.85), :] = True

    horiz_mask = np.zeros_like(torso_mask)
    if horizontal == "left":
        horiz_mask[:, x0:x0 + int(width * side_fraction)] = True
    elif horizontal == "right":
        horiz_mask[:, x1 - int(width * side_fraction):x1 + 1] = True
    else:  # "centre"
        horiz_mask[:, x0 + int(width * 0.25):x1 - int(width * 0.25) + 1] = True

    return torso_mask & vert_mask & horiz_mask


def find_inscribed_anchor(region_mask):
    """Center of the region's largest inscribed circle, via distance
    transform. Deterministic and always strictly inside the region."""
    if region_mask.sum() == 0:
        return None, 0.0
    dist = ndi.distance_transform_edt(region_mask)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    return (int(cx), int(cy)), float(dist[cy, cx])


def fit_artwork_to_radius(artwork: Image.Image, radius_px):
    w, h = artwork.size
    half_diag = 0.5 * (w ** 2 + h ** 2) ** 0.5
    target_half_diag = radius_px * MAX_ARTWORK_FRACTION_OF_RADIUS
    scale = min(1.0, target_half_diag / half_diag) if half_diag > 0 else 1.0
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return artwork if new_size == (w, h) else artwork.resize(new_size, Image.LANCZOS)


def composite_artwork(garment_image, artwork, anchor_xy):
    canvas = garment_image.convert("RGBA")
    aw, ah = artwork.size
    cx, cy = anchor_xy
    canvas.alpha_composite(artwork.convert("RGBA"), dest=(cx - aw // 2, cy - ah // 2))
    return canvas.convert("RGB")


def place_artwork(garment_image_path, artwork_path, instruction, model, cfg):
    """Deterministic: same inputs and model weights always give the same
    output -- no randomness anywhere on this path."""
    garment_image = Image.open(garment_image_path).convert("RGB")
    artwork = Image.open(artwork_path).convert("RGBA")

    body_mask, sleeve_mask = predict_regions(model, cfg, garment_image)
    shoulder_y = get_shoulder_y(np.array(garment_image))
    sleeve_fraction = sleeve_mask.sum() / max(int(body_mask.sum() + sleeve_mask.sum()), 1)
    used_flat_lay_fallback = shoulder_y is None

    if used_flat_lay_fallback:
        # No person detected: the model's body/sleeve class split was
        # trained almost entirely on worn garments and isn't trustworthy
        # here (a real flat-lay coat photo got labeled mostly "sleeve").
        # Fall back to a geometric torso estimate from the raw silhouette.
        torso_mask = infer_flat_lay_torso(body_mask | sleeve_mask)
    else:
        torso_mask = body_mask & ~sleeve_mask

    debug = {"used_flat_lay_fallback": used_flat_lay_fallback, "sleeve_fraction": sleeve_fraction}

    if torso_mask.sum() < MIN_TORSO_AREA_PX:
        return PlacementResult(
            status="rejected",
            reason=f"Predicted garment body area ({int(torso_mask.sum())}px) is too small or absent"
                   f"{' (flat-lay fallback attempted)' if used_flat_lay_fallback else ''}; "
                   f"refusing to guess a placement.",
            predicted_body=body_mask, predicted_sleeve=sleeve_mask, debug=debug,
        )

    parsed = parse_instruction(instruction)
    target_region = build_target_region(torso_mask, parsed["horizontal"], parsed["vertical"], shoulder_y=shoulder_y)
    debug = {**debug, "parsed_instruction": parsed, "shoulder_anchor_used": not used_flat_lay_fallback}

    anchor_xy, radius = find_inscribed_anchor(target_region)
    if anchor_xy is None or radius < MIN_INSCRIBED_RADIUS_PX:
        reason = (f"Requested region ('{instruction}') resolved to a region too small/thin "
                  f"(max inscribed radius={radius:.1f}px, need >={MIN_INSCRIBED_RADIUS_PX}px) "
                  f"to place artwork without risking a seam crossing or spillover.")
        if used_flat_lay_fallback:
            reason += " The flat-lay fallback was already attempted and also found nothing usable."
        return PlacementResult(
            status="rejected", reason=reason,
            predicted_body=body_mask, predicted_sleeve=sleeve_mask, target_region=target_region, debug=debug,
        )

    fitted_artwork = fit_artwork_to_radius(artwork, radius)
    composited = composite_artwork(garment_image, fitted_artwork, anchor_xy)

    return PlacementResult(
        status="ok",
        reason="placed" if not used_flat_lay_fallback else "placed (via flat-lay geometric fallback)",
        composited_image=composited,
        predicted_body=body_mask,
        predicted_sleeve=sleeve_mask,
        target_region=target_region,
        anchor_xy=anchor_xy,
        artwork_size_px=fitted_artwork.size,
        debug={**debug, "inscribed_radius_px": radius},
    )
