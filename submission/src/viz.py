"""Shared overlay rendering for the demo/eval/prediction-check scripts."""

import numpy as np
from PIL import Image, ImageDraw

BODY_COLOR = [30, 144, 255]
SLEEVE_COLOR = [255, 0, 255]
TARGET_COLOR = [255, 220, 0]


def overlay_regions(image_arr, body_mask, sleeve_mask, target_region=None, alpha=0.5):
    overlay = image_arr.astype(float).copy()
    color = np.zeros_like(overlay)
    color[body_mask] = BODY_COLOR
    color[sleeve_mask] = SLEEVE_COLOR
    if target_region is not None:
        color[target_region] = TARGET_COLOR
    has_color = color.sum(-1) > 0
    overlay[has_color] = (1 - alpha) * overlay[has_color] + alpha * color[has_color]
    return overlay.astype("uint8")


def rejection_placard(size_wh, reason, wrap_width=40):
    placard = Image.new("RGB", size_wh, (245, 245, 245))
    wrapped = "\n".join(reason[i:i + wrap_width] for i in range(0, len(reason), wrap_width))
    ImageDraw.Draw(placard).text((10, 10), f"REJECTED:\n{wrapped}", fill=(180, 0, 0))
    return placard
