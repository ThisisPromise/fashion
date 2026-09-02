"""Fills an arbitrary region mask with a fabric texture.

Shared by both pipelines (flats via classical CV, photos via the segmentation
model) -- neither cares how its mask was produced, this just needs a boolean
mask the same size as the image and a fabric image to apply inside it.
"""

import numpy as np
from PIL import Image


def _tile_fabric_to_size(fabric: Image.Image, width, height):
    """Repeats the fabric image to cover a width x height area."""
    fw, fh = fabric.size
    tiles_x = -(-width // fw)   # ceil division
    tiles_y = -(-height // fh)
    canvas = Image.new("RGB", (tiles_x * fw, tiles_y * fh))
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            canvas.paste(fabric, (tx * fw, ty * fh))
    return canvas.crop((0, 0, width, height))


def fill_region_with_fabric(image: Image.Image, region_mask: np.ndarray,
                             fabric: Image.Image, mode="tile",
                             preserve_shading=False):
    """Returns a new image with `fabric` applied everywhere `region_mask` is
    True. mode="tile" repeats the fabric pattern (good for a large panel);
    mode="fit" scales the fabric to cover the region's bounding box once
    (good for a small region like a collar, avoids visible tile seams).
    preserve_shading=True keeps the original image's lightness under the new
    fabric color -- for photos with real folds/shadows; flats are flat, so
    this defaults off.
    """
    ys, xs = np.where(region_mask)
    if len(ys) == 0:
        return image.copy()

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    region_w, region_h = x1 - x0, y1 - y0

    fabric_rgb = fabric.convert("RGB")
    if mode == "tile":
        patch = _tile_fabric_to_size(fabric_rgb, region_w, region_h)
    else:  # "fit"
        patch = fabric_rgb.resize((region_w, region_h), Image.LANCZOS)

    result = image.convert("RGB").copy()
    patch_arr = np.array(patch)

    if preserve_shading:
        original_patch = np.array(image.convert("RGB").crop((x0, y0, x1, y1))).astype(np.float32)
        orig_lightness = original_patch.mean(axis=2, keepdims=True) / 255.0
        patch_arr = (patch_arr.astype(np.float32) * (0.5 + orig_lightness)).clip(0, 255)
        patch_arr = patch_arr.astype(np.uint8)

    result_arr = np.array(result)
    local_mask = region_mask[y0:y1, x0:x1]
    result_arr[y0:y1, x0:x1][local_mask] = patch_arr[local_mask]

    return Image.fromarray(result_arr)
