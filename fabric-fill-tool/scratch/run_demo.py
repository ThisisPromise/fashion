"""End-to-end proof: segment the synthetic flat, visualize the regions found,
then fill two of them (collar + left sleeve) with the synthetic fabric."""

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "fabric-fill-tool/src")
from flats_segmentation import extract_regions  # noqa: E402
from fabric_fill import fill_region_with_fabric  # noqa: E402

FLAT_PATH = "fabric-fill-tool/scratch/sample_flat.png"
FABRIC_PATH = "fabric-fill-tool/scratch/sample_fabric.png"

REGION_COLORS = [
    (255, 99, 71), (100, 149, 237), (60, 179, 113),
    (238, 130, 238), (255, 165, 0), (0, 206, 209),
]


def visualize_regions(regions, line_mask, out_path):
    canvas = np.full((*line_mask.shape, 3), 255, dtype=np.uint8)
    for i, (region_id, mask) in enumerate(regions.items()):
        canvas[mask] = REGION_COLORS[i % len(REGION_COLORS)]
    canvas[line_mask] = (0, 0, 0)
    Image.fromarray(canvas).save(out_path)
    print(f"wrote {out_path}")


def describe(regions):
    print(f"\nFound {len(regions)} regions:")
    described = []
    for region_id, mask in regions.items():
        ys, xs = np.where(mask)
        cx, cy = xs.mean(), ys.mean()
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        label = "?"
        if cy < ys.min() + 5 or (h < 80 and cy < 250):
            label = "collar"
        elif w < 150 and cx < 250:
            label = "left sleeve"
        elif w < 150 and cx > 350:
            label = "right sleeve"
        elif cx < 300:
            label = "torso (left half, next to zip)"
        else:
            label = "torso (right half, next to zip)"
        print(f"  region {region_id}: {mask.sum()}px, center=({cx:.0f},{cy:.0f}) -> guessed: {label}")
        described.append((region_id, mask, label))
    return described


def main():
    regions, line_mask = extract_regions(FLAT_PATH)
    visualize_regions(regions, line_mask, "fabric-fill-tool/scratch/region_map.png")
    described = describe(regions)

    flat_img = Image.open(FLAT_PATH).convert("RGB")
    fabric = Image.open(FABRIC_PATH)

    collar_mask = next(mask for _, mask, label in described if label == "collar")
    sleeve_mask = next(mask for _, mask, label in described if label == "left sleeve")

    result = fill_region_with_fabric(flat_img, collar_mask, fabric, mode="fit")
    result = fill_region_with_fabric(result, sleeve_mask, fabric, mode="tile")
    result.save("fabric-fill-tool/scratch/result_filled.png")
    print("\nwrote fabric-fill-tool/scratch/result_filled.png")


if __name__ == "__main__":
    main()
