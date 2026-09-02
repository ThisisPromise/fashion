import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "fabric-fill-tool/src")
from flats_segmentation import extract_regions  # noqa: E402

REGION_COLORS = [
    (255, 99, 71), (100, 149, 237), (60, 179, 113), (238, 130, 238),
    (255, 165, 0), (0, 206, 209), (154, 205, 50), (219, 112, 147),
    (72, 209, 204), (255, 215, 0), (147, 112, 219), (0, 191, 255),
]


def visualize(path, out_path):
    regions, line_mask = extract_regions(path)
    canvas = np.full((*line_mask.shape, 3), 255, dtype=np.uint8)
    for i, (region_id, mask) in enumerate(regions.items()):
        canvas[mask] = REGION_COLORS[i % len(REGION_COLORS)]
    canvas[line_mask] = (0, 0, 0)
    Image.fromarray(canvas).save(out_path)
    print(f"{path}: {len(regions)} regions found -> {out_path}")
    sizes = sorted((int(m.sum()) for m in regions.values()), reverse=True)
    print(f"  region sizes (px): {sizes[:15]}{'...' if len(sizes) > 15 else ''}")


if __name__ == "__main__":
    visualize("sketch.jpg", "fabric-fill-tool/scratch/sketch_regions.png")
    visualize("sketch1.png", "fabric-fill-tool/scratch/sketch1_regions.png")
