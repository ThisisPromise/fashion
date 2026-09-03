"""Region segmentation for clean fashion flats / technical drawings.

No model, no training data -- a flat's regions are already defined by its own
closed linework, so this finds them with classical image processing:
threshold the ink, close small gaps in the lines so nothing leaks between
regions, then label every enclosed area as its own region.
"""

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

LINE_THRESHOLD = 200          # pixel darkness below this counts as "ink"
GAP_CLOSE_ITERATIONS = 2      # how aggressively small gaps in lines get bridged
MIN_REGION_AREA_PX = 150      # connected components smaller than this are treated as noise (buttons, stitch marks)


def load_grayscale(image_path):
    return np.array(Image.open(image_path).convert("L"))


def binarize_lines(gray, threshold=LINE_THRESHOLD):
    """True wherever the pixel is dark enough to be linework."""
    return gray < threshold


def close_line_gaps(line_mask, iterations=GAP_CLOSE_ITERATIONS):
    """Bridges small gaps in the linework so flood fill can't leak through
    an almost-but-not-quite-closed contour."""
    closed = ndi.binary_dilation(line_mask, iterations=iterations)
    closed = ndi.binary_erosion(closed, iterations=iterations)
    return closed


def label_enclosed_regions(line_mask):
    """Labels every area NOT covered by linework. Each enclosed shape (collar,
    sleeve, torso panel, ...) gets its own label; the label touching the image
    border is background, not a garment region."""
    fillable = ~line_mask
    labeled, n = ndi.label(fillable, structure=np.ones((3, 3)))  # 8-connectivity
    return labeled, n


def find_background_labels(labeled):
    """Any label touching the outer edge of the image is background/outside
    the garment, not an enclosed region."""
    border = np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1],
    ])
    return set(border.tolist()) - {0}


def extract_regions(image_path, threshold=LINE_THRESHOLD,
                     gap_close_iterations=GAP_CLOSE_ITERATIONS,
                     min_area=MIN_REGION_AREA_PX):
    """Returns {region_id: bool mask} for every enclosed region in a flat,
    background and small-noise components already filtered out."""
    gray = load_grayscale(image_path)
    line_mask = binarize_lines(gray, threshold)
    line_mask = close_line_gaps(line_mask, gap_close_iterations)

    labeled, n = label_enclosed_regions(line_mask)
    background_labels = find_background_labels(labeled)

    regions = {}
    for label_id in range(1, n + 1):
        if label_id in background_labels:
            continue
        mask = labeled == label_id
        area = int(mask.sum())
        if area < min_area:
            continue
        regions[label_id] = mask

    return regions, line_mask


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python flats_segmentation.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    regions, line_mask = extract_regions(path)
    print(f"Found {len(regions)} regions in {path}:")
    for region_id, mask in sorted(regions.items(), key=lambda kv: -kv[1].sum()):
        ys, xs = np.where(mask)
        print(f"  region {region_id}: {mask.sum()} px, "
              f"bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")
