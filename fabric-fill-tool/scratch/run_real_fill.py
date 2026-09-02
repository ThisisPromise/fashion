import sys

from PIL import Image

sys.path.insert(0, "fabric-fill-tool/src")
from flats_segmentation import extract_regions  # noqa: E402
from fabric_fill import fill_region_with_fabric  # noqa: E402

FABRIC_PATH = "fabric-fill-tool/scratch/sample_fabric.png"


def fill_top_regions(image_path, out_path, top_n=2):
    regions, _ = extract_regions(image_path)
    img = Image.open(image_path).convert("RGB")
    fabric = Image.open(FABRIC_PATH)

    ranked = sorted(regions.items(), key=lambda kv: -kv[1].sum())[:top_n]
    result = img
    for region_id, mask in ranked:
        result = fill_region_with_fabric(result, mask, fabric, mode="tile")
    result.save(out_path)
    print(f"{image_path}: filled top {top_n} regions -> {out_path}")


if __name__ == "__main__":
    fill_top_regions("sketch.jpg", "fabric-fill-tool/scratch/sketch_filled.png")
    fill_top_regions("sketch1.png", "fabric-fill-tool/scratch/sketch1_filled.png")
