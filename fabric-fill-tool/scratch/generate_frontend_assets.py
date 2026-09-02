"""Precomputes everything the interactive demo page needs:
- a resized base image (for a reasonable payload size)
- a label-map PNG, same resolution, where each pixel's RED channel value IS
  the region id (0 = background/line, not clickable) -- lossless PNG so the
  browser can read exact region ids back out per-pixel
- base64 data URIs for the base image, label map, and fabric swatches,
  written to a JS file the HTML page includes directly
"""

import base64
import io
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "fabric-fill-tool/src")
from flats_segmentation import extract_regions  # noqa: E402

TARGET_WIDTH = 460


def to_data_uri(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_asset(image_path, key):
    # Segment at FULL resolution -- this is where it actually worked well.
    # Downscaling before segmenting blurs line gaps shut differently and
    # breaks regions (confirmed: it merged the skirt into background on the
    # dress sketch). Only the *output* gets downscaled, and the label map
    # specifically with NEAREST, so discrete region ids never blend into a
    # different, meaningless id at a boundary -- same principle as resizing
    # a class-id mask elsewhere in this project.
    regions, line_mask = extract_regions(image_path)

    img = Image.open(image_path).convert("RGB")
    label_map_full = np.zeros(line_mask.shape, dtype=np.uint8)
    region_info = {}
    for i, (region_id, mask) in enumerate(sorted(regions.items(), key=lambda kv: -kv[1].sum()), start=1):
        label_map_full[mask] = i
        ys, xs = np.where(mask)
        region_info[i] = {
            "area": int(mask.sum()),
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
        }

    scale = TARGET_WIDTH / img.width
    new_size = (TARGET_WIDTH, round(img.height * scale))

    img_small = img.resize(new_size, Image.LANCZOS)

    label_img_full = Image.fromarray(
        np.stack([label_map_full, np.zeros_like(label_map_full), np.zeros_like(label_map_full)], axis=-1)
    )
    label_img_small = label_img_full.resize(new_size, Image.NEAREST)

    return {
        "base_data_uri": to_data_uri(img_small),
        "label_data_uri": to_data_uri(label_img_small),
        "width": new_size[0],
        "height": new_size[1],
        "region_count": len(regions),
        "region_info": region_info,
    }


def main():
    assets = {
        "sketch": build_asset("sketch.jpg", "sketch"),
        "sketch1": build_asset("sketch1.png", "sketch1"),
    }

    fabric1 = Image.open("fabric-fill-tool/scratch/sample_fabric.png")
    fabric2_path = "fabric-fill-tool/scratch/sample_fabric2.png"
    # second swatch: cooler palette, different shapes, so the demo can show
    # two different groups getting two different fabrics
    from PIL import ImageDraw
    import random
    random.seed(1)
    size = 80
    f2 = Image.new("RGB", (size, size), color=(20, 60, 130))
    d = ImageDraw.Draw(f2)
    colors = [(255, 200, 40), (240, 240, 240), (200, 30, 90)]
    for i in range(7):
        c = colors[i % len(colors)]
        x, y = random.randint(0, size), random.randint(0, size)
        r = random.randint(8, 18)
        d.rectangle((x - r, y - r, x + r, y + r), fill=c)
    f2.save(fabric2_path)

    fabrics = {
        "fabric1": to_data_uri(fabric1),
        "fabric2": to_data_uri(Image.open(fabric2_path)),
    }

    out_path = "fabric-fill-tool/scratch/frontend_assets.py"
    with open(out_path, "w") as f:
        f.write("ASSETS = " + repr(assets) + "\n")
        f.write("FABRICS = " + repr(fabrics) + "\n")

    for key, a in assets.items():
        print(f"{key}: {a['width']}x{a['height']}, {a['region_count']} regions, "
              f"payload ~{len(a['base_data_uri']) + len(a['label_data_uri'])} chars")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
