"""Builds the training subset from Fashionpedia's val2020 split and
rasterizes 3-class masks: 0=background, 1=body, 2=sleeve. Boundary isn't a
learned class -- it's derived from body/sleeve at inference time
(placement.py)."""

import json
import os
import random

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

RAW_IMAGE_DIR = "data/fashionpedia/images_raw/test"
ANNOTATIONS_PATH = "data/fashionpedia/instances_attributes_val2020.json"

OUT_IMAGE_DIR = "data/processed/images"
OUT_MASK_DIR = "data/processed/masks"
SPLIT_MANIFEST_PATH = "data/processed/split_manifest.csv"

UPPERBODY_AND_WHOLEBODY_GARMENT_IDS = {0, 1, 2, 3, 4, 5, 9, 10, 11, 12}
SLEEVE_ID = 31

VAL_FRACTION = 0.15
SEED = 0


def polygon_or_rle_to_binary_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


def build_target_mask(image_info, annotations_for_image):
    h, w = image_info["height"], image_info["width"]
    body = np.zeros((h, w), dtype=bool)
    sleeve = np.zeros((h, w), dtype=bool)

    for ann in annotations_for_image:
        cat = ann["category_id"]
        if cat not in UPPERBODY_AND_WHOLEBODY_GARMENT_IDS and cat != SLEEVE_ID:
            continue
        m = polygon_or_rle_to_binary_mask(ann["segmentation"], h, w)
        if cat == SLEEVE_ID:
            sleeve |= m
        else:
            body |= m

    target = np.zeros((h, w), dtype=np.uint8)
    target[body] = 1
    target[sleeve] = 2  # sleeve overrides body in overlap regions
    return target


def main():
    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    qualifying_ids = []
    for image_id, anns in anns_by_image.items():
        cats = {a["category_id"] for a in anns}
        if (cats & UPPERBODY_AND_WHOLEBODY_GARMENT_IDS) and SLEEVE_ID in cats:
            qualifying_ids.append(image_id)

    print(f"Qualifying images (upper/wholebody garment + sleeve annotation): {len(qualifying_ids)}")

    kept, skipped_missing_file = [], []
    for image_id in sorted(qualifying_ids):
        info = images_by_id[image_id]
        src_path = os.path.join(RAW_IMAGE_DIR, info["file_name"])
        if not os.path.exists(src_path):
            skipped_missing_file.append(image_id)
            continue
        kept.append(image_id)

    print(f"Images with a matching file on disk: {len(kept)}  (missing: {len(skipped_missing_file)})")

    for image_id in kept:
        info = images_by_id[image_id]
        target = build_target_mask(info, anns_by_image[image_id])

        src_path = os.path.join(RAW_IMAGE_DIR, info["file_name"])
        img = Image.open(src_path).convert("RGB")

        stem = str(image_id)
        img.save(os.path.join(OUT_IMAGE_DIR, f"{stem}.jpg"), quality=95)
        Image.fromarray(target, mode="L").save(os.path.join(OUT_MASK_DIR, f"{stem}.png"))

    random.seed(SEED)
    shuffled = kept.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * VAL_FRACTION)))
    val_ids = set(shuffled[:n_val])

    with open(SPLIT_MANIFEST_PATH, "w") as f:
        f.write("image_id,split\n")
        for image_id in kept:
            split = "val" if image_id in val_ids else "train"
            f.write(f"{image_id},{split}\n")

    n_train = len(kept) - len(val_ids)
    print(f"Split manifest written: {n_train} train / {len(val_ids)} val -> {SPLIT_MANIFEST_PATH}")

    # Sanity check: class pixel coverage, so an empty-mask bug would show up now.
    body_px = sleeve_px = total_px = 0
    for image_id in kept[:50]:
        m = np.array(Image.open(os.path.join(OUT_MASK_DIR, f"{image_id}.png")))
        body_px += (m == 1).sum()
        sleeve_px += (m == 2).sum()
        total_px += m.size
    print(f"Sanity check over first 50 images: body={body_px/total_px:.3f} "
          f"sleeve={sleeve_px/total_px:.3f} of pixels")


if __name__ == "__main__":
    main()
