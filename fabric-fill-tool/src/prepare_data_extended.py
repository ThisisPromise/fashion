"""Builds two extended training subsets from the same Fashionpedia val2020
data as the original submission -- one per model, kept as two SEPARATE
datasets/checkpoints rather than one bigger one, per the plan:

  upperbody_extended: same 741 upper/wholebody-garment images as the
    original (identical qualifying filter: main garment + sleeve present),
    now also rasterizing collar/pocket/zipper/neckline wherever Fashionpedia
    happens to have them annotated on those same images. Doesn't touch or
    require the original submission/ files at all.

  lowerbody: a new garment scope entirely -- pants/shorts/skirt, which the
    original model deliberately excluded. Requires pocket/zipper on
    trousers/skirts is far from universal, so unlike sleeve for the
    upperbody set, no part class is required to qualify here -- just the
    main garment. That's a real difference from the original recipe, not an
    oversight.

Precedence when parts overlap: painted in the order listed per config, so
later entries win ties -- same "more specific wins" idea as the original
sleeve-overrides-body rule, extended to the new parts.
"""

import json
import os
import random

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

SUBMISSION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "submission")
RAW_IMAGE_DIR = os.path.join(SUBMISSION_DIR, "data", "fashionpedia", "images_raw", "test")
ANNOTATIONS_PATH = os.path.join(SUBMISSION_DIR, "data", "fashionpedia", "instances_attributes_val2020.json")

OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

VAL_FRACTION = 0.15
SEED = 0

# category id -> name, from Fashionpedia's actual 46-category ontology
# (verified against KMnP/fashionpedia-api's category_attributes_descriptions.json)
CATEGORY_NAMES = {
    0: "shirt/blouse", 1: "top/t-shirt", 2: "sweater", 3: "cardigan", 4: "jacket", 5: "vest",
    6: "pants", 7: "shorts", 8: "skirt", 9: "coat", 10: "dress", 11: "jumpsuit", 12: "cape",
    16: "tie", 19: "belt",
    27: "hood", 28: "collar", 29: "lapel", 30: "epaulette", 31: "sleeve", 32: "pocket",
    33: "neckline", 34: "buckle", 35: "zipper",
}

DATASETS = {
    "upperbody_extended": {
        "main_garment_ids": {0, 1, 2, 3, 4, 5, 9, 10, 11, 12},
        "required_id": 31,  # same qualifying rule as the original: must also have sleeve
        # painted in order -- later wins overlaps. class 0 is always background,
        # class 1 is always "body" (the main_garment_ids union).
        "part_ids_in_priority_order": [33, 28, 32, 35, 31],  # neckline, collar, pocket, zipper, sleeve
        "class_names": ["background", "body", "neckline", "collar", "pocket", "zipper", "sleeve"],
    },
    "lowerbody": {
        "main_garment_ids": {6, 7, 8},  # pants, shorts, skirt
        "required_id": None,  # no part is universal enough to require, unlike sleeve
        "part_ids_in_priority_order": [32, 35],  # pocket, zipper
        "class_names": ["background", "body", "pocket", "zipper"],
    },
}


def polygon_or_rle_to_binary_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


def build_target_mask(image_info, annotations_for_image, config):
    h, w = image_info["height"], image_info["width"]
    main_ids = config["main_garment_ids"]
    part_ids = config["part_ids_in_priority_order"]
    relevant_ids = main_ids | set(part_ids)

    masks_by_cat = {}
    for ann in annotations_for_image:
        cat = ann["category_id"]
        if cat not in relevant_ids:
            continue
        m = polygon_or_rle_to_binary_mask(ann["segmentation"], h, w)
        masks_by_cat[cat] = masks_by_cat.get(cat, np.zeros((h, w), dtype=bool)) | m

    body = np.zeros((h, w), dtype=bool)
    for cat in main_ids:
        if cat in masks_by_cat:
            body |= masks_by_cat[cat]

    target = np.zeros((h, w), dtype=np.uint8)
    target[body] = 1
    for class_id, cat in enumerate(part_ids, start=2):
        if cat in masks_by_cat:
            target[masks_by_cat[cat]] = class_id
    return target


def build_dataset(dataset_key):
    config = DATASETS[dataset_key]
    out_dir = os.path.join(OUT_ROOT, dataset_key)
    out_image_dir = os.path.join(out_dir, "images")
    out_mask_dir = os.path.join(out_dir, "masks")
    split_manifest_path = os.path.join(out_dir, "split_manifest.csv")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    print(f"\n=== {dataset_key} ===")
    print(f"main garment categories: {[CATEGORY_NAMES[c] for c in sorted(config['main_garment_ids'])]}")
    print(f"part classes (priority order): {[CATEGORY_NAMES[c] for c in config['part_ids_in_priority_order']]}")

    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    main_ids = config["main_garment_ids"]
    required_id = config["required_id"]

    qualifying_ids = []
    for image_id, anns in anns_by_image.items():
        cats = {a["category_id"] for a in anns}
        if not (cats & main_ids):
            continue
        if required_id is not None and required_id not in cats:
            continue
        qualifying_ids.append(image_id)

    print(f"Qualifying images: {len(qualifying_ids)}")

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
        target = build_target_mask(info, anns_by_image[image_id], config)

        src_path = os.path.join(RAW_IMAGE_DIR, info["file_name"])
        img = Image.open(src_path).convert("RGB")

        stem = str(image_id)
        img.save(os.path.join(out_image_dir, f"{stem}.jpg"), quality=95)
        Image.fromarray(target).convert("L").save(os.path.join(out_mask_dir, f"{stem}.png"))

    random.seed(SEED)
    shuffled = kept.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * VAL_FRACTION)))
    val_ids = set(shuffled[:n_val])

    with open(split_manifest_path, "w") as f:
        f.write("image_id,split\n")
        for image_id in kept:
            split = "val" if image_id in val_ids else "train"
            f.write(f"{image_id},{split}\n")

    n_train = len(kept) - len(val_ids)
    print(f"Split manifest written: {n_train} train / {len(val_ids)} val -> {split_manifest_path}")

    num_classes = len(config["class_names"])
    counts = np.zeros(num_classes, dtype=np.int64)
    total_px = 0
    for image_id in kept[:50]:
        m = np.array(Image.open(os.path.join(out_mask_dir, f"{image_id}.png")))
        for c in range(num_classes):
            counts[c] += (m == c).sum()
        total_px += m.size
    coverage = {name: round(counts[i] / total_px, 4) for i, name in enumerate(config["class_names"])}
    print(f"Sanity check over first 50 images (pixel coverage): {coverage}")

    return {
        "dataset_key": dataset_key,
        "n_total": len(kept),
        "n_train": n_train,
        "n_val": len(val_ids),
        "class_names": config["class_names"],
        "images_dir": out_image_dir,
        "masks_dir": out_mask_dir,
        "split_manifest": split_manifest_path,
    }


if __name__ == "__main__":
    import sys
    keys = sys.argv[1:] or list(DATASETS.keys())
    for key in keys:
        build_dataset(key)
