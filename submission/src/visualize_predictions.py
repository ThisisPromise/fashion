"""Shows ground truth vs. model prediction for random (or specific) images.

Usage:
    .venv\\Scripts\\python.exe -m src.visualize_predictions
    .venv\\Scripts\\python.exe -m src.visualize_predictions --split train --n 8
    .venv\\Scripts\\python.exe -m src.visualize_predictions --image-id 17313

Also saves to outputs/predictions_preview.png. Blue = body, magenta = sleeve.
"""

import argparse
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.placement import load_model, predict_regions
from src.viz import overlay_regions

CHECKPOINT_PATH = "outputs/checkpoint.pt"
SPLIT_MANIFEST_PATH = "data/processed/split_manifest.csv"
OUT_PATH = "outputs/predictions_preview.png"


def load_ids(split):
    ids = []
    with open(SPLIT_MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                ids.append(row["image_id"])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=6, help="number of images to show")
    parser.add_argument("--seed", type=int, default=None, help="omit for a different random sample each run")
    parser.add_argument("--image-id", default=None, help="show one specific image id instead of a random sample")
    args = parser.parse_args()

    model, cfg = load_model(CHECKPOINT_PATH)

    if args.image_id:
        sample = [args.image_id]
    else:
        ids = load_ids(args.split)
        if args.seed is not None:
            random.seed(args.seed)
        sample = random.sample(ids, min(args.n, len(ids)))

    fig, axes = plt.subplots(3, len(sample), figsize=(3.3 * len(sample), 10), squeeze=False)

    for i, sid in enumerate(sample):
        img = Image.open(f"data/processed/images/{sid}.jpg").convert("RGB")
        gt = np.array(Image.open(f"data/processed/masks/{sid}.png"))
        img_arr = np.array(img)

        body_pred, sleeve_pred = predict_regions(model, cfg, img)

        axes[0, i].imshow(img_arr)
        axes[0, i].set_title(f"{sid}\noriginal", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay_regions(img_arr, gt == 1, gt == 2))
        axes[1, i].set_title("ground truth", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(overlay_regions(img_arr, body_pred, sleeve_pred))
        axes[2, i].set_title("model prediction", fontsize=9)
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=110)
    print(f"Saved to {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
