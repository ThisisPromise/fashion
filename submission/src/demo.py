"""Try the placement engine on any garment photo + artwork PNG.

Usage:
    .venv\\Scripts\\python.exe -m src.demo --garment path\\to\\photo.jpg \\
        --artwork artwork\\badge_circle.png --instruction "front, left chest"

Shows the original photo, predicted regions (blue=body, magenta=sleeve,
yellow=target zone), and the result (or the rejection reason). Also saves
to outputs\\demo_output.png.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.placement import load_model, place_artwork
from src.viz import overlay_regions, rejection_placard

CHECKPOINT_PATH = "outputs/checkpoint.pt"
OUT_PATH = "outputs/demo_output.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--garment", required=True, help="path to a garment photo")
    parser.add_argument("--artwork", required=True, help="path to a transparent artwork/logo PNG")
    parser.add_argument("--instruction", required=True, help='e.g. "front, left chest"')
    args = parser.parse_args()

    model, cfg = load_model(CHECKPOINT_PATH)
    result = place_artwork(args.garment, args.artwork, args.instruction, model, cfg)

    garment_img = np.array(Image.open(args.garment).convert("RGB"))
    overlay = overlay_regions(garment_img, result.predicted_body, result.predicted_sleeve, result.target_region)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    axes[0].imshow(garment_img)
    axes[0].set_title("original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("predicted regions\n(blue=body, magenta=sleeve, yellow=target zone)", fontsize=9)
    axes[1].axis("off")

    if result.status == "ok":
        axes[2].imshow(np.array(result.composited_image))
        axes[2].set_title(f"result: placed OK\nanchor={result.anchor_xy}", fontsize=9)
    else:
        axes[2].imshow(np.array(rejection_placard(garment_img.shape[1::-1], result.reason)))
        axes[2].set_title("result: REJECTED", color="darkred", fontsize=9)
    axes[2].axis("off")

    plt.suptitle(f"instruction: '{args.instruction}'")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=110)
    print(f"status={result.status}  reason={result.reason}")
    print(f"Saved to {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
