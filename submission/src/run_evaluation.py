"""Runs evaluation_cases.csv end to end, rendering the placement result
(or rejection reason) plus the predicted regions for each case.

Usage: .venv\\Scripts\\python.exe -m src.run_evaluation
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.placement import load_model, place_artwork
from src.viz import overlay_regions, rejection_placard

CASES_PATH = "evaluation_cases.csv"
ARTWORK_DIR = "artwork"
OUT_DIR = "outputs/evaluation_renders"
CHECKPOINT_PATH = "outputs/checkpoint.pt"


def render_case(case, model, cfg, out_dir):
    garment_path = case["garment_image"]  # path relative to project root
    artwork_path = os.path.join(ARTWORK_DIR, case["artwork"])

    result = place_artwork(garment_path, artwork_path, case["instruction"], model, cfg)

    garment_img = np.array(Image.open(garment_path).convert("RGB"))
    overlay = overlay_regions(garment_img, result.predicted_body, result.predicted_sleeve, result.target_region)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    axes[0].imshow(overlay)
    axes[0].axis("off")
    axes[0].set_title("predicted regions\n(blue=body, magenta=sleeve, yellow=target zone)", fontsize=8)

    if result.status == "ok":
        axes[1].imshow(np.array(result.composited_image))
        axes[1].set_title(f"{case['case_id']}: placed OK\nanchor={result.anchor_xy}", fontsize=8)
    else:
        axes[1].imshow(np.array(rejection_placard(garment_img.shape[1::-1], result.reason)))
        axes[1].set_title(f"{case['case_id']}: REJECTED", fontsize=8, color="darkred")
    axes[1].axis("off")

    plt.suptitle(f"{case['case_id']}  |  garment={case['garment_image']}  "
                 f"artwork={case['artwork']}  instruction='{case['instruction']}'", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{case['case_id']}.png")
    plt.savefig(out_path, dpi=110)
    plt.close(fig)

    return result, out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_model(CHECKPOINT_PATH)

    with open(CASES_PATH) as f:
        cases = list(csv.DictReader(f))

    summary_rows = []
    for case in cases:
        result, out_path = render_case(case, model, cfg, OUT_DIR)
        print(f"{case['case_id']}: status={result.status}  reason={result.reason}  -> {out_path}")
        summary_rows.append({
            "case_id": case["case_id"],
            "garment_image": case["garment_image"],
            "instruction": case["instruction"],
            "status": result.status,
            "reason": result.reason,
            "anchor_xy": result.anchor_xy,
            "artwork_size_px": result.artwork_size_px,
        })

    summary_path = os.path.join(OUT_DIR, "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    n_ok = sum(1 for r in summary_rows if r["status"] == "ok")
    print(f"\n{n_ok}/{len(summary_rows)} cases placed successfully. Summary: {summary_path}")


if __name__ == "__main__":
    main()
