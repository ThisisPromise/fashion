"""Training entrypoint for the upperbody_extended and lowerbody datasets.
Same core recipe as submission/src/train.py -- weighted cross-entropy +
Dice loss, split learning rates for any unfrozen backbone tail, 2M
trainable-parameter cap enforced -- reusing the submission project's
model.py rather than duplicating it. Two additions on top of that recipe:
a class-balanced sampler (compute_sample_weights) so rare classes are seen
more often per epoch, and checkpoint selection by validation mean IoU
rather than validation loss, since the two can diverge and mean IoU is the
metric that's actually reported.

Usage (run from the repository root, since the config files use paths
relative to it):
    python fabric-fill-tool/src/train_extended.py --config fabric-fill-tool/configs/upperbody_extended.json
"""

import argparse
import csv
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(TOOL_DIR)
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")
sys.path.insert(0, SUBMISSION_DIR)
sys.path.insert(0, SRC_DIR)

from src.model import TinyGarmentSegModel, count_frozen_parameters, count_trainable_parameters  # noqa: E402
from dataset_extended import GarmentSegDataset  # noqa: E402


def compute_class_weights(data_dir, split_manifest_path, num_classes):
    train_ids = []
    with open(split_manifest_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train_ids.append(row["image_id"])

    mask_dir = os.path.join(data_dir, "masks")
    counts = np.zeros(num_classes, dtype=np.int64)
    for image_id in train_ids:
        m = np.array(Image.open(os.path.join(mask_dir, f"{image_id}.png")))
        for c in range(num_classes):
            counts[c] += (m == c).sum()

    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-12))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(data_dir, split_manifest_path, num_classes):
    """Per-pixel class weighting alone doesn't fix a class that's simply
    ABSENT from most training images -- an image with no zipper in it never
    gets a chance to teach the model about zipper, no matter how the loss
    is weighted. This oversamples images that actually contain a rare
    class, so the epoch sees them more often, on top of (not instead of)
    the loss weighting."""
    train_ids = []
    with open(split_manifest_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train_ids.append(row["image_id"])

    mask_dir = os.path.join(data_dir, "masks")
    presence = np.zeros((len(train_ids), num_classes), dtype=bool)
    for i, image_id in enumerate(train_ids):
        m = np.array(Image.open(os.path.join(mask_dir, f"{image_id}.png")))
        for c in range(num_classes):
            presence[i, c] = (m == c).any()

    class_image_freq = presence.mean(axis=0)  # fraction of images containing each class
    class_rarity = 1.0 / np.maximum(class_image_freq, 1e-6)

    sample_weights = np.ones(len(train_ids), dtype=np.float64)
    for c in range(2, num_classes):  # skip background (0) and body (1) -- never rare
        sample_weights = np.maximum(sample_weights, np.where(presence[:, c], class_rarity[c], 1.0))

    return train_ids, torch.tensor(sample_weights, dtype=torch.double)


def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_onehot).sum(dims)
    union = probs.sum(dims) + targets_onehot.sum(dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice_per_class.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_confusion(preds, targets, num_classes):
    mask = (targets >= 0) & (targets < num_classes)
    idx = num_classes * targets[mask].to(torch.int64) + preds[mask].to(torch.int64)
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confusion(confusion):
    tp = torch.diag(confusion).float()
    fp = confusion.sum(0).float() - tp
    fn = confusion.sum(1).float() - tp
    denom = tp + fp + fn
    return torch.where(denom > 0, tp / denom, torch.full_like(denom, float("nan")))


@torch.no_grad()
def evaluate(model, loader, num_classes, device, criterion):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    n_batches = 0
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        confusion += compute_confusion(preds.cpu(), masks.cpu(), num_classes)
    iou = iou_from_confusion(confusion)
    return total_loss / max(n_batches, 1), iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    set_seed(cfg["seed"])
    device = torch.device("cpu")

    data_dir = cfg["data_dir"]
    split_manifest_path = os.path.join(data_dir, "split_manifest.csv")

    train_ds = GarmentSegDataset(data_dir, "train", image_size=cfg["image_size"])
    val_ds = GarmentSegDataset(data_dir, "val", image_size=cfg["image_size"])

    _train_ids_for_sampler, sample_weights = compute_sample_weights(data_dir, split_manifest_path, cfg["num_classes"])
    assert _train_ids_for_sampler == train_ds.ids, "sample weight order must match dataset order"
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = TinyGarmentSegModel(
        num_classes=cfg["num_classes"], decoder_ch=cfg["decoder_channels"],
        unfreeze_last_n_backbone_layers=cfg.get("unfreeze_last_n_backbone_layers", 0),
    ).to(device)
    n_trainable = count_trainable_parameters(model)
    n_frozen = count_frozen_parameters(model)
    print(f"Trainable parameters: {n_trainable} (cap: 2,000,000)")
    print(f"Frozen (backbone) parameters: {n_frozen}")
    assert n_trainable < 2_000_000, "Trainable parameter count exceeds the 2M cap"

    backbone_params = model.backbone_trainable_parameters()
    backbone_param_ids = {id(p) for p in backbone_params}
    decoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
    optimizer = torch.optim.Adam([
        {"params": decoder_params, "lr": cfg["lr"]},
        {"params": backbone_params, "lr": cfg["lr"] * cfg.get("backbone_lr_multiplier", 0.1)},
    ])

    class_weights = compute_class_weights(data_dir, split_manifest_path, cfg["num_classes"])
    print(f"Class weights (sqrt inverse-frequency, from train split): "
          f"{dict(zip(cfg['class_names'], class_weights.tolist()))}")

    def criterion(logits, masks):
        ce = F.cross_entropy(logits, masks, weight=class_weights)
        dice = dice_loss(logits, masks, cfg["num_classes"])
        return ce + dice

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_val_miou = -1.0

    os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        val_loss, val_iou = evaluate(model, val_loader, cfg["num_classes"], device, criterion)
        val_miou = torch.nanmean(val_iou).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        print(f"epoch {epoch:02d}/{cfg['epochs']}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
              f"per_class={[round(x, 3) for x in val_iou.tolist()]}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_miou": val_miou,
                "trainable_params": n_trainable,
                "frozen_params": n_frozen,
            }, cfg["checkpoint_path"])

    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val_loss, final_val_iou = evaluate(model, val_loader, cfg["num_classes"], device, criterion)

    metrics = {
        "best_epoch": ckpt["epoch"],
        "val_loss": final_val_loss,
        "per_class_iou": {
            name: (None if np.isnan(v) else v)
            for name, v in zip(cfg["class_names"], final_val_iou.tolist())
        },
        "mean_iou": torch.nanmean(final_val_iou).item(),
        "trainable_params": n_trainable,
        "frozen_params": n_frozen,
    }
    with open(cfg["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)
    print("Final validation metrics (best checkpoint):", json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["val_miou"], label="val mIoU", color="green")
    axes[1].set_title("Validation mean IoU")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(cfg["curves_path"], dpi=120)
    print("Saved training curves to", cfg["curves_path"])


if __name__ == "__main__":
    main()
