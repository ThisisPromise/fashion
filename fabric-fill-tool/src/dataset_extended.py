"""Same dataset logic as submission/src/dataset.py -- geometric+photometric
augmentation, NEAREST resize for masks -- just parameterized by directory
and image size instead of hardcoded module constants, so one class serves
both the upperbody_extended and lowerbody datasets."""

import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_split_ids(split_manifest_path, split):
    ids = []
    with open(split_manifest_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                ids.append(row["image_id"])
    return ids


class GarmentSegDataset(Dataset):
    def __init__(self, data_dir, split, image_size=320, augment=None):
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.split_manifest_path = os.path.join(data_dir, "split_manifest.csv")
        self.ids = load_split_ids(self.split_manifest_path, split)
        self.image_size = image_size
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = Image.open(os.path.join(self.image_dir, f"{image_id}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{image_id}.png"))

        if self.augment:
            img, mask = self._augment(img, mask)
        else:
            img = TF.resize(img, [self.image_size, self.image_size])
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img_t, mask_t, image_id

    def _augment(self, img, mask):
        size = self.image_size

        scale = random.uniform(0.85, 1.0)
        w, h = img.size
        crop_w, crop_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        img = img.crop((left, top, left + crop_w, top + crop_h))
        mask = mask.crop((left, top, left + crop_w, top + crop_h))

        img = TF.resize(img, [size, size])
        mask = TF.resize(mask, [size, size], interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))

        return img, mask
