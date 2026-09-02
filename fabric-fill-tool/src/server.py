"""Local dev server for the fabric-fill tool.

Two segmentation pipelines behind one shared response shape, so the
frontend never needs to know which one produced a given image:
  - flats_segmentation: classical CV (threshold, close gaps, flood-fill)
    for clean line-art sketches/technical flats.
  - photo_segmentation: the trained neural model from the submission
    project (body/sleeve), for real garment photographs.
Both get downscaled for the browser the same way: base image with
LANCZOS, label map with NEAREST so region ids never blend at a boundary.
"""

import base64
import io
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import flats_segmentation
import photo_segmentation

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(TOOL_DIR)
STATIC_DIR = os.path.join(TOOL_DIR, "static")

TARGET_WIDTH = 460
MAX_UPLOAD_DIMENSION = 2200  # guard against huge phone-camera uploads slowing segmentation

FLAT_PRESETS = {
    "sketch": os.path.join(PROJECT_ROOT, "sketch.jpg"),
    "sketch1": os.path.join(PROJECT_ROOT, "sketch1.png"),
}
PHOTO_PRESETS = {
    "photo1": os.path.join(TOOL_DIR, "sample_photos", "photo1.jpg"),
    "photo2": os.path.join(TOOL_DIR, "sample_photos", "photo2.jpg"),
}
# photo2 is a full-body runway shot -- trousers are actually visible in it,
# so it doubles as a lowerbody test case. photo1 is cropped at the waist,
# no legs in frame, so it's upperbody-only.
PHOTO_AREA_CHECKPOINTS = {
    "upperbody": photo_segmentation.UPPERBODY_CHECKPOINT,
    "lowerbody": photo_segmentation.LOWERBODY_CHECKPOINT,
}

app = Flask(__name__, static_folder=None)


def to_data_uri(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_asset(img, regions):
    """Shared by both pipelines: bake {region_id: mask} into a downscaled
    label map + base image, in the exact response shape the frontend
    expects, regardless of which segmentation produced `regions`."""
    label_map_full = np.zeros((img.height, img.width), dtype=np.uint8)
    for i, (region_id, mask) in enumerate(sorted(regions.items(), key=lambda kv: -kv[1].sum()), start=1):
        if i > 255:
            break
        label_map_full[mask] = i

    scale = TARGET_WIDTH / img.width
    new_size = (TARGET_WIDTH, max(1, round(img.height * scale)))

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
        "region_count": min(len(regions), 255),
    }


def _clamp_size(img):
    if max(img.size) > MAX_UPLOAD_DIMENSION:
        scale = MAX_UPLOAD_DIMENSION / max(img.size)
        return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def segment_flat(pil_img):
    img = _clamp_size(pil_img.convert("RGB"))
    tmp_path = os.path.join(SRC_DIR, "_upload_tmp_flat.png")
    img.save(tmp_path)
    try:
        regions, _line_mask = flats_segmentation.extract_regions(tmp_path)
    finally:
        os.remove(tmp_path)
    return build_asset(img, regions)


def segment_photo(pil_img, checkpoint_path=photo_segmentation.DEFAULT_CHECKPOINT):
    img = _clamp_size(pil_img.convert("RGB"))
    tmp_path = os.path.join(SRC_DIR, "_upload_tmp_photo.png")
    img.save(tmp_path)
    try:
        regions = photo_segmentation.extract_regions(tmp_path, checkpoint_path)
    finally:
        os.remove(tmp_path)
    return build_asset(img, regions)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/preset/<name>")
def api_preset(name):
    path = FLAT_PRESETS.get(name)
    if not path or not os.path.exists(path):
        return jsonify({"error": "unknown preset"}), 404
    img = Image.open(path)
    img.load()
    return jsonify(segment_flat(img))


@app.route("/api/preset_photo/<name>")
def api_preset_photo(name):
    path = PHOTO_PRESETS.get(name)
    if not path or not os.path.exists(path):
        return jsonify({"error": "unknown preset"}), 404
    area = request.args.get("area", "upperbody")
    checkpoint_path = PHOTO_AREA_CHECKPOINTS.get(area, photo_segmentation.DEFAULT_CHECKPOINT)
    img = Image.open(path)
    img.load()
    return jsonify(segment_photo(img, checkpoint_path))


@app.route("/api/segment", methods=["POST"])
def api_segment():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no file uploaded"}), 400
    try:
        img = Image.open(file.stream)
        img.load()
    except Exception:
        return jsonify({"error": "could not read that as an image"}), 400
    return jsonify(segment_flat(img))


@app.route("/api/segment_photo", methods=["POST"])
def api_segment_photo():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no file uploaded"}), 400
    try:
        img = Image.open(file.stream)
        img.load()
    except Exception:
        return jsonify({"error": "could not read that as an image"}), 400
    area = request.args.get("area", "upperbody")
    checkpoint_path = PHOTO_AREA_CHECKPOINTS.get(area, photo_segmentation.DEFAULT_CHECKPOINT)
    return jsonify(segment_photo(img, checkpoint_path))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
