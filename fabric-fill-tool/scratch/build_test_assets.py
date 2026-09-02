"""Generates a synthetic fashion flat and a synthetic fabric swatch, purely to
validate the segmentation + fill pipeline before real assets exist.

The flat is deliberately simple, axis-aligned rectangles so region boundaries
are guaranteed to align exactly: torso (split by a center "zip" line into
left/right), a collar flush on top, a sleeve flush on each side.
"""

import random

from PIL import Image, ImageDraw

W, H = 600, 700


def build_flat():
    img = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(img)
    line_width = 4

    torso = (200, 150, 400, 550)
    collar = (260, 100, 340, 150)
    left_sleeve = (100, 150, 200, 350)
    right_sleeve = (400, 150, 500, 350)

    draw.rectangle(torso, outline=0, width=line_width)
    draw.rectangle(collar, outline=0, width=line_width)
    draw.rectangle(left_sleeve, outline=0, width=line_width)
    draw.rectangle(right_sleeve, outline=0, width=line_width)
    draw.line([(300, 150), (300, 550)], fill=0, width=line_width)  # zip, splits torso

    img.save("fabric-fill-tool/scratch/sample_flat.png")
    print("wrote fabric-fill-tool/scratch/sample_flat.png")


def build_fabric_swatch():
    random.seed(0)
    size = 80
    img = Image.new("RGB", (size, size), color=(220, 40, 40))
    draw = ImageDraw.Draw(img)
    colors = [(255, 210, 0), (10, 90, 160), (20, 140, 60)]
    for i in range(6):
        c = colors[i % len(colors)]
        x, y = random.randint(0, size), random.randint(0, size)
        r = random.randint(10, 22)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=c)
    img.save("fabric-fill-tool/scratch/sample_fabric.png")
    print("wrote fabric-fill-tool/scratch/sample_fabric.png")


if __name__ == "__main__":
    build_flat()
    build_fabric_swatch()
