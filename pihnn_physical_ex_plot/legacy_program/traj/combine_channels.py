"""
Image-channel composition utility.

Example:
    python combine_channels.py \
        --img1 D:\\semixcode\\hnn_exp\\6\\traj\\composite_image_1.jpg \
        --img2 D:\\semixcode\\hnn_exp\\6\\traj\\composite_image_2.jpg \
        --out D:\\semixcode\\hnn_exp\\6\\traj\\composed_output.jpg

The script extracts the red channel from ``img1`` and the green and blue
channels from ``img2``. If the image sizes differ, ``img2`` is resized to
match ``img1``. The result is saved as a new RGB image.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def combine_channels(img1: Image.Image, img2: Image.Image) -> Image.Image:

    if img2.size != img1.size:
        img2 = img2.resize(img1.size, resample=Image.LANCZOS)

    a = np.array(img1)  # H x W x 3
    b = np.array(img2)


    r = a[:, :, 0]
    g = b[:, :, 1]
    bl = b[:, :, 2]

    out = np.stack([r, g, bl], axis=2).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine channels: R from img1 and G,B from img2")
    p.add_argument("--img1", type=Path, required=True, help="Path to source image for red channel")
    p.add_argument("--img2", type=Path, required=True, help="Path to source image for green+blue channels")
    p.add_argument("--out", type=Path, default=Path("composed_output.jpg"), help="Output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.img1.exists():
        raise FileNotFoundError(f"img1 not found: {args.img1}")
    if not args.img2.exists():
        raise FileNotFoundError(f"img2 not found: {args.img2}")

    img1 = load_rgb(args.img1)
    img2 = load_rgb(args.img2)

    out_img = combine_channels(img1, img2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(args.out, quality=95)
    print(f"Saved combined image to: {args.out}")


if __name__ == "__main__":
    main()
