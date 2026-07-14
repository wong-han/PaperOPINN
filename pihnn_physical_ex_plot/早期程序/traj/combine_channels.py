"""
合成图像通道工具

用法示例:
    python combine_channels.py \
        --img1 D:\\semixcode\\hnn_exp\\6\\traj\\composite_image_1.jpg \
        --img2 D:\\semixcode\\hnn_exp\\6\\traj\\composite_image_2.jpg \
        --out D:\\semixcode\\hnn_exp\\6\\traj\\composed_output.jpg

脚本行为：
- 从 img1 提取红色通道（R）
- 从 img2 提取绿色与蓝色通道（G,B）
- 如果两张图像大小不同，会把 img2 缩放到 img1 的尺寸（保持 img1 为基准）
- 输出一张新的 RGB 图像
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
    # 保证 img2 与 img1 大小一致（以 img1 为基准）
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, resample=Image.LANCZOS)

    a = np.array(img1)  # H x W x 3
    b = np.array(img2)

    # 提取通道，确保 dtype 为 uint8
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
