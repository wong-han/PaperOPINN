"""
保存并合成图像脚本

功能：
- 将两张原图拷贝到 `saved/` 子目录
- 从 img1 提取红色通道并保存为 `*_R.png`
- 从 img2 提取绿+蓝通道并保存为 `*_GB.png`（R通道置0）
- 生成合成图（R from img1, G+B from img2）并保存为 `composed_output.jpg`

使用默认路径（与之前相同），也支持命令行参数覆盖。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from PIL import Image
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img1', type=Path, default=Path(r'D:\semixcode\hnn_exp\6\traj\composite_image_1.jpg'))
    p.add_argument('--img2', type=Path, default=Path(r'D:\semixcode\hnn_exp\6\traj\composite_image_2.jpg'))
    p.add_argument('--out-dir', type=Path, default=Path(r'D:\semixcode\hnn_exp\6\traj\saved'))
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_channel_images(img1_path: Path, img2_path: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    ensure_dir(out_dir)
    if not img1_path.exists():
        raise FileNotFoundError(f"img1 not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"img2 not found: {img2_path}")

    # copy originals
    dest1 = out_dir / img1_path.name
    dest2 = out_dir / img2_path.name
    shutil.copy2(img1_path, dest1)
    shutil.copy2(img2_path, dest2)

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # resize img2 to img1 size if needed
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, resample=Image.LANCZOS)

    a = np.array(img1)
    b = np.array(img2)

    # R image (from img1) - save as RGB where R=original, G=B=0
    r = a[:, :, 0]
    zeros = np.zeros_like(r)
    r_rgb = np.stack([r, zeros, zeros], axis=2).astype(np.uint8)
    r_img = Image.fromarray(r_rgb, mode='RGB')
    r_path = out_dir / (img1_path.stem + '_R.png')
    r_img.save(r_path)

    # GB image (from img2) - set R=0
    gb = np.stack([np.zeros_like(b[:, :, 0]), b[:, :, 1], b[:, :, 2]], axis=2).astype(np.uint8)
    gb_img = Image.fromarray(gb)
    gb_path = out_dir / (img2_path.stem + '_GB.png')
    gb_img.save(gb_path)

    # composed color image R(from img1), G,B(from img2)
    composed = np.stack([r, b[:, :, 1], b[:, :, 2]], axis=2).astype(np.uint8)
    comp_img = Image.fromarray(composed)
    comp_path = out_dir / 'composed_output.jpg'
    comp_img.save(comp_path, quality=95)

    return dest1, dest2, r_path, gb_path, comp_path


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    print(f"Using img1: {args.img1}")
    print(f"Using img2: {args.img2}")
    saved_orig1, saved_orig2, r_path, gb_path, comp_path = save_channel_images(args.img1, args.img2, args.out_dir)
    print('Saved originals to:', saved_orig1, saved_orig2)
    print('Saved R-channel image:', r_path)
    print('Saved GB-channel image:', gb_path)
    print('Saved composed image:', comp_path)

if __name__ == '__main__':
    main()
