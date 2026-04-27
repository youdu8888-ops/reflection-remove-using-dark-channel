from __future__ import annotations

from pathlib import Path
import argparse
import shutil
from typing import Optional

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "datasets" / "raw_data"
PROCESSED_ROOT = ROOT / "datasets" / "processed_data"
CROP_SIZE = 224


def resolve_real89_src(override: Optional[Path]) -> Path:
    """Prefer explicit path, then <repo>/real89, then datasets/raw_data/real89."""
    candidates = []
    if override is not None:
        candidates.append(override.resolve())
    candidates.append(ROOT / "real89")
    candidates.append(RAW_ROOT / "real89")
    for p in candidates:
        if p.is_dir() and (p / "blended").is_dir() and (p / "transmission_layer").is_dir():
            return p
    raise FileNotFoundError(
        "未找到成对 real 数据。请将包含 blended/ 与 transmission_layer/ 的目录放在以下其一：\n"
        f"  {ROOT / 'real89'}\n  或 {RAW_ROOT / 'real89'}\n"
        "或使用: python datasets/prepare_train_data.py --real-only --real-src D:\\path\\to\\real89"
    )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def center_crop_224(img: Image.Image) -> Image.Image:
    width, height = img.size
    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError(f"Image is smaller than {CROP_SIZE}x{CROP_SIZE}: {width}x{height}")
    left = (width - CROP_SIZE) // 2
    top = (height - CROP_SIZE) // 2
    return img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))


def prepare_voc() -> int:
    list_path = ROOT / "VOC2012_224_train_png.txt"
    src_dir = RAW_ROOT / "VOCdevkit" / "VOC2012" / "JPEGImages"
    dst_dir = PROCESSED_ROOT / "VOCdevkit" / "VOC2012" / "PNGImages"

    if not list_path.exists():
        raise FileNotFoundError(f"Missing list file: {list_path}")
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing raw VOC directory: {src_dir}")

    reset_dir(dst_dir)
    names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]

    for index, png_name in enumerate(names, start=1):
        src_path = src_dir / png_name.replace(".png", ".jpg")
        dst_path = dst_dir / png_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing raw VOC image: {src_path}")
        with Image.open(src_path) as img:
            center_crop_224(img.convert("RGB")).save(dst_path)
        if index % 2000 == 0 or index == len(names):
            print(f"[VOC] {index}/{len(names)} processed")

    return len(names)


def prepare_real_train(src_dir: Optional[Path] = None) -> tuple[int, int]:
    """Copy real89 (blended + transmission_layer) into processed_data/real_train for training."""
    src_dir = src_dir or resolve_real89_src(None)
    dst_dir = PROCESSED_ROOT / "real_train"

    if not (src_dir / "blended").exists() or not (src_dir / "transmission_layer").exists():
        raise FileNotFoundError(f"Expected 'blended' and 'transmission_layer' under {src_dir}")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    blended = len([p for p in (dst_dir / "blended").iterdir() if p.is_file()])
    transmission = len([p for p in (dst_dir / "transmission_layer").iterdir() if p.is_file()])
    return blended, transmission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备 ERRNet 训练数据：VOC 合成用 PNG，或从 real89 复制成对真实数据到 processed_data/real_train。"
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="只处理 real89 → processed_data/real_train（不需要 VOC 原文）",
    )
    parser.add_argument(
        "--real-src",
        type=Path,
        default=None,
        help="成对数据根目录（含 blended/ 与 transmission_layer/）；默认用仓库根下 real89 或 datasets/raw_data/real89",
    )
    args = parser.parse_args()

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    if args.real_only:
        src = resolve_real89_src(args.real_src)
        print(f"[i] 使用源目录: {src}")
        real_blended, real_transmission = prepare_real_train(src)
        print("\nDone (real only).")
        print(f"real_train/blended: {real_blended}")
        print(f"real_train/transmission_layer: {real_transmission}")
        print(f"输出目录: {PROCESSED_ROOT / 'real_train'}")
        return

    voc_count = prepare_voc()
    real_src = resolve_real89_src(args.real_src)
    print(f"[i] 使用源目录: {real_src}")
    real_blended, real_transmission = prepare_real_train(real_src)

    print("\nDone.")
    print(f"VOC PNGImages: {voc_count}")
    print(f"real_train/blended: {real_blended}")
    print(f"real_train/transmission_layer: {real_transmission}")


if __name__ == "__main__":
    main()

