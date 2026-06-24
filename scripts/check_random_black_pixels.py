from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ["Train", "Val", "Test"]
DOMAINS = ["Urban", "Rural"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random LoveDA images and inspect RGB=(0,0,0) pixels."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/LoveDA"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/dataset/random_black_pixel_check.csv"))
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_image_paths(dataset_root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for split in SPLITS:
        for domain in DOMAINS:
            image_dir = dataset_root / split / domain / "images_png"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("*.png")):
                rows.append((split, domain, image_path))
    return rows


def summarize_black_pixels(image_path: Path) -> dict:
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    black_mask = np.all(image == 0, axis=2)
    total_pixels = int(black_mask.size)
    black_pixels = int(black_mask.sum())
    rows_with_black = np.where(black_mask.any(axis=1))[0]
    cols_with_black = np.where(black_mask.any(axis=0))[0]

    return {
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
        "total_pixels": total_pixels,
        "black_pixels": black_pixels,
        "black_ratio": black_pixels / total_pixels,
        "rows_with_black": int(rows_with_black.size),
        "cols_with_black": int(cols_with_black.size),
        "top_row": int(rows_with_black[0]) if rows_with_black.size else "",
        "bottom_row": int(rows_with_black[-1]) if rows_with_black.size else "",
        "left_col": int(cols_with_black[0]) if cols_with_black.size else "",
        "right_col": int(cols_with_black[-1]) if cols_with_black.size else "",
        "touches_top": bool(rows_with_black.size and rows_with_black[0] == 0),
        "touches_bottom": bool(rows_with_black.size and rows_with_black[-1] == image.shape[0] - 1),
        "touches_left": bool(cols_with_black.size and cols_with_black[0] == 0),
        "touches_right": bool(cols_with_black.size and cols_with_black[-1] == image.shape[1] - 1),
    }


def main() -> None:
    args = parse_args()
    all_images = collect_image_paths(args.dataset_root)
    if not all_images:
        raise SystemExit("No images found under dataset root")

    rng = random.Random(args.seed)
    sample_size = min(args.sample_size, len(all_images))
    sampled = rng.sample(all_images, sample_size)

    rows = []
    for split, domain, image_path in sampled:
        summary = summarize_black_pixels(image_path)
        rows.append(
            {
                "split": split,
                "domain": domain,
                "sample_id": image_path.stem,
                "image_path": str(image_path.resolve()),
                **summary,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} sampled-image summaries to {args.output_csv}")


if __name__ == "__main__":
    main()
