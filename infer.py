from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models import SegFormerMiTB2
from src.transforms.augmentations import NormalizeTensor
from src.utils import LOVE_DA_CLASS_COLORS, LOVE_DA_IGNORE_INDEX, LOVE_DA_NUM_CLASSES
from src.utils.io import ensure_dir, load_normalization_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference with a SegFormer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/infer_mask.png"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mean, std = load_normalization_stats()
    normalize = NormalizeTensor(mean, std)

    image = np.array(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = normalize(tensor).unsqueeze(0).to(device)

    model = SegFormerMiTB2(num_classes=LOVE_DA_NUM_CLASSES, ignore_index=LOVE_DA_IGNORE_INDEX, use_pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["student_state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model.forward_logits(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in LOVE_DA_CLASS_COLORS.items():
        color_mask[pred == class_id] = color

    ensure_dir(args.output.parent)
    Image.fromarray(color_mask).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
