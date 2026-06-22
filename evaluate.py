from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import LoveDADataset
from src.metrics.segmentation import SegmentationMetricTracker
from src.models import SegFormerMiTB2
from src.transforms.augmentations import CenterCropPair, NormalizeTensor
from src.utils import LOVE_DA_IGNORE_INDEX, LOVE_DA_NUM_CLASSES, LOVE_DA_ROOT
from src.utils.io import load_normalization_stats


def sanitize_mask_indices(mask: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    mask = mask.clone()
    invalid = (mask < 0) | (mask >= num_classes)
    if invalid.any():
        mask[invalid] = ignore_index
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SegFormer checkpoint on LoveDA.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path(LOVE_DA_ROOT))
    parser.add_argument("--split", type=str, default="Val")
    parser.add_argument("--domain", type=str, default="Rural")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mean, std = load_normalization_stats()

    dataset = LoveDADataset(
        dataset_root=args.dataset_root,
        splits=(args.split,),
        domains=(args.domain,),
        require_masks=True,
        ignore_black_padding=True,
        joint_transform=CenterCropPair(crop_size=args.crop_size),
        image_transform=NormalizeTensor(mean, std),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SegFormerMiTB2(num_classes=LOVE_DA_NUM_CLASSES, ignore_index=LOVE_DA_IGNORE_INDEX, use_pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["student_state_dict"])
    model.eval()

    metric = SegmentationMetricTracker(num_classes=LOVE_DA_NUM_CLASSES, ignore_index=LOVE_DA_IGNORE_INDEX)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            masks = masks.clone()
            masks[~valid_mask] = LOVE_DA_IGNORE_INDEX
            masks = sanitize_mask_indices(masks, LOVE_DA_NUM_CLASSES, LOVE_DA_IGNORE_INDEX)
            logits = model.forward_logits(images)
            metric.update(logits, masks)

    metrics = metric.compute()
    print(metrics)


if __name__ == "__main__":
    main()
