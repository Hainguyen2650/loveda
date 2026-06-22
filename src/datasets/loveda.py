from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.constants import (
    LOVE_DA_DOMAINS,
    LOVE_DA_IMAGE_SUBDIR,
    LOVE_DA_MASK_SPLITS,
    LOVE_DA_MASK_SUBDIR,
    LOVE_DA_ROOT,
    LOVE_DA_SPLITS,
)


@dataclass(frozen=True)
class LoveDARecord:
    split: str
    domain: str
    sample_id: str
    image_path: Path
    mask_path: Path | None


def _validate_choices(name: str, values: tuple[str, ...], allowed: tuple[str, ...]) -> tuple[str, ...]:
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported {name}: {invalid}. Allowed values: {allowed}")
    return values


def build_loveda_records(
    dataset_root: str | Path = LOVE_DA_ROOT,
    splits: tuple[str, ...] | None = None,
    domains: tuple[str, ...] | None = None,
    require_masks: bool = False,
) -> list[LoveDARecord]:
    root = Path(dataset_root)
    split_values = _validate_choices("splits", splits or LOVE_DA_SPLITS, LOVE_DA_SPLITS)
    domain_values = _validate_choices("domains", domains or LOVE_DA_DOMAINS, LOVE_DA_DOMAINS)

    records: list[LoveDARecord] = []
    for split in split_values:
        for domain in domain_values:
            image_dir = root / split / domain / LOVE_DA_IMAGE_SUBDIR
            mask_dir = root / split / domain / LOVE_DA_MASK_SUBDIR
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("*.png")):
                sample_id = image_path.stem
                mask_path = mask_dir / f"{sample_id}.png"
                if not mask_path.exists():
                    mask_path = None
                if require_masks and mask_path is None:
                    continue
                records.append(
                    LoveDARecord(
                        split=split,
                        domain=domain,
                        sample_id=sample_id,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )
    return records


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.int64))


class LoveDADataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path = LOVE_DA_ROOT,
        splits: tuple[str, ...] | None = None,
        domains: tuple[str, ...] | None = None,
        require_masks: bool = False,
        ignore_black_padding: bool = False,
        joint_transform: Callable[[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]] | None = None,
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mask_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        sample_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        return_meta: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.records = build_loveda_records(
            dataset_root=self.dataset_root,
            splits=splits,
            domains=domains,
            require_masks=require_masks,
        )
        if not self.records:
            raise ValueError("No LoveDA records found for the requested split/domain configuration.")

        self.require_masks = require_masks
        self.ignore_black_padding = ignore_black_padding
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.sample_transform = sample_transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]

        image = np.array(Image.open(record.image_path).convert("RGB"), dtype=np.uint8)
        mask = None
        if record.mask_path is not None:
            mask = np.array(Image.open(record.mask_path), dtype=np.uint8)

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        image_tensor = image_to_tensor(image)
        if self.ignore_black_padding:
            valid_mask = torch.from_numpy(np.any(image != 0, axis=2)).to(torch.bool)
        else:
            valid_mask = torch.ones((image.shape[0], image.shape[1]), dtype=torch.bool)

        if self.image_transform is not None:
            image_tensor = self.image_transform(image_tensor)

        sample: dict[str, object] = {
            "image": image_tensor,
            "valid_mask": valid_mask,
        }

        if mask is not None:
            mask_tensor = mask_to_tensor(mask)
            if self.mask_transform is not None:
                mask_tensor = self.mask_transform(mask_tensor)
            sample["mask"] = mask_tensor
        elif self.require_masks:
            raise RuntimeError(f"Mask required but missing for sample {record.sample_id}")

        if self.return_meta:
            sample["meta"] = {
                "split": record.split,
                "domain": record.domain,
                "sample_id": record.sample_id,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path) if record.mask_path is not None else None,
                "has_mask": record.mask_path is not None,
                "is_mask_split": record.split in LOVE_DA_MASK_SPLITS,
            }

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample
