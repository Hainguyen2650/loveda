from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import ColorJitter, GaussianBlur


@dataclass
class RandomCropFlip:
    crop_size: int = 512
    hflip_prob: float = 0.5
    vflip_prob: float = 0.0
    min_valid_ratio: float = 0.5
    max_tries: int = 10

    def _sample_crop(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int]:
        height, width = image.shape[:2]
        if height <= self.crop_size or width <= self.crop_size:
            return 0, 0

        best_top, best_left = 0, 0
        best_ratio = -1.0
        for _ in range(self.max_tries):
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            image_crop = image[top : top + self.crop_size, left : left + self.crop_size]
            image_ratio = float(np.any(image_crop != 0, axis=2).mean())
            if mask is not None:
                mask_crop = mask[top : top + self.crop_size, left : left + self.crop_size]
                label_ratio = float((mask_crop != 0).mean())
                ratio = min(image_ratio, label_ratio)
            else:
                ratio = image_ratio
            if ratio >= self.min_valid_ratio:
                return top, left
            if ratio > best_ratio:
                best_ratio = ratio
                best_top, best_left = top, left
        return best_top, best_left

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        height, width = image.shape[:2]
        if height > self.crop_size and width > self.crop_size:
            top, left = self._sample_crop(image, mask)
            image = image[top : top + self.crop_size, left : left + self.crop_size]
            if mask is not None:
                mask = mask[top : top + self.crop_size, left : left + self.crop_size]

        if random.random() < self.hflip_prob:
            image = np.flip(image, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()

        if random.random() < self.vflip_prob:
            image = np.flip(image, axis=0).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()

        return image, mask


@dataclass
class CenterCropPair:
    crop_size: int = 512

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        height, width = image.shape[:2]
        crop_h = min(self.crop_size, height)
        crop_w = min(self.crop_size, width)
        top = max((height - crop_h) // 2, 0)
        left = max((width - crop_w) // 2, 0)
        image = image[top : top + crop_h, left : left + crop_w]
        if mask is not None:
            mask = mask[top : top + crop_h, left : left + crop_w]
        return image, mask


class NormalizeTensor:
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=image.device, dtype=image.dtype)
        std = self.std.to(device=image.device, dtype=image.dtype)
        return (image - mean) / std


class TargetViewGenerator:
    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        weak_jitter_strength: float = 0.1,
        strong_jitter_strength: float = 0.4,
    ) -> None:
        self.normalize = NormalizeTensor(mean, std)
        self.weak_jitter = ColorJitter(
            brightness=weak_jitter_strength,
            contrast=weak_jitter_strength,
            saturation=weak_jitter_strength,
            hue=min(0.05, weak_jitter_strength / 2.0),
        )
        self.strong_jitter = ColorJitter(
            brightness=strong_jitter_strength,
            contrast=strong_jitter_strength,
            saturation=strong_jitter_strength,
            hue=min(0.1, strong_jitter_strength / 3.0),
        )
        self.strong_blur = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    def weak(self, image: torch.Tensor) -> torch.Tensor:
        image = self.weak_jitter(image)
        return self.normalize(image)

    def strong(self, image: torch.Tensor) -> torch.Tensor:
        image = self.strong_jitter(image)
        if random.random() < 0.5:
            image = self.strong_blur(image)
        return self.normalize(image)


class TargetSampleTransform:
    def __init__(self, view_generator: TargetViewGenerator, keep_raw_image: bool = False) -> None:
        self.view_generator = view_generator
        self.keep_raw_image = keep_raw_image

    def __call__(self, sample: dict[str, object]) -> dict[str, object]:
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected sample['image'] to be a torch.Tensor.")
        sample["weak_image"] = self.view_generator.weak(image)
        sample["strong_image"] = self.view_generator.strong(image)
        if not self.keep_raw_image:
            sample.pop("image", None)
        return sample
