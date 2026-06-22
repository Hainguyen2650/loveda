from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 0, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask_f = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_f
        one_hot = one_hot * valid_mask_f

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        denominator = torch.sum(probs + one_hot, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)

        if self.ignore_index < num_classes:
            keep = [idx for idx in range(num_classes) if idx != self.ignore_index]
            dice = dice[keep]

        return 1.0 - dice.mean()


class SupervisedSegmentationLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = 0,
        class_weights: torch.Tensor | None = None,
        use_dice: bool = False,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss(ignore_index=ignore_index) if use_dice else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ce.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0
        loss = self.ce(logits, targets)
        if self.use_dice and self.dice is not None:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        return loss


def build_pseudo_mask(
    teacher_logits: torch.Tensor,
    confidence_threshold: float,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(teacher_logits, dim=1)
    confidence, pseudo_targets = probs.max(dim=1)
    keep_mask = confidence >= confidence_threshold
    if valid_mask is not None:
        keep_mask = keep_mask & valid_mask.bool()
    return pseudo_targets, keep_mask


def masked_consistency_ce(
    student_logits: torch.Tensor,
    pseudo_targets: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    if not keep_mask.any():
        return student_logits.sum() * 0.0
    per_pixel = F.cross_entropy(student_logits, pseudo_targets, reduction="none")
    keep_mask_f = keep_mask.float()
    denom = keep_mask_f.sum().clamp_min(1.0)
    return (per_pixel * keep_mask_f).sum() / denom
