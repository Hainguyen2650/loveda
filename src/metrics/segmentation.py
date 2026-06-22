from __future__ import annotations

import torch


class SegmentationMetricTracker:
    def __init__(self, num_classes: int, ignore_index: int = 0) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        valid_mask = targets != self.ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        if preds.numel() == 0:
            return
        indices = targets * self.num_classes + preds
        cm = torch.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += cm.view(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict[str, float]:
        cm = self.confusion_matrix.float()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))

        valid_classes = [idx for idx in range(self.num_classes) if idx != self.ignore_index]
        miou = iou[valid_classes].mean().item() if valid_classes else 0.0

        total = cm.sum().clamp_min(1.0)
        pixel_acc = tp.sum().item() / total.item()
        return {
            "miou": miou,
            "pixel_accuracy": pixel_acc,
        }
