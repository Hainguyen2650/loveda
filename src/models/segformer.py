from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _pick_cache_root() -> Path | None:
    candidates = [
        _REPO_ROOT / ".cache",
        Path("/dev/shm/computer-vision-cache"),
        Path.home() / ".cache" / "computer-vision",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except OSError:
            continue
    return None


_LOCAL_CACHE_ROOT = _pick_cache_root()
if _LOCAL_CACHE_ROOT is not None:
    _LOCAL_TMPDIR = _LOCAL_CACHE_ROOT / "tmp"
    _LOCAL_HF_HOME = _LOCAL_CACHE_ROOT / "huggingface"
    _LOCAL_TORCHINDUCTOR = _LOCAL_CACHE_ROOT / "torchinductor"
    for cache_dir in (_LOCAL_TMPDIR, _LOCAL_HF_HOME, _LOCAL_TORCHINDUCTOR):
        cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", str(_LOCAL_TMPDIR))
    os.environ.setdefault("TEMP", str(_LOCAL_TMPDIR))
    os.environ.setdefault("TMP", str(_LOCAL_TMPDIR))
    os.environ.setdefault("HF_HOME", str(_LOCAL_HF_HOME))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_LOCAL_TORCHINDUCTOR))

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.utils.constants import (
    LOVE_DA_IGNORE_INDEX,
    LOVE_DA_NUM_CLASSES,
    SEGFORMER_MIT_B2_CONFIG,
)


class SegFormerMiTB2(nn.Module):
    def __init__(
        self,
        num_classes: int = LOVE_DA_NUM_CLASSES,
        ignore_index: int = LOVE_DA_IGNORE_INDEX,
        pretrained_model_name_or_path: str | None = None,
        use_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        config = SegformerConfig(
            num_labels=num_classes,
            semantic_loss_ignore_index=ignore_index,
            **SEGFORMER_MIT_B2_CONFIG,
        )

        if use_pretrained and pretrained_model_name_or_path is not None:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_model_name_or_path,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
            except Exception:
                self.model = SegformerForSemanticSegmentation(config)
        else:
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return {
            "logits": logits,
            "loss": None,
        }

    @torch.no_grad()
    def forward_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.eval()
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


def create_teacher_from_student(student: nn.Module) -> nn.Module:
    teacher = deepcopy(student)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(student: nn.Module, teacher: nn.Module, momentum: float = 0.99) -> None:
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)
