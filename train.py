from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.datasets import LoveDADataset
from src.losses.segmentation import (
    SupervisedSegmentationLoss,
    build_pseudo_mask,
    masked_consistency_ce,
)
from src.metrics.segmentation import SegmentationMetricTracker
from src.models import SegFormerMiTB2, create_teacher_from_student, update_ema_teacher
from src.transforms.augmentations import (
    CenterCropPair,
    NormalizeTensor,
    RandomCropFlip,
    TargetSampleTransform,
    TargetViewGenerator,
)
from src.utils import LOVE_DA_IGNORE_INDEX, LOVE_DA_NUM_CLASSES, LOVE_DA_ROOT
from src.utils.io import (
    append_jsonl,
    ensure_dir,
    load_class_weights,
    load_normalization_stats,
    save_checkpoint,
    save_json,
)


DEFAULT_PRETRAINED_PATH = Path("models/segformer-b2-ade-512-512")


def configure_cuda_stability(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except AttributeError:
        pass
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban-to-Rural UDA training with SegFormer-MiT-B2.")
    parser.add_argument("--dataset-root", type=Path, default=Path(LOVE_DA_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints/uda_segformer_b2_u2r"))
    parser.add_argument("--source-domain", type=str, default="Urban")
    parser.add_argument("--target-domain", type=str, default="Rural")
    parser.add_argument("--val-domain", type=str, default="Rural")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--target-num-workers",
        type=int,
        default=1,
        help="Worker count for target-domain loading. Lower values are more stable on some local CUDA setups.",
    )
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=1,
        help="Worker count for validation loading.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--teacher-momentum", type=float, default=0.99)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument(
        "--source-only-warmup-epochs",
        type=int,
        default=0,
        help="Number of initial epochs that disable target consistency and train on source supervision only.",
    )
    parser.add_argument(
        "--consistency-rampup-epochs",
        type=int,
        default=0,
        help="Number of epochs used to linearly ramp the consistency weight after warmup.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="Optional per-step micro-batch size for CUDA stability. Use 0 to auto-select.",
    )
    parser.add_argument("--use-dice", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pretrained-model-name",
        type=str,
        default=str(DEFAULT_PRETRAINED_PATH) if DEFAULT_PRETRAINED_PATH.exists() else "",
    )
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training. Disabled by default for local stability.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases experiment logging.")
    parser.add_argument("--wandb-project", type=str, default="loveda-u2r-uda")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated wandb tags.")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=("online", "offline", "disabled"))
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="Optional debug cap for steps per epoch. Use 0 to keep the full epoch.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pin_memory. Disabled by default because some local CUDA/WSL setups are unstable with pinned host buffers.",
    )
    return parser.parse_args()


def make_dataloader(
    dataset: LoveDADataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def slice_batch(batch: dict[str, object], start: int, end: int) -> dict[str, object]:
    sliced: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[start:end]
        elif isinstance(value, dict):
            nested: dict[str, object] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, torch.Tensor):
                    nested[nested_key] = nested_value[start:end]
                elif isinstance(nested_value, list):
                    nested[nested_key] = nested_value[start:end]
                elif isinstance(nested_value, tuple):
                    nested[nested_key] = nested_value[start:end]
                else:
                    nested[nested_key] = nested_value
            sliced[key] = nested
        elif isinstance(value, list):
            sliced[key] = value[start:end]
        elif isinstance(value, tuple):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def summarize_batch_meta(batch: dict[str, object]) -> str:
    meta = batch.get("meta")
    if not isinstance(meta, dict):
        return "<meta unavailable>"
    sample_ids = meta.get("sample_id")
    image_paths = meta.get("image_path")
    return f"sample_id={sample_ids} image_path={image_paths}"


def assert_finite(name: str, tensor: torch.Tensor, batch: dict[str, object] | None = None) -> None:
    if torch.isfinite(tensor).all():
        return
    detached = tensor.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() > 0:
        min_value = float(finite.min().cpu())
        max_value = float(finite.max().cpu())
    else:
        min_value = float("nan")
        max_value = float("nan")
    batch_desc = summarize_batch_meta(batch) if batch is not None else "<batch unavailable>"
    raise RuntimeError(f"Non-finite tensor detected: {name} min={min_value} max={max_value} {batch_desc}")


def compute_consistency_scale(epoch: int, warmup_epochs: int, rampup_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if rampup_epochs <= 0:
        return 1.0
    ramp_step = epoch - warmup_epochs
    return min(1.0, ramp_step / float(rampup_epochs))


def resolve_micro_batch_size(args: argparse.Namespace, device: torch.device, batch_size: int) -> int:
    if args.micro_batch_size > 0:
        return max(1, min(args.micro_batch_size, batch_size))
    if device.type == "cuda" and batch_size > 2:
        return 2
    return batch_size


def resolve_physical_batch_size(args: argparse.Namespace, device: torch.device) -> int:
    if device.type == "cuda" and args.batch_size > 2:
        return 2
    return args.batch_size


def init_wandb(args: argparse.Namespace, config_payload: dict[str, Any]):
    if not args.use_wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb` or disable logging with `--wandb-mode disabled`."
        ) from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config=config_payload,
        tags=tags,
        mode=args.wandb_mode,
    )
    return run


def apply_invalid_ignore(mask: torch.Tensor, valid_mask: torch.Tensor, ignore_index: int) -> torch.Tensor:
    mask = mask.clone()
    mask[~valid_mask] = ignore_index
    return mask


def sanitize_mask_indices(mask: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    mask = mask.clone()
    invalid = (mask < 0) | (mask >= num_classes)
    if invalid.any():
        mask[invalid] = ignore_index
    return mask


def evaluate(
    model: SegFormerMiTB2,
    loader: DataLoader,
    device: torch.device,
    ignore_index: int,
) -> dict[str, float]:
    metric = SegmentationMetricTracker(num_classes=LOVE_DA_NUM_CLASSES, ignore_index=ignore_index)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            images = batch["image"]
            masks = apply_invalid_ignore(batch["mask"], batch["valid_mask"], ignore_index)
            masks = sanitize_mask_indices(masks, LOVE_DA_NUM_CLASSES, ignore_index)
            logits = model.forward_logits(images)
            metric.update(logits, masks)
    return metric.compute()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    configure_cuda_stability(device)
    output_dir = ensure_dir(args.output_dir)
    log_path = output_dir / "train_log.jsonl"
    config_path = output_dir / "config.json"

    mean, std = load_normalization_stats()
    class_weights = load_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)
    physical_batch_size = resolve_physical_batch_size(args, device)
    effective_grad_accum_steps = args.grad_accum_steps * math.ceil(args.batch_size / physical_batch_size)

    source_joint = RandomCropFlip(crop_size=args.crop_size, hflip_prob=0.5, vflip_prob=0.0, min_valid_ratio=0.6)
    target_joint = RandomCropFlip(crop_size=args.crop_size, hflip_prob=0.5, vflip_prob=0.0, min_valid_ratio=0.6)
    val_joint = CenterCropPair(crop_size=args.crop_size)

    normalize = NormalizeTensor(mean, std)
    target_views = TargetViewGenerator(mean, std)
    target_sample_transform = TargetSampleTransform(target_views, keep_raw_image=False)

    source_train = LoveDADataset(
        dataset_root=args.dataset_root,
        splits=("Train",),
        domains=(args.source_domain,),
        require_masks=True,
        ignore_black_padding=True,
        joint_transform=source_joint,
        image_transform=normalize,
    )
    target_train = LoveDADataset(
        dataset_root=args.dataset_root,
        splits=("Train",),
        domains=(args.target_domain,),
        require_masks=False,
        ignore_black_padding=True,
        joint_transform=target_joint,
        image_transform=None,
        sample_transform=target_sample_transform,
    )
    val_dataset = LoveDADataset(
        dataset_root=args.dataset_root,
        splits=("Val",),
        domains=(args.val_domain,),
        require_masks=True,
        ignore_black_padding=True,
        joint_transform=val_joint,
        image_transform=normalize,
    )

    source_loader = make_dataloader(
        source_train,
        physical_batch_size,
        args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )
    target_loader = make_dataloader(
        target_train,
        physical_batch_size,
        args.target_num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )
    val_loader = make_dataloader(
        val_dataset,
        physical_batch_size,
        args.val_num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    student = SegFormerMiTB2(
        num_classes=LOVE_DA_NUM_CLASSES,
        ignore_index=LOVE_DA_IGNORE_INDEX,
        pretrained_model_name_or_path=args.pretrained_model_name or None,
        use_pretrained=args.use_pretrained,
    ).to(device)
    teacher = None

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    supervised_loss_fn = SupervisedSegmentationLoss(
        ignore_index=LOVE_DA_IGNORE_INDEX,
        class_weights=class_weights,
        use_dice=args.use_dice,
    )
    amp_enabled = device.type == "cuda" and args.amp
    scaler = GradScaler("cuda", enabled=amp_enabled)

    save_json(
        config_path,
        {
            "source_domain": args.source_domain,
            "target_domain": args.target_domain,
            "val_domain": args.val_domain,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "physical_batch_size": physical_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "target_num_workers": args.target_num_workers,
            "val_num_workers": args.val_num_workers,
            "teacher_momentum": args.teacher_momentum,
            "consistency_weight": args.consistency_weight,
            "confidence_threshold": args.confidence_threshold,
            "source_only_warmup_epochs": args.source_only_warmup_epochs,
            "consistency_rampup_epochs": args.consistency_rampup_epochs,
            "grad_clip_norm": args.grad_clip_norm,
            "effective_grad_accum_steps": effective_grad_accum_steps,
            "micro_batch_size": args.micro_batch_size,
            "max_steps_per_epoch": args.max_steps_per_epoch,
            "pin_memory": args.pin_memory,
            "normalization_mean": mean,
            "normalization_std": std,
        },
    )
    with config_path.open("r", encoding="utf-8") as fp:
        wandb_config = json.load(fp)
    wandb_run = init_wandb(args, wandb_config)

    best_miou = -math.inf
    source_iter = cycle_loader(source_loader)
    target_iter = cycle_loader(target_loader)
    steps_per_epoch = max(len(source_loader), len(target_loader))
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    train_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.time()
        consistency_scale = compute_consistency_scale(
            epoch=epoch,
            warmup_epochs=args.source_only_warmup_epochs,
            rampup_epochs=args.consistency_rampup_epochs,
        )
        effective_consistency_weight = args.consistency_weight * consistency_scale
        if effective_consistency_weight > 0.0 and teacher is None:
            teacher = create_teacher_from_student(student).to(device)
        elif effective_consistency_weight == 0.0:
            teacher = None

        for step in range(steps_per_epoch):
            global_step = (epoch - 1) * steps_per_epoch + step + 1
            source_batch_cpu = next(source_iter)
            target_batch_cpu = next(target_iter) if effective_consistency_weight > 0.0 else None
            current_batch_size = int(source_batch_cpu["image"].shape[0])
            micro_batch_size = resolve_micro_batch_size(args, device, current_batch_size)

            source_loss_accum = 0.0
            target_loss_accum = 0.0
            total_loss_accum = 0.0
            keep_ratio_accum = 0.0
            teacher_conf_mean_accum = 0.0
            teacher_conf_p90_accum = 0.0
            teacher_conf_max_accum = 0.0

            source_batch_for_error: dict[str, object] | None = None
            target_batch_for_error: dict[str, object] | None = None

            for micro_start in range(0, current_batch_size, micro_batch_size):
                micro_end = min(micro_start + micro_batch_size, current_batch_size)
                micro_weight = (micro_end - micro_start) / float(current_batch_size)

                source_batch = move_batch_to_device(slice_batch(source_batch_cpu, micro_start, micro_end), device)
                target_batch = (
                    move_batch_to_device(slice_batch(target_batch_cpu, micro_start, micro_end), device)
                    if target_batch_cpu is not None
                    else None
                )
                source_batch_for_error = source_batch
                target_batch_for_error = target_batch

                source_images = source_batch["image"]
                source_masks = apply_invalid_ignore(source_batch["mask"], source_batch["valid_mask"], LOVE_DA_IGNORE_INDEX)
                source_masks = sanitize_mask_indices(source_masks, LOVE_DA_NUM_CLASSES, LOVE_DA_IGNORE_INDEX)

                with autocast(device_type=device.type, enabled=amp_enabled):
                    source_logits = student(source_images)["logits"]
                    assert_finite("source_logits", source_logits, source_batch)
                    source_loss = supervised_loss_fn(source_logits, source_masks)
                    assert_finite("source_loss", source_loss, source_batch)

                    if effective_consistency_weight > 0.0:
                        assert target_batch is not None
                        assert teacher is not None
                        target_valid = target_batch["valid_mask"]
                        target_weak = target_batch["weak_image"]
                        with torch.no_grad():
                            teacher_logits = teacher.forward_logits(target_weak)
                            assert_finite("teacher_logits", teacher_logits, target_batch)
                            teacher_probs = torch.softmax(teacher_logits, dim=1)
                            teacher_confidence, pseudo_targets = teacher_probs.max(dim=1)
                            pseudo_targets, keep_mask = build_pseudo_mask(
                                teacher_logits,
                                confidence_threshold=args.confidence_threshold,
                                valid_mask=target_valid,
                            )

                        if keep_mask.any():
                            target_strong = target_batch["strong_image"]
                            student_target_logits = student(target_strong)["logits"]
                            assert_finite("student_target_logits", student_target_logits, target_batch)
                            target_loss = masked_consistency_ce(student_target_logits, pseudo_targets, keep_mask)
                            assert_finite("target_loss", target_loss, target_batch)
                        else:
                            target_loss = source_loss.new_zeros(())
                    else:
                        target_loss = source_loss.new_zeros(())
                        keep_mask = torch.zeros_like(source_batch["valid_mask"], dtype=torch.bool)
                        teacher_confidence = source_loss.new_zeros(source_batch["valid_mask"].shape)

                    total_loss = source_loss + effective_consistency_weight * target_loss
                    total_loss = total_loss * (micro_weight / effective_grad_accum_steps)
                    assert_finite("total_loss", total_loss, source_batch)

                if amp_enabled:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                source_loss_accum += float(source_loss.detach().cpu()) * micro_weight
                target_loss_accum += float(target_loss.detach().cpu()) * micro_weight
                total_loss_accum += float(
                    (source_loss + effective_consistency_weight * target_loss).detach().cpu()
                ) * micro_weight
                keep_ratio_accum += float(keep_mask.float().mean().detach().cpu()) * micro_weight
                teacher_conf_mean_accum += float(teacher_confidence.mean().detach().cpu()) * micro_weight
                teacher_conf_p90_accum += float(
                    torch.quantile(teacher_confidence.detach().float().view(-1), 0.90).cpu()
                ) * micro_weight
                teacher_conf_max_accum = max(
                    teacher_conf_max_accum,
                    float(teacher_confidence.max().detach().cpu()),
                )

            should_step = ((step + 1) % effective_grad_accum_steps == 0) or (step + 1 == steps_per_epoch)
            if should_step:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if effective_consistency_weight > 0.0:
                    assert teacher is not None
                    update_ema_teacher(student, teacher, momentum=args.teacher_momentum)

            append_jsonl(
                log_path,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step": step + 1,
                    "source_loss": source_loss_accum,
                    "target_loss": target_loss_accum,
                    "total_loss": total_loss_accum,
                    "pseudo_keep_ratio": keep_ratio_accum,
                    "teacher_conf_mean": teacher_conf_mean_accum,
                    "teacher_conf_p90": teacher_conf_p90_accum,
                    "teacher_conf_max": teacher_conf_max_accum,
                    "consistency_scale": float(consistency_scale),
                    "effective_consistency_weight": float(effective_consistency_weight),
                    "physical_batch_size": physical_batch_size,
                    "effective_grad_accum_steps": effective_grad_accum_steps,
                    "micro_batch_size": micro_batch_size,
                },
            )

            step_done = step + 1
            elapsed_epoch = time.time() - epoch_start_time
            iter_per_sec = step_done / max(elapsed_epoch, 1e-6)
            remaining_iters_epoch = steps_per_epoch - step_done
            eta_epoch_sec = remaining_iters_epoch / max(iter_per_sec, 1e-6)

            total_steps_done = (epoch - 1) * steps_per_epoch + step_done
            total_steps_all = args.epochs * steps_per_epoch
            elapsed_total = time.time() - train_start_time
            total_iter_per_sec = total_steps_done / max(elapsed_total, 1e-6)
            remaining_total_steps = total_steps_all - total_steps_done
            eta_total_sec = remaining_total_steps / max(total_iter_per_sec, 1e-6)

            source_loss_value = source_loss_accum
            target_loss_value = target_loss_accum
            total_loss_value = total_loss_accum
            keep_ratio_value = keep_ratio_accum
            teacher_conf_mean_value = teacher_conf_mean_accum
            teacher_conf_p90_value = teacher_conf_p90_accum
            teacher_conf_max_value = teacher_conf_max_accum

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/step_in_epoch": step_done,
                        "train/global_step": global_step,
                        "train/source_loss": source_loss_value,
                        "train/target_loss": target_loss_value,
                        "train/total_loss": total_loss_value,
                        "train/pseudo_keep_ratio": keep_ratio_value,
                        "train/teacher_conf_mean": teacher_conf_mean_value,
                        "train/teacher_conf_p90": teacher_conf_p90_value,
                        "train/teacher_conf_max": teacher_conf_max_value,
                        "train/consistency_scale": float(consistency_scale),
                        "train/effective_consistency_weight": float(effective_consistency_weight),
                        "train/physical_batch_size": physical_batch_size,
                        "train/effective_grad_accum_steps": effective_grad_accum_steps,
                        "train/eta_epoch_min": eta_epoch_sec / 60.0,
                        "train/eta_total_min": eta_total_sec / 60.0,
                    },
                    step=global_step,
                )

            print(
                (
                    f"\r[train] epoch {epoch:03d}/{args.epochs:03d} "
                    f"iter {step_done:04d}/{steps_per_epoch:04d} "
                    f"src {source_loss_value:.4f} "
                    f"tgt {target_loss_value:.4f} "
                    f"total {total_loss_value:.4f} "
                    f"keep {keep_ratio_value:.3f} "
                    f"conf_mean {teacher_conf_mean_value:.3f} "
                    f"conf_p90 {teacher_conf_p90_value:.3f} "
                    f"uda_w {effective_consistency_weight:.3f} "
                    f"eta_epoch {eta_epoch_sec/60.0:.1f}m "
                    f"eta_total {eta_total_sec/60.0:.1f}m"
                ),
                end="",
                flush=True,
            )

        print()

        val_metrics = evaluate(student, val_loader, device, LOVE_DA_IGNORE_INDEX)
        val_miou = val_metrics["miou"]
        epoch_summary = {
            "epoch": epoch,
            "val/miou": val_metrics["miou"],
            "val/pixel_accuracy": val_metrics["pixel_accuracy"],
            "val/best_miou_before_update": best_miou if math.isfinite(best_miou) else None,
            "train_epoch/consistency_scale": float(consistency_scale),
            "train_epoch/effective_consistency_weight": float(effective_consistency_weight),
        }
        if wandb_run is not None:
            wandb_run.log(epoch_summary, step=epoch * steps_per_epoch)

        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                {
                    "epoch": epoch,
                    "student_state_dict": student.state_dict(),
                    "teacher_state_dict": teacher.state_dict() if teacher is not None else student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": vars(args),
                },
            )

        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "epoch": epoch,
                    "student_state_dict": student.state_dict(),
                    "teacher_state_dict": teacher.state_dict() if teacher is not None else student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": vars(args),
                },
            )

        print(
            f"[epoch {epoch:03d}] "
            f"val_mIoU={val_metrics['miou']:.4f} "
            f"val_pixel_acc={val_metrics['pixel_accuracy']:.4f}"
        )

    if wandb_run is not None:
        wandb_run.summary["best_miou"] = best_miou
        wandb_run.finish()


if __name__ == "__main__":
    main()
