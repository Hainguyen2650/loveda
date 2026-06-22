from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_normalization_stats(
    csv_path: str | Path = "outputs/dataset/training/normalization_recommendation.csv",
    fallback_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    fallback_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    path = Path(csv_path)
    if not path.exists():
        return fallback_mean, fallback_std

    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if len(rows) < 3:
        return fallback_mean, fallback_std

    channel_order = ["R", "G", "B"]
    mean = []
    std = []
    by_channel = {row["channel"]: row for row in rows}
    for channel in channel_order:
        row = by_channel.get(channel)
        if row is None:
            return fallback_mean, fallback_std
        mean.append(float(row["train_mean"]) / 255.0)
        std.append(float(row["train_std"]) / 255.0)
    return tuple(mean), tuple(std)


def load_class_weights(
    csv_path: str | Path = "outputs/dataset/training/class_weights.csv",
    weight_column: str = "inverse_sqrt_frequency_weight",
) -> torch.Tensor | None:
    path = Path(csv_path)
    if not path.exists():
        return None

    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return None

    max_class_id = max(int(row["class_id"]) for row in rows)
    weights = torch.ones(max_class_id + 1, dtype=torch.float32)
    for row in rows:
        class_id = int(row["class_id"])
        weights[class_id] = float(row[weight_column])
    return weights
