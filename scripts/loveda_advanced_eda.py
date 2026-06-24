from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats


DATASET_ROOT = Path("data/LoveDA")
OUTPUT_ROOT = Path("outputs/dataset")
CLASS_NAMES = {
    0: "no-data",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}
SPLITS = ["Train", "Val", "Test"]
DOMAINS = ["Urban", "Rural"]
MASK_SPLITS = ["Train", "Val"]
FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


@dataclass
class SampleRecord:
    split: str
    domain: str
    sample_id: str
    image_path: Path
    mask_path: Path | None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    headers = [str(col) for col in columns]
    values = [[str(row[col]) for col in columns] for row in rows]
    widths = [len(header) for header in headers]
    for row in values:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    lines = [format_row(headers), sep]
    for row in values:
        lines.append(format_row(row))
    return "\n".join(lines)


def build_sample_records() -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in SPLITS:
        for domain in DOMAINS:
            image_dir = DATASET_ROOT / split / domain / "images_png"
            mask_dir = DATASET_ROOT / split / domain / "masks_png"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("*.png")):
                sample_id = image_path.stem
                mask_path = mask_dir / f"{sample_id}.png"
                records.append(
                    SampleRecord(
                        split=split,
                        domain=domain,
                        sample_id=sample_id,
                        image_path=image_path,
                        mask_path=mask_path if mask_path.exists() else None,
                    )
                )
    return records


def average_hash(image: Image.Image, size: int = 8) -> int:
    arr = np.array(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    bits = (arr >= arr.mean()).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(lhs: int, rhs: int) -> int:
    return (lhs ^ rhs).bit_count()


def run_integrity_and_leakage(records: list[SampleRecord]) -> None:
    output_dir = OUTPUT_ROOT / "integrity"
    rows: list[dict] = []
    hash_rows: list[dict] = []
    exact_hash_groups: dict[str, list[dict]] = defaultdict(list)
    hash_candidates: list[dict] = []
    summary_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in records:
        image_read_ok = True
        mask_read_ok = record.mask_path is None
        image_size = ""
        mask_size = ""
        size_match = record.mask_path is None
        unexpected_mask_values = 0
        mask_min = ""
        mask_max = ""
        image_md5 = ""
        ahash = ""

        try:
            image_bytes = record.image_path.read_bytes()
            image_md5 = hashlib.md5(image_bytes).hexdigest()
            image = Image.open(record.image_path).convert("RGB")
            image.load()
            image_size = f"{image.width}x{image.height}"
            ahash_value = average_hash(image)
            ahash = f"{ahash_value:016x}"
            hash_candidates.append(
                {
                    "split": record.split,
                    "domain": record.domain,
                    "sample_id": record.sample_id,
                    "image_path": str(record.image_path.resolve()),
                    "image_md5": image_md5,
                    "average_hash": ahash,
                    "average_hash_int": ahash_value,
                }
            )
            exact_hash_groups[image_md5].append(
                {
                    "split": record.split,
                    "domain": record.domain,
                    "sample_id": record.sample_id,
                    "image_path": str(record.image_path.resolve()),
                    "average_hash": ahash,
                }
            )
        except Exception:
            image_read_ok = False

        if record.mask_path is not None:
            try:
                mask = np.array(Image.open(record.mask_path), dtype=np.uint8)
                mask_read_ok = True
                mask_size = f"{mask.shape[1]}x{mask.shape[0]}"
                size_match = image_size == mask_size
                unique_values = np.unique(mask)
                unexpected_mask_values = int(np.count_nonzero((unique_values < 0) | (unique_values > 7)))
                mask_min = int(unique_values.min()) if unique_values.size else ""
                mask_max = int(unique_values.max()) if unique_values.size else ""
            except Exception:
                mask_read_ok = False
                size_match = False

        key = (record.split, record.domain)
        summary_counts[key]["samples"] += 1
        summary_counts[key]["image_read_ok"] += int(image_read_ok)
        summary_counts[key]["mask_present"] += int(record.mask_path is not None)
        summary_counts[key]["mask_read_ok"] += int(mask_read_ok)
        summary_counts[key]["size_match_ok"] += int(size_match)
        summary_counts[key]["unexpected_mask_value_files"] += int(unexpected_mask_values > 0)

        rows.append(
            {
                "split": record.split,
                "domain": record.domain,
                "sample_id": record.sample_id,
                "image_path": str(record.image_path.resolve()),
                "mask_path": str(record.mask_path.resolve()) if record.mask_path is not None else "",
                "image_read_ok": image_read_ok,
                "mask_read_ok": mask_read_ok,
                "image_size": image_size,
                "mask_size": mask_size,
                "size_match": size_match,
                "mask_present": record.mask_path is not None,
                "mask_min": mask_min,
                "mask_max": mask_max,
                "unexpected_mask_values": unexpected_mask_values,
                "image_md5": image_md5,
                "average_hash": ahash,
            }
        )

    summary_rows = []
    for (split, domain), counts in sorted(summary_counts.items()):
        summary_rows.append(
            {
                "split": split,
                "domain": domain,
                "samples": counts["samples"],
                "image_read_ok": counts["image_read_ok"],
                "mask_present": counts["mask_present"],
                "mask_read_ok": counts["mask_read_ok"],
                "size_match_ok": counts["size_match_ok"],
                "unexpected_mask_value_files": counts["unexpected_mask_value_files"],
            }
        )

    exact_duplicate_rows = []
    for image_md5, items in exact_hash_groups.items():
        if len(items) < 2:
            continue
        pair_count = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair_count += 1
                exact_duplicate_rows.append(
                    {
                        "image_md5": image_md5,
                        "left_split": items[i]["split"],
                        "left_domain": items[i]["domain"],
                        "left_sample_id": items[i]["sample_id"],
                        "left_image_path": items[i]["image_path"],
                        "right_split": items[j]["split"],
                        "right_domain": items[j]["domain"],
                        "right_sample_id": items[j]["sample_id"],
                        "right_image_path": items[j]["image_path"],
                    }
                )

    near_duplicate_rows = []
    for i in range(len(hash_candidates)):
        left = hash_candidates[i]
        for j in range(i + 1, len(hash_candidates)):
            right = hash_candidates[j]
            if left["split"] == right["split"]:
                continue
            distance = hamming_distance(left["average_hash_int"], right["average_hash_int"])
            if distance <= 4:
                near_duplicate_rows.append(
                    {
                        "hamming_distance": distance,
                        "left_split": left["split"],
                        "left_domain": left["domain"],
                        "left_sample_id": left["sample_id"],
                        "left_image_path": left["image_path"],
                        "right_split": right["split"],
                        "right_domain": right["domain"],
                        "right_sample_id": right["sample_id"],
                        "right_image_path": right["image_path"],
                    }
                )

    leakage_summary_rows = [
        {"metric": "exact_duplicate_pairs", "value": len(exact_duplicate_rows)},
        {"metric": "near_duplicate_pairs_hamming_le_4", "value": len(near_duplicate_rows)},
    ]

    write_csv(
        output_dir / "file_integrity_checks.csv",
        rows,
        [
            "split",
            "domain",
            "sample_id",
            "image_path",
            "mask_path",
            "image_read_ok",
            "mask_read_ok",
            "image_size",
            "mask_size",
            "size_match",
            "mask_present",
            "mask_min",
            "mask_max",
            "unexpected_mask_values",
            "image_md5",
            "average_hash",
        ],
    )
    write_csv(
        output_dir / "integrity_summary.csv",
        summary_rows,
        [
            "split",
            "domain",
            "samples",
            "image_read_ok",
            "mask_present",
            "mask_read_ok",
            "size_match_ok",
            "unexpected_mask_value_files",
        ],
    )
    write_csv(
        output_dir / "exact_duplicate_pairs.csv",
        exact_duplicate_rows,
        [
            "image_md5",
            "left_split",
            "left_domain",
            "left_sample_id",
            "left_image_path",
            "right_split",
            "right_domain",
            "right_sample_id",
            "right_image_path",
        ],
    )
    write_csv(
        output_dir / "near_duplicate_pairs.csv",
        near_duplicate_rows,
        [
            "hamming_distance",
            "left_split",
            "left_domain",
            "left_sample_id",
            "left_image_path",
            "right_split",
            "right_domain",
            "right_sample_id",
            "right_image_path",
        ],
    )
    write_csv(output_dir / "leakage_summary.csv", leakage_summary_rows, ["metric", "value"])


def run_class_presence_and_domain_shift() -> None:
    coverage_dir = OUTPUT_ROOT / "coverage"
    drift_dir = OUTPUT_ROOT / "domain_shift"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    drift_dir.mkdir(parents=True, exist_ok=True)

    mask_sample = pd.read_csv(OUTPUT_ROOT / "mask_mt" / "mask_sample_class_counts.csv")
    class_distribution = pd.read_csv(OUTPUT_ROOT / "class_distribution_summary.csv")
    sample_inventory = pd.read_csv(OUTPUT_ROOT / "sample_inventory.csv")

    presence_rows = []
    for (split, domain), group in mask_sample.groupby(["split", "domain"]):
        image_count = len(group)
        pixel_dist_lookup = {
            int(row.class_id): float(row.pixel_ratio)
            for row in class_distribution[
                (class_distribution["split"] == split) & (class_distribution["domain"] == domain)
            ].itertuples(index=False)
        }
        for class_id in range(8):
            present_mask = group[f"count_{class_id}"] > 0
            images_with_class = int(present_mask.sum())
            image_fraction = images_with_class / image_count if image_count else 0.0
            avg_ratio_when_present = (
                float(group.loc[present_mask, f"ratio_{class_id}"].mean()) if images_with_class else 0.0
            )
            mean_ratio_all_images = float(group[f"ratio_{class_id}"].mean())
            presence_rows.append(
                {
                    "split": split,
                    "domain": domain,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "images_with_class": images_with_class,
                    "image_count": image_count,
                    "image_fraction": round(image_fraction, 8),
                    "avg_ratio_when_present": round(avg_ratio_when_present, 8),
                    "mean_ratio_all_images": round(mean_ratio_all_images, 8),
                    "pixel_ratio": round(pixel_dist_lookup.get(class_id, 0.0), 8),
                    "rare_class_by_image_presence": image_fraction < 0.1,
                }
            )

    presence_df = pd.DataFrame(presence_rows)
    presence_df["presence_pixel_gap"] = presence_df["pixel_ratio"] - presence_df["image_fraction"]
    presence_df.sort_values(["split", "domain", "class_id"]).to_csv(
        coverage_dir / "class_image_presence_summary.csv", index=False
    )
    presence_df.sort_values("image_fraction").to_csv(
        coverage_dir / "rare_class_summary.csv", index=False
    )

    sample_inventory["brightness_mean"] = sample_inventory[["mean_r", "mean_g", "mean_b"]].mean(axis=1)
    sample_inventory["brightness_std"] = sample_inventory[["std_r", "std_g", "std_b"]].mean(axis=1)
    image_features = [
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "brightness_mean",
        "brightness_std",
    ]
    pair_specs = [
        ("Train", "Urban", "Train", "Rural"),
        ("Val", "Urban", "Val", "Rural"),
        ("Train", "Urban", "Val", "Urban"),
        ("Train", "Rural", "Val", "Rural"),
    ]
    drift_rows = []
    for left_split, left_domain, right_split, right_domain in pair_specs:
        left = sample_inventory[
            (sample_inventory["split"] == left_split) & (sample_inventory["domain"] == left_domain)
        ]
        right = sample_inventory[
            (sample_inventory["split"] == right_split) & (sample_inventory["domain"] == right_domain)
        ]
        for feature in image_features:
            left_values = left[feature].to_numpy()
            right_values = right[feature].to_numpy()
            delta = float(left_values.mean() - right_values.mean())
            left_std = float(left_values.std(ddof=0))
            right_std = float(right_values.std(ddof=0))
            pooled = math.sqrt((left_std**2 + right_std**2) / 2.0) if (left_std or right_std) else 0.0
            cohen_d = delta / pooled if pooled else 0.0
            wasserstein = float(stats.wasserstein_distance(left_values, right_values))
            drift_rows.append(
                {
                    "left_split": left_split,
                    "left_domain": left_domain,
                    "right_split": right_split,
                    "right_domain": right_domain,
                    "feature": feature,
                    "left_mean": round(float(left_values.mean()), 6),
                    "right_mean": round(float(right_values.mean()), 6),
                    "delta_mean": round(delta, 6),
                    "cohen_d": round(cohen_d, 6),
                    "wasserstein_distance": round(wasserstein, 6),
                }
            )
    write_csv(
        drift_dir / "image_feature_drift.csv",
        drift_rows,
        [
            "left_split",
            "left_domain",
            "right_split",
            "right_domain",
            "feature",
            "left_mean",
            "right_mean",
            "delta_mean",
            "cohen_d",
            "wasserstein_distance",
        ],
    )

    class_drift_rows = []
    for left_split, left_domain, right_split, right_domain in pair_specs:
        left = presence_df[
            (presence_df["split"] == left_split) & (presence_df["domain"] == left_domain)
        ].set_index("class_id")
        right = presence_df[
            (presence_df["split"] == right_split) & (presence_df["domain"] == right_domain)
        ].set_index("class_id")
        for class_id in range(8):
            class_drift_rows.append(
                {
                    "left_split": left_split,
                    "left_domain": left_domain,
                    "right_split": right_split,
                    "right_domain": right_domain,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "left_pixel_ratio": round(float(left.loc[class_id, "pixel_ratio"]), 8),
                    "right_pixel_ratio": round(float(right.loc[class_id, "pixel_ratio"]), 8),
                    "delta_pixel_ratio": round(
                        float(left.loc[class_id, "pixel_ratio"] - right.loc[class_id, "pixel_ratio"]), 8
                    ),
                    "left_image_fraction": round(float(left.loc[class_id, "image_fraction"]), 8),
                    "right_image_fraction": round(float(right.loc[class_id, "image_fraction"]), 8),
                    "delta_image_fraction": round(
                        float(left.loc[class_id, "image_fraction"] - right.loc[class_id, "image_fraction"]), 8
                    ),
                }
            )
    write_csv(
        drift_dir / "class_drift_summary.csv",
        class_drift_rows,
        [
            "left_split",
            "left_domain",
            "right_split",
            "right_domain",
            "class_id",
            "class_name",
            "left_pixel_ratio",
            "right_pixel_ratio",
            "delta_pixel_ratio",
            "left_image_fraction",
            "right_image_fraction",
            "delta_image_fraction",
        ],
    )


def run_spatial_structure_analysis(records: list[SampleRecord]) -> None:
    spatial_dir = OUTPUT_ROOT / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    class_buffers: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    class_meta: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sample_rows = []

    for record in records:
        if record.mask_path is None:
            continue
        mask = np.array(Image.open(record.mask_path), dtype=np.uint8)
        total_components = 0
        present_class_count = 0
        thin_class_components = 0
        adjacency_total = (mask.shape[0] * (mask.shape[1] - 1)) + ((mask.shape[0] - 1) * mask.shape[1])
        boundary_density = (
            (mask[:, 1:] != mask[:, :-1]).sum() + (mask[1:, :] != mask[:-1, :]).sum()
        ) / adjacency_total

        row = {
            "split": record.split,
            "domain": record.domain,
            "sample_id": record.sample_id,
            "boundary_density": round(float(boundary_density), 8),
            "total_components": 0,
            "present_class_count": 0,
            "building_components": 0,
            "road_components": 0,
            "building_mean_component_area": 0.0,
            "road_mean_component_area": 0.0,
        }

        for class_id in range(1, 8):
            class_mask = mask == class_id
            if not class_mask.any():
                continue
            present_class_count += 1
            labeled, num_components = ndimage.label(class_mask, structure=FOUR_CONNECTED)
            component_sizes = np.bincount(labeled.ravel())[1:]
            if component_sizes.size == 0:
                continue
            total_components += int(num_components)
            if class_id in (2, 3):
                thin_class_components += int(num_components)
            key = (record.split, record.domain, class_id)
            class_buffers[key].append(component_sizes.astype(np.int64))
            class_meta[key]["images_with_class"] += 1
            class_meta[key]["total_components"] += int(num_components)
            class_meta[key]["total_pixels"] += int(component_sizes.sum())

            if class_id == 2:
                row["building_components"] = int(num_components)
                row["building_mean_component_area"] = round(float(component_sizes.mean()), 6)
            elif class_id == 3:
                row["road_components"] = int(num_components)
                row["road_mean_component_area"] = round(float(component_sizes.mean()), 6)

        row["total_components"] = total_components
        row["present_class_count"] = present_class_count
        row["thin_class_components"] = thin_class_components
        sample_rows.append(row)

    sample_spatial_df = pd.DataFrame(sample_rows).sort_values(["split", "domain", "sample_id"])
    sample_spatial_df.to_csv(spatial_dir / "sample_spatial_summary.csv", index=False)

    class_rows = []
    for (split, domain, class_id), buffers in sorted(class_buffers.items()):
        sizes = np.concatenate(buffers) if buffers else np.array([], dtype=np.int64)
        meta = class_meta[(split, domain, class_id)]
        images_with_class = int(meta["images_with_class"])
        total_components = int(meta["total_components"])
        mean_components_per_image = total_components / images_with_class if images_with_class else 0.0
        class_rows.append(
            {
                "split": split,
                "domain": domain,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "images_with_class": images_with_class,
                "total_components": total_components,
                "mean_components_per_image": round(mean_components_per_image, 6),
                "mean_component_area": round(float(sizes.mean()), 6) if sizes.size else 0.0,
                "median_component_area": round(float(np.median(sizes)), 6) if sizes.size else 0.0,
                "p90_component_area": round(float(np.percentile(sizes, 90)), 6) if sizes.size else 0.0,
                "max_component_area": int(sizes.max()) if sizes.size else 0,
                "component_area_std": round(float(sizes.std(ddof=0)), 6) if sizes.size else 0.0,
            }
        )
    write_csv(
        spatial_dir / "class_component_stats.csv",
        class_rows,
        [
            "split",
            "domain",
            "class_id",
            "class_name",
            "images_with_class",
            "total_components",
            "mean_components_per_image",
            "mean_component_area",
            "median_component_area",
            "p90_component_area",
            "max_component_area",
            "component_area_std",
        ],
    )


def run_outlier_validation_and_training_outputs(records: list[SampleRecord]) -> None:
    outlier_dir = OUTPUT_ROOT / "outliers"
    training_dir = OUTPUT_ROOT / "training"
    manual_review_dir = OUTPUT_ROOT / "review"
    outlier_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    manual_review_dir.mkdir(parents=True, exist_ok=True)

    sample_inventory = pd.read_csv(OUTPUT_ROOT / "sample_inventory.csv")
    mask_outliers = pd.read_csv(outlier_dir / "mask_outliers.csv")
    rgb_outliers = pd.read_csv(outlier_dir / "rgb_outliers.csv")
    mask_sample = pd.read_csv(OUTPUT_ROOT / "mask_mt" / "mask_sample_class_counts.csv")
    component_stats = pd.read_csv(OUTPUT_ROOT / "spatial" / "class_component_stats.csv")
    class_presence = pd.read_csv(OUTPUT_ROOT / "coverage" / "class_image_presence_summary.csv")
    rgb_global = pd.read_csv(OUTPUT_ROOT / "full_rgb_mt" / "rgb_global_stats.csv")

    inventory_lookup = sample_inventory[["split", "domain", "sample_id", "image_path", "mask_path"]]
    rgb_review = (
        rgb_outliers.merge(inventory_lookup, on=["split", "domain", "sample_id"], how="left")
        .sort_values("rgb_outlier_score", ascending=False)
        .head(50)
        .assign(review_type="rgb")
    )
    mask_review = (
        mask_outliers.merge(inventory_lookup, on=["split", "domain", "sample_id"], how="left")
        .sort_values("mask_outlier_score", ascending=False)
        .head(50)
        .assign(review_type="mask")
    )
    review_queue = pd.concat([rgb_review, mask_review], ignore_index=True, sort=False)
    review_queue.to_csv(manual_review_dir / "manual_review_queue.csv", index=False)

    train_rgb = rgb_global[rgb_global["split"] == "Train"].copy()
    train_pixel_total = train_rgb.groupby("channel")["pixel_count"].sum().to_dict()
    normalization_rows = []
    for channel in ["R", "G", "B"]:
        channel_rows = train_rgb[train_rgb["channel"] == channel]
        weighted_mean = (channel_rows["mean"] * channel_rows["pixel_count"]).sum() / train_pixel_total[channel]
        weighted_var = (
            ((channel_rows["std"] ** 2) + (channel_rows["mean"] ** 2)) * channel_rows["pixel_count"]
        ).sum() / train_pixel_total[channel] - weighted_mean**2
        normalization_rows.append(
            {
                "channel": channel,
                "train_mean": round(float(weighted_mean), 6),
                "train_std": round(float(math.sqrt(max(weighted_var, 0.0))), 6),
            }
        )
    write_csv(training_dir / "normalization_recommendation.csv", normalization_rows, ["channel", "train_mean", "train_std"])

    train_presence = class_presence[
        (class_presence["split"] == "Train") & (class_presence["domain"].isin(["Urban", "Rural"]))
    ].copy()
    train_presence = train_presence[train_presence["class_id"] != 0]
    freq = train_presence.groupby("class_id")["pixel_ratio"].sum()
    freq = freq / freq.sum()
    median_freq = float(np.median(freq.to_numpy()))
    class_weight_rows = []
    for class_id, value in freq.items():
        class_weight_rows.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "pixel_ratio": round(float(value), 8),
                "median_frequency_weight": round(float(median_freq / value), 8),
                "inverse_sqrt_frequency_weight": round(float(1.0 / math.sqrt(value)), 8),
            }
        )
    write_csv(
        training_dir / "class_weights.csv",
        class_weight_rows,
        [
            "class_id",
            "class_name",
            "pixel_ratio",
            "median_frequency_weight",
            "inverse_sqrt_frequency_weight",
        ],
    )

    train_mask = mask_sample[mask_sample["split"] == "Train"].copy()
    sanity_rows = []
    seen_ids: set[tuple[str, str, str]] = set()
    for domain in DOMAINS:
        domain_group = train_mask[train_mask["domain"] == domain]
        domain_outliers = rgb_outliers[
            (rgb_outliers["split"] == "Train") & (rgb_outliers["domain"] == domain)
        ].head(4)
        for row in domain_outliers.itertuples(index=False):
            key = (row.split, row.domain, str(row.sample_id))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            sanity_rows.append({"split": row.split, "domain": row.domain, "sample_id": row.sample_id, "reason": "rgb_outlier"})

        for class_id in range(1, 8):
            ratio_col = f"ratio_{class_id}"
            candidate = domain_group.sort_values(ratio_col, ascending=False).iloc[0]
            key = (candidate["split"], candidate["domain"], str(candidate["sample_id"]))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            sanity_rows.append(
                {
                    "split": candidate["split"],
                    "domain": candidate["domain"],
                    "sample_id": candidate["sample_id"],
                    "reason": f"high_{CLASS_NAMES[class_id]}_ratio",
                }
            )
    sanity_df = pd.DataFrame(sanity_rows).merge(
        inventory_lookup, on=["split", "domain", "sample_id"], how="left"
    )
    sanity_df.to_csv(training_dir / "sanity_subset.csv", index=False)

    road_building_stats = component_stats[
        component_stats["class_name"].isin(["road", "building"]) & (component_stats["split"] == "Train")
    ]
    avg_boundary_density = float(
        pd.read_csv(OUTPUT_ROOT / "spatial" / "sample_spatial_summary.csv")
        .query("split == 'Train'")["boundary_density"]
        .mean()
    )
    recommendation_text = f"""# LoveDA Training-Oriented EDA Recommendations

## Normalization

Recommended train-only RGB normalization values:

{markdown_table(normalization_rows, ["channel", "train_mean", "train_std"])}

## No-data label policy

- Treat class `0` as `ignore_index=0`.
- Exclude class `0` from loss weighting and metric aggregation.

## Class weighting

Suggested reference weights:

{markdown_table(class_weight_rows, ["class_id", "class_name", "pixel_ratio", "median_frequency_weight", "inverse_sqrt_frequency_weight"])}

Practical recommendation:
- start with `CrossEntropyLoss(ignore_index=0)`
- then move to `CrossEntropy + Dice`
- only add class weights if minority classes still underperform

## Crop and tiling strategy

Observed:
- all images are fixed at `1024x1024`
- average train boundary density is `{avg_boundary_density:.6f}`
- roads and buildings show high component fragmentation relative to other classes

Recommendation:
- start with `512x512` random crops for training
- use class-aware crop sampling when the crop contains too much background/no-data
- use `50%` overlap during validation/inference tiling if whole-image inference is not feasible

## Domain-aware evaluation

- report metrics separately for `Urban` and `Rural`
- do not rely on one merged score only

## Sanity subset

A small overfit/debug subset was exported to:
- `outputs/dataset/training/sanity_subset.csv`

Use it first to validate:
- dataloader correctness
- label handling
- loss reduction
- basic model overfit behavior
"""
    write_markdown(training_dir / "training_recommendations.md", recommendation_text)


def main() -> None:
    records = build_sample_records()
    run_integrity_and_leakage(records)
    run_class_presence_and_domain_shift()
    run_spatial_structure_analysis(records)
    run_outlier_validation_and_training_outputs(records)
    print("Advanced EDA outputs generated.")


if __name__ == "__main__":
    main()
