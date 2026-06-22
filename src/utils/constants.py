from __future__ import annotations

from pathlib import Path


LOVE_DA_ROOT = Path("data/LoveDA")

LOVE_DA_SPLITS = ("Train", "Val", "Test")
LOVE_DA_DOMAINS = ("Urban", "Rural")
LOVE_DA_MASK_SPLITS = ("Train", "Val")

LOVE_DA_IGNORE_INDEX = 0
LOVE_DA_NUM_CLASSES = 8

LOVE_DA_CLASS_NAMES = {
    0: "no-data",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}

LOVE_DA_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (0, 0, 255),
    5: (159, 129, 183),
    6: (0, 255, 0),
    7: (255, 195, 128),
}

LOVE_DA_IMAGE_SUBDIR = "images_png"
LOVE_DA_MASK_SUBDIR = "masks_png"

SEGFORMER_MIT_B2_CONFIG = {
    "depths": [3, 4, 6, 3],
    "hidden_sizes": [64, 128, 320, 512],
    "decoder_hidden_size": 768,
    "num_attention_heads": [1, 2, 5, 8],
    "sr_ratios": [8, 4, 2, 1],
    "mlp_ratios": [4, 4, 4, 4],
    "patch_sizes": [7, 3, 3, 3],
    "strides": [4, 2, 2, 2],
    "hidden_act": "gelu",
    "drop_path_rate": 0.1,
}
