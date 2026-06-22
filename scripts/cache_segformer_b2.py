from __future__ import annotations

import argparse
from pathlib import Path

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


DEFAULT_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
DEFAULT_OUTPUT_DIR = Path("models/segformer-b2-ade-512-512")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache a pretrained SegFormer-B2 checkpoint locally.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = SegformerImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id)

    processor.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    print(f"Saved pretrained checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
