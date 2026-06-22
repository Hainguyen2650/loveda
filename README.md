# Computer Vision

This repository now covers two linked tracks for LoveDA:

- dataset analysis and cleanup
- Urban-to-Rural unsupervised domain adaptation for semantic segmentation

The current modeling direction uses `SegFormer-MiT-B2` with a teacher-student consistency setup.

## Current Status

Implemented:

- fast RGB and mask EDA scanners in `scripts/`
- consolidated EDA notebook in `notebooks/loveda_eda.ipynb`
- LoveDA dataset module in `src/datasets/loveda.py`
- SegFormer wrapper in `src/models/segformer.py`
- segmentation losses and metrics in `src/losses/` and `src/metrics/`
- training pipeline in `train.py`
- evaluation and inference entry points in `evaluate.py` and `infer.py`
- local pretrained checkpoint cache in `models/segformer-b2-ade-512-512/`
- `wandb` experiment logging

Key notes:

- `brainstorm/README.md`
- `brainstorm/02-loveda-session.md`
- `brainstorm/04-uda-plan.md`

## Repository Layout

- `scripts/`: raw scanners, cleanup tools, and derived EDA utilities
- `notebooks/`: visualization and brainstorming notebooks
- `src/`: reusable training code
- `outputs/`: generated CSVs, figures, and checkpoints
- `models/`: cached pretrained weights
- `brainstorm/`: project notes and manual session reports

## EDA Workflow

Run raw scans first:

```bash
cc -O3 -pthread -Ithird_party scripts/loveda_rgb_eda_mt.c -lm -o /tmp/loveda_rgb_eda_mt
/tmp/loveda_rgb_eda_mt --dataset-root data/LoveDA --output-dir outputs/dataset/full_rgb_mt --threads 20

cc -O3 -pthread -Ithird_party scripts/loveda_mask_eda_mt.c -lm -o /tmp/loveda_mask_eda_mt
/tmp/loveda_mask_eda_mt --dataset-root data/LoveDA --output-dir outputs/dataset/mask_mt --threads 20
```

Then generate derived summaries:

```bash
python scripts/loveda_advanced_eda.py
```

## UDA Training

Recommended local training run:

```bash
python train.py \
  --dataset-root data/LoveDA \
  --source-domain Urban \
  --target-domain Rural \
  --val-domain Rural \
  --crop-size 512 \
  --batch-size 2 \
  --epochs 20 \
  --lr 6e-5 \
  --teacher-momentum 0.99 \
  --consistency-weight 0.5 \
  --confidence-threshold 0.20 \
  --source-only-warmup-epochs 2 \
  --consistency-rampup-epochs 3 \
  --grad-clip-norm 1.0 \
  --use-pretrained \
  --use-wandb \
  --wandb-project loveda-u2r-uda \
  --wandb-run-name segformer-b2-u2r-thr020-warm2-ramp3 \
  --wandb-tags segformer-b2,uda,u2r,loveda \
  --wandb-mode online
```

Short smoke test:

```bash
python train.py \
  --dataset-root data/LoveDA \
  --source-domain Urban \
  --target-domain Rural \
  --val-domain Rural \
  --crop-size 512 \
  --batch-size 2 \
  --epochs 1 \
  --max-steps-per-epoch 30 \
  --lr 6e-5 \
  --teacher-momentum 0.99 \
  --consistency-weight 0.5 \
  --confidence-threshold 0.20 \
  --grad-clip-norm 1.0 \
  --use-pretrained \
  --use-wandb \
  --wandb-project loveda-u2r-uda \
  --wandb-run-name smoke-thr020 \
  --wandb-tags smoke,segformer-b2,uda,u2r \
  --wandb-mode online
```

## Practical Notes

- the ADE20K classifier-head mismatch warning is expected when loading pretrained SegFormer weights for LoveDA
- `target_loss` can be `0` in early training if pseudo-label filtering keeps no target pixels
- low `pseudo_keep_ratio` means the run is behaving closer to source-only supervision than true UDA
- `wandb` can be run in offline mode and synced later
