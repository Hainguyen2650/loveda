# Repository Guidelines

## Project Structure & Module Organization

This repository is centered on LoveDA dataset analysis and early modeling prep.

- `scripts/`: operational entry points. Current core tools are `loveda_rgb_eda_mt.c`, `loveda_mask_eda_mt.c`, `count_large_padding_images_mt.c`, and `loveda_advanced_eda.py`.
- `notebooks/`: analysis notebook surfaces. Use `notebooks/loveda_eda.ipynb` as the main visualization notebook.
- `outputs/`: generated artifacts only. Dataset tables live under `outputs/dataset/`, charts under `outputs/figures/`.
- `src/`: future reusable training code (`datasets/`, `models/`, `losses/`, `metrics/`, `transforms/`, `utils/`).
- `brainstorm/`: project notes and session reports.
- `third_party/`: vendored dependencies such as `stb_image.h`.

## Build, Test, and Development Commands

- Build RGB EDA scanner:
  ```bash
  cc -O3 -pthread -Ithird_party scripts/loveda_rgb_eda_mt.c -lm -o /tmp/loveda_rgb_eda_mt
  ```
- Run RGB scan:
  ```bash
  /tmp/loveda_rgb_eda_mt --dataset-root data/LoveDA --output-dir outputs/dataset/full_rgb_mt --threads 20
  ```
- Build and run mask scan:
  ```bash
  cc -O3 -pthread -Ithird_party scripts/loveda_mask_eda_mt.c -lm -o /tmp/loveda_mask_eda_mt
  /tmp/loveda_mask_eda_mt --dataset-root data/LoveDA --output-dir outputs/dataset/mask_mt --threads 20
  ```
- Generate derived summaries:
  ```bash
  ./.venv/bin/python scripts/loveda_advanced_eda.py
  ```

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints where practical, `snake_case` for functions/files.
- C: keep functions small, prefer explicit structs, `snake_case` for helpers, `UPPER_CASE` for constants/macros.
- Notebook/script names should stay dataset-specific and descriptive, e.g. `loveda_*`.
- Do not mix generated outputs with source files.

## Testing Guidelines

There is no formal test suite yet. Validate changes with:

- successful C compilation
- a smoke run on a small or full dataset slice
- notebook JSON validity after edits

When changing EDA logic, verify that affected CSV schemas remain stable or document the change in `README.md` and `brainstorm/`.

## Commit & Pull Request Guidelines

This repo has no commit history yet. Use short imperative commit messages, e.g.:

- `add padding-aware rgb scanner`
- `merge loveda notebooks into one`

PRs should include:
- scope summary
- commands run
- affected output folders
- screenshots only when notebook/chart changes matter visually

## Data & Safety Notes

- Treat `data/` as source data; avoid manual edits unless the task is explicit dataset cleanup.
- Large deletions or filtering steps should write a CSV report first, then update `brainstorm/03-padding-session.md` or the relevant session note.

## Brainstorm Note Rules

- `brainstorm/README.md` must be written in English.
- It is the central tracker for this project’s note workflow.
- It should list the active Markdown files in `brainstorm/`, especially session reports and definition notes.
- While working, keep `brainstorm/README.md` updated so it reflects the current note structure and project tracking state.
- Do not treat `README.md` as the place for full manual work logs.
- When the user explicitly asks to `update report`, write a manual summary of the work performed, the sequence of changes, and the results into the relevant report note under `brainstorm/`.
