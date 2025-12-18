# Abstract

The recent surge in large language models has automated translations of spoken and written languages. However, these advances remain largely inaccessible to American Sign Language (ASL) users, whose language relies on complex visual cues. Isolated sign language recognition (ISLR)—the task of classifying videos of individual signs—can help bridge this gap, but is currently limited by scarce per-sign data, high signer variability, and substantial computational costs. We propose a model for ISLR that reduces computational requirements while maintaining robustness to signer variation. Our approach integrates (i) a pose estimation pipeline to extract hand and face joint coordinates, (ii) a segmentation module that isolates relevant information, and (iii) a ResNet–Transformer backbone to jointly model spatial and temporal dependencies.

Our paper can be found at https://arxiv.org/pdf/2512.14876

# Dataset

ASL Citizen: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/

To load in the data into dataloaders, run the script
* train_loader, val_loader, test_loader, label_to_idx = get_data_loaders(VIDEO_DIR, SPLIT_DIR, batch_size=BATCH_SIZE)

To use remove_extras and keep only the n most common glosses use the command line:
`python remove_extras.py <videosFilepath> <jointPath> <trainPath> <valPath> <testPath> [newName] [numGlosses]`

# Training

Training can be done on the HPC with the `train.py` script. If the system uses slurm, submit the job with `sbatch train.sh` (after defining the desired computational resources).

# Project Setup & Quickstart (with `uv`)

> This project uses [uv](https://docs.astral.sh/uv/) for Python packaging, virtualenvs, and dependency management.
> **First-time setup:** make sure to **install `uv`** and **run `uv sync`** to download all dependencies.

## 1) Prerequisites

* Python 3.10+ (project is pinned in `pyproject.toml` / `uv.lock`)
* `uv` installed

**Install `uv`**

* macOS / Linux:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  exec $SHELL -l
  uv --version
  ```
* Windows (PowerShell):

  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  uv --version
  ```

## 2) Install dependencies

```bash
# from the project root (where pyproject.toml lives)
uv venv .venv          # optional but recommended: create a local venv
uv sync                # installs/locks all runtime + dev deps
```

> `uv sync` reads `pyproject.toml` / `uv.lock` and installs everything into the active environment.

## 3) Run the app

```bash
# pick the one that matches your entrypoint
uv run python main.py
```

## 4) Common tasks

* **Run tests**

  ```bash
  uv run pytest
  ```
* **Format & lint (example with Ruff)**

  ```bash
  uvx ruff format
  uvx ruff check --fix
  ```
* **Add a dependency**

  ```bash
  uv add requests
  ```
* **Add a dev dependency**

  ```bash
  uv add -D pytest
  ```
* **Pin a Python version (recommended)**

  ```bash
  uv python install 3.11
  uv python pin 3.11
  uv sync
  ```

## 5) Troubleshooting

* **`uv` not found**: open a new terminal (or run `exec $SHELL -l` on macOS/Linux) so your PATH updates.
* **Weird env issues**: recreate the venv and resync.

  ```bash
  uv venv --recreate .venv
  uv sync
  ```


