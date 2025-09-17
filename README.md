# Dataset

MS-ASL: https://www.microsoft.com/en-us/research/project/ms-asl/

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


