# Dependency, Requirements, and Docker Image Strategy

This document explains how this repo manages Python dependencies and how those dependencies map to future Docker images.

The goal is to keep the project installable, reproducible, and flexible enough to support:

- slim CPU images for lightweight pipeline stages
- CPU ML images for training/evaluation without GPUs
- Ray images for distributed local or cluster execution
- GPU images that build on top of CUDA/PyTorch base images

---

## 1. Why `pyproject.toml` exists

`pyproject.toml` is the source of truth for this Python package.

It defines:

- the package name
- the package version
- the Python version requirement
- the base dependencies
- optional dependency groups such as ML, Ray, dev tools, and notebooks
- how `pip` should build and install the package

Because this repo uses a `src/` layout, the package source lives under:

```text
src/geo_mlops/
```

After installation, the package can be imported from anywhere:

```python
import geo_mlops
```

Without a proper `pyproject.toml`, the repo is mostly just a folder of scripts. With it, the repo becomes an installable Python package.

---

## 2. Relationship between `pyproject.toml` and requirements files

The dependency definitions live in `pyproject.toml`.

The `requirements-*.txt` files are only convenience entrypoints into those dependency groups.

In other words:

```text
pyproject.toml      = source of truth
requirements*.txt   = shortcuts for pip/Docker/CI
```

For example, this command:

```bash
python -m pip install -r requirements-ray.txt
```

may internally point to:

```text
-e .[ml,ray]
```

which tells `pip` to install this repo in editable mode with the `ml` and `ray` extras defined in `pyproject.toml`.

---

## 3. Dependency groups

The repo intentionally separates lightweight runtime dependencies from heavier ML, Ray, and dev dependencies.

### Base dependencies

The base dependencies are installed with:

```bash
python -m pip install .
```

or, for development:

```bash
python -m pip install -e .
```

These dependencies support the core pipeline code:

- contracts
- CLI utilities
- config loading
- tiling/splitting
- geospatial IO
- MLflow integration
- lightweight pipeline stages

Base dependencies should avoid heavy packages like PyTorch and Ray unless absolutely required.

### `ml` extra

Installed with:

```bash
python -m pip install -e ".[ml]"
```

This includes:

- PyTorch
- torchvision
- Transformers

Use this for single-node model training, evaluation, and inference.

### `ray` extra

Installed with:

```bash
python -m pip install -e ".[ray]"
```

This includes Ray dependencies.

In practice, Ray model execution usually needs both `ml` and `ray`:

```bash
python -m pip install -e ".[ml,ray]"
```

### `models` extra

Installed with:

```bash
python -m pip install -e ".[models]"
```

This includes model-library dependencies such as Transformers, but intentionally does not install PyTorch.

This is useful for GPU Docker images that already start from a PyTorch CUDA base image.

### `dev` extra

Installed with:

```bash
python -m pip install -e ".[dev]"
```

This includes developer tools such as:

- pytest
- ruff
- mypy
- pre-commit

### `notebooks` extra

Installed with:

```bash
python -m pip install -e ".[notebooks]"
```

This includes Jupyter/IPython tooling.

---

## 4. Requirements files

The requirements files are intentionally small.

### `requirements.txt`

```text
-e .
```

Installs the base repo only.

Use for slim runtime environments.

### `requirements-dev.txt`

```text
-e .[dev]
```

Installs the base repo plus developer tools.

Use for local linting/testing or lightweight CI.

### `requirements-ml.txt`

```text
-e .[ml]
```

Installs the base repo plus PyTorch, torchvision, and Transformers.

Use for CPU training/evaluation images.

### `requirements-ray.txt`

```text
-e .[ml,ray]
```

Installs the base repo plus ML and Ray dependencies.

Use for local Ray clusters, Ray Train, Ray evaluation, or CPU distributed execution.

### `requirements-all.txt`

```text
-e .[ml,ray,dev,notebooks]
```

Installs the full local development environment.

Use for a complete local dev setup.

### `requirements-gpu.txt`

```text
-e .[models,ray]
```

Installs the base repo plus Transformers and Ray, but does not install PyTorch.

This file assumes the Docker base image already provides GPU-enabled PyTorch.

For example:

```dockerfile
FROM pytorch/pytorch:<cuda-runtime-tag>
```

Then:

```dockerfile
RUN python -m pip install -r requirements-gpu.txt
```

This avoids accidentally reinstalling mismatched PyTorch/CUDA wheels inside the image.

---

## 5. Why `requirements-gpu.txt` does not install CUDA

`requirements-gpu.txt` does not install CUDA, cuDNN, PyTorch, or torchvision.

That is intentional.

GPU dependencies should come from one of two places.

### Recommended: PyTorch CUDA base image

Example:

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

This base image provides the CUDA runtime, cuDNN, and GPU-enabled PyTorch.

Then the repo installs only the remaining project dependencies:

```dockerfile
RUN python -m pip install -r requirements-gpu.txt
```

### Alternative: install CUDA wheels manually

Example:

```dockerfile
FROM python:3.11-slim

RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN python -m pip install -r requirements-gpu.txt
```

This works, but it is easier to mismatch CUDA, PyTorch, torchvision, and the host NVIDIA driver.

For this repo, the recommended GPU strategy is to start from a PyTorch CUDA base image.

---

## 6. Docker image mapping

The dependency split enables multiple image types.

### Slim CPU image

Purpose:

- contracts
- CLIs
- tiling
- splitting
- gating
- metadata inspection
- lightweight MLflow logic

Install command:

```dockerfile
RUN python -m pip install -r requirements.txt
```

Equivalent to:

```bash
python -m pip install .
```

### CPU ML image

Purpose:

- single-node training
- single-node evaluation
- model inference
- CPU model debugging

Install command:

```dockerfile
RUN python -m pip install -r requirements-ml.txt
```

Equivalent to:

```bash
python -m pip install -e ".[ml]"
```

### CPU Ray image

Purpose:

- local Ray cluster testing
- Ray Train on CPU
- Ray evaluation/inference on CPU
- fake multinode Docker Compose cluster

Install command:

```dockerfile
RUN python -m pip install -r requirements-ray.txt
```

Equivalent to:

```bash
python -m pip install -e ".[ml,ray]"
```

### GPU Ray image

Purpose:

- GPU training
- GPU inference
- GPU Ray workers
- future KubeRay/Argo execution

Base image:

```dockerfile
FROM pytorch/pytorch:<cuda-runtime-tag>
```

Install command:

```dockerfile
RUN python -m pip install -r requirements-gpu.txt
```

Equivalent to:

```bash
python -m pip install -e ".[models,ray]"
```

The GPU image relies on the base image for PyTorch/CUDA.

### Dev image or CI install

Purpose:

- linting
- unit tests
- packaging checks
- import tests

Install command:

```dockerfile
RUN python -m pip install -r requirements-dev.txt
```

If tests require ML/Ray imports, use:

```dockerfile
RUN python -m pip install -r requirements-all.txt
```

---

## 7. Validation commands

Before building Docker images, validate each dependency profile in a clean environment.

### Base install

```bash
conda create -n mlops-base-test python=3.11 -y
conda activate mlops-base-test
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import geo_mlops, rasterio, mlflow; print('base OK')"
```

### ML install

```bash
conda create -n mlops-ml-test python=3.11 -y
conda activate mlops-ml-test
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -c "import geo_mlops, torch, torchvision, transformers; print('ml OK')"
```

### Ray install

```bash
conda create -n mlops-ray-test python=3.11 -y
conda activate mlops-ray-test
python -m pip install --upgrade pip
python -m pip install -r requirements-ray.txt
python -c "import geo_mlops, torch, transformers, ray; print('ray OK')"
```

### Full local dev install

```bash
conda create -n mlops-all-test python=3.11 -y
conda activate mlops-all-test
python -m pip install --upgrade pip
python -m pip install -r requirements-all.txt
python -c "import geo_mlops, torch, transformers, ray, pytest; print('all OK')"
pytest -q
```

If all profiles install cleanly, the repo is ready for Docker image work.

---

## 8. Recommended file layout

Keep these files at the repository root:

```text
pyproject.toml
requirements.txt
requirements-dev.txt
requirements-ml.txt
requirements-ray.txt
requirements-all.txt
requirements-gpu.txt
```

Reason:

- Python packaging tools expect `pyproject.toml` at the project root.
- `pip install .` expects to run from the root containing `pyproject.toml`.
- Dockerfiles and CI pipelines commonly install from root-level requirements files.
- Keeping these files at root makes the repo easier for other engineers, CI systems, and Docker builds to understand.

Do not move `pyproject.toml` into a subdirectory unless the actual Python package root also moves.

If the number of requirements files grows later, they can be moved into a directory such as:

```text
requirements/
  base.txt
  dev.txt
  ml.txt
  ray.txt
  all.txt
  gpu.txt
```

However, for the current repo, root-level files are simpler and more standard.

---

## 10. Summary

The intended dependency strategy is:

```text
pyproject.toml
  source of truth

requirements*.txt
  small convenience wrappers around pyproject extras

Dockerfiles
  choose the right requirements file depending on image type

slim image
  requirements.txt

CPU ML image
  requirements-ml.txt

CPU Ray image
  requirements-ray.txt

GPU Ray image
  PyTorch CUDA base image + requirements-gpu.txt

dev/CI
  requirements-dev.txt or requirements-all.txt
```

This keeps the repo lightweight by default while still supporting training, Ray, and GPU execution when needed.
