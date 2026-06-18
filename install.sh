#!/usr/bin/env bash
set -e

# Core Python deps (assumes torch matching your CUDA is already installed).
pip install -r requirements.txt

# PyTorch3D (build from source; pick the ref matching your torch/CUDA if needed).
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Differentiable Gaussian rasterizer (CUDA extension, vendored).
pip install ./submodules/diff-gaussian-rasterization
