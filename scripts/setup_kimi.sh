#!/bin/bash
#SBATCH --job-name=kimi_setup
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This source file is part of the VCR project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# Kimi-VL requires Python 3.10+ due to type hint syntax in the model code
# Load gcc first, then python (which depends on gcc), then rust
ml gcc/14.2.0
ml python/3.12.1
ml cuda/11.7.1
ml rust/1.81.0  # Required to build tiktoken

# Activate the virtual environment
python3 -m venv /home/groups/roxanad/sonnet/vcr/kimi_env
source /home/groups/roxanad/sonnet/vcr/kimi_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

# Create all directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME

which python
python --version

# Install dependencies
pip3 install --no-cache-dir --upgrade pip

# Install numpy first from PyPI (before PyTorch changes the index)
pip3 install --no-cache-dir numpy==1.26.4

# PyTorch with CUDA support (2.2.0 is earliest available for cu118 + Python 3.12)
pip3 install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Kimi-VL requires transformers==4.48.2 exactly (newer versions have SDPA compatibility issues)
pip3 install --no-cache-dir transformers==4.48.2
# Use --no-deps for accelerate to prevent it from pulling in a different torch version
pip3 install --no-cache-dir accelerate --no-deps
pip3 install --no-cache-dir psutil  # accelerate dependency
pip3 install --no-cache-dir pillow
pip3 install --no-cache-dir requests
# Kimi-VL processor dependencies (the model's remote code has many undocumented deps)
pip3 install --no-cache-dir tiktoken==0.7.0 protobuf blobfile sentencepiece
# Pin scipy to version with prebuilt wheels for Python 3.12
pip3 install --no-cache-dir pandas scipy==1.11.4 scikit-learn matplotlib

# Flash attention is recommended to avoid OOM errors with long generation
pip3 install --no-cache-dir flash-attn --no-build-isolation

# Run the Kimi-VL test
python /home/groups/roxanad/sonnet/vcr/src/experiments/test_kimi_vl.py
