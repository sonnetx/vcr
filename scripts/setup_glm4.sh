#!/bin/bash
#SBATCH --job-name=glm4_setup
#SBATCH --partition=roxanad
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

# GLM-4.1V requires Python 3.10+ and transformers >= 4.57.1
# Use GCC 10.x for CUDA 11.x compatibility (CUDA 11.x requires GCC < 12.0)
ml gcc/10.3.0
ml python/3.12.1
ml cuda/11.7.1
ml rust/1.81.0  # Required to build tiktoken

# Remove existing environment if it exists (in case of partial/failed install)
rm -rf /home/groups/roxanad/sonnet/vcr/glm4_env

# Create a separate virtual environment for GLM4 (different transformers version than Kimi)
python3 -m venv /home/groups/roxanad/sonnet/vcr/glm4_env
source /home/groups/roxanad/sonnet/vcr/glm4_env/bin/activate

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
pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first from PyPI (before PyTorch changes the index)
pip3 install --no-cache-dir numpy==1.26.4
pip3 install --no-cache-dir einops
pip3 install --no-cache-dir open_clip_torch

# PyTorch with CUDA support (2.4.0 needed for transformers >= 4.57 which uses torch.is_autocast_enabled(device_type))
pip3 install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# GLM-4.1V requires transformers >= 4.57.1 for Glm4vForConditionalGeneration class
pip3 install --no-cache-dir "transformers>=4.57.1"
# Use --no-deps for accelerate to prevent it from pulling in a different torch version
pip3 install --no-cache-dir accelerate --no-deps
pip3 install --no-cache-dir psutil  # accelerate dependency
pip3 install --no-cache-dir pillow
pip3 install --no-cache-dir requests
# GLM processor dependencies
# Pin tiktoken to 0.7.0 - newer versions require Rust edition 2024 which isn't stable in Rust 1.81
pip3 install --no-cache-dir tiktoken==0.7.0 protobuf sentencepiece
# Pin pandas, scipy, scikit-learn to versions with prebuilt wheels for Python 3.12
# Use --only-binary to prevent source builds
pip3 install --no-cache-dir --only-binary :all: pandas==2.2.3
pip3 install --no-cache-dir --only-binary :all: scipy==1.12.0
pip3 install --no-cache-dir --only-binary :all: scikit-learn==1.4.2
pip3 install --no-cache-dir matplotlib

# Visualization dependencies
pip3 install --no-cache-dir matplotlib-venn seaborn

# Anthropic API (Claude judge)
pip3 install --no-cache-dir anthropic

# Flash attention is optional - skip it to avoid build issues with GCC/CUDA version mismatches
# The model will fall back to standard attention (may be slower but works fine for inference)
# pip3 install --no-cache-dir flash-attn --no-build-isolation

# Verify transformers version has the required class
python -c "from transformers import Glm4vForConditionalGeneration; print('Glm4vForConditionalGeneration available!')"

# Run a quick test to verify the environment
echo ""
echo "GLM4 environment setup complete!"
echo "To test, run: python /home/groups/roxanad/sonnet/vcr/src/experiments/test_glm4.py"
