#!/bin/bash
#SBATCH --job-name=kimi_vcr
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This source file is part of the VCR project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# Load modules FIRST (required for shared libraries)
ml gcc/14.2.0
ml python/3.12.1
ml cuda/11.7.1
# Note: rust is only needed for building tiktoken, not for running

# Then activate venv
source /home/groups/roxanad/sonnet/vcr/kimi_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

# Create directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME
mkdir -p logs

# Run the VCR experiment with Kimi-VL
# Default: 5 random seeds, malignant_prob task definition, auto-detect layer
python /home/groups/roxanad/sonnet/vcr/src/experiments/bootstrap_resample_for_pvalues_kimi.py \
    --model Kimi-VL-A3B-Thinking \
    --filter_skin_tone All \
    --random_seeds 0 1 2 3 4 \
    --task_definition malignant_prob \
    --top_k_concepts 20
