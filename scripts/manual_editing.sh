#!/bin/bash
#SBATCH --job-name=vcr
#SBATCH --partition=roxanad
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This source file is part of the VCR project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

ml python/3.9.0
ml cuda/11.7.1
ml gcc/14.2.0

# Activate the virtual environment
source /home/groups/roxanad/sonnet/vcr/flam_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

# Create all directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME 

which python

# Install dependencies
pip3 install --no-cache-dir google-genai Pillow tenacity

python /home/groups/roxanad/sonnet/vcr/src/utils/sort_images.py