#!/bin/bash
#SBATCH --job-name=vcr
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=80G
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

which python
nvcc --version

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

# Create all directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME 
which python

models=("OpenFlamingo-4B" "OpenFlamingo-3B-Instruct")
task_definitions=("contrastive" "malignant_prob")

for model in "${models[@]}"; do
    for task_definition in "${task_definitions[@]}"; do
        echo "================================================"
        echo "Running: $model, task_definition=$task_definition"
        echo "================================================"
        python /home/groups/roxanad/sonnet/vcr/src/experiments/lesion_overlay_experiment.py \
            --model "$model" \
            --task_definition "$task_definition" \
            --background_dir /home/groups/roxanad/sonnet/vcr/background \
            --lesion_dir /home/groups/roxanad/sonnet/vcr/lesions \
            --original_dir /scratch/users/sonnet/ddi \
            --results_dir /scratch/users/sonnet/image_overlay 
    done
done

echo "================================================"
echo "All experiments complete!"
echo "================================================"