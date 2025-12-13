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

# Run all combinations of model, use_demos, and filter_skin_tone

models=("MedFlamingo") # ("OpenFlamingo-4B" "OpenFlamingo-3B-Instruct")
skin_tones=("All" "12" "56")
task_definitions=("contrastive" "malignant_prob") # "malignant_prob"

for model in "${models[@]}"; do
    for skin_tone in "${skin_tones[@]}"; do
        for task_definition in "${task_definitions[@]}"; do
            # Run without demos
            echo "================================================"
            echo "Running: $model, skin_tone=$skin_tone, no ICL"
            echo "================================================"
            python /home/groups/roxanad/sonnet/vcr/src/experiments/bootstrap_resample_for_pvalues.py \
                --model "$model" \
                --filter_skin_tone "$skin_tone" \
                --task_definition "$task_definition"
            
            # Run with demos
            echo "================================================"
            echo "Running: $model, skin_tone=$skin_tone, with ICL"
            echo "================================================"
            python /home/groups/roxanad/sonnet/vcr/src/experiments/bootstrap_resample_for_pvalues.py \
                --model "$model" \
                --filter_skin_tone "$skin_tone" \
                --task_definition "$task_definition" \
                --use_demos
        done
    done
done

echo "================================================"
echo "All experiments complete!"
echo "================================================"