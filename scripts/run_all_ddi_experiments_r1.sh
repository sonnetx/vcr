#!/bin/bash
#SBATCH --job-name=vcr_r1
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
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

# Activate R1/Qwen virtual environment
source /home/groups/roxanad/sonnet/vcr/qwen_vl_env/bin/activate

which python
nvcc --version

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create all directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME logs

# Run all combinations of model, skin_tone, task_definition, and layer
# Note: R1 models don't use ICL demos in the same way as Flamingo

models=("R1-Onevision-7B")
skin_tones=("All" "12" "56")
task_definitions=("malignant_prob" "contrastive")
layers=("model.language_model.layers.27")  # Add more layers if needed: "model.language_model.layers.15"

for model in "${models[@]}"; do
    for skin_tone in "${skin_tones[@]}"; do
        for task_definition in "${task_definitions[@]}"; do
            for layer in "${layers[@]}"; do
                echo "================================================"
                echo "Running: $model"
                echo "  Skin tone: $skin_tone"
                echo "  Task: $task_definition"
                echo "  Layer: $layer"
                echo "================================================"

                python /home/groups/roxanad/sonnet/vcr/src/experiments/bootstrap_resample_for_pvalues_r1.py \
                    --model "$model" \
                    --filter_skin_tone "$skin_tone" \
                    --task_definition "$task_definition" \
                    --layer "$layer" \
                    --random_seeds 0 1 2 3 4 \
                    --top_k_concepts 20

                # Check exit status
                if [ $? -eq 0 ]; then
                    echo "✓ Completed successfully"
                else
                    echo "✗ Failed with exit code $?"
                fi
                echo ""
            done
        done
    done
done

echo "================================================"
echo "All R1 experiments complete!"
echo "================================================"
