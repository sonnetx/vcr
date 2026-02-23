#!/bin/bash
#SBATCH --job-name=r1_vcr
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

# Load modules
ml python/3.9.0
ml cuda/11.7.1
ml gcc/14.2.0

# Activate R1/Qwen virtual environment
source /home/groups/roxanad/sonnet/vcr/qwen_vl_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

# Memory management - helps avoid fragmentation OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME
mkdir -p logs

SCRIPT_DIR="/home/groups/roxanad/sonnet/vcr"

# Configuration
skin_tones=("All")
task_definitions=("malignant_prob")
model="R1-Onevision-7B"
layer="model.language_model.layers.27"

for skin_tone in "${skin_tones[@]}"; do
    for task_definition in "${task_definitions[@]}"; do
        echo "================================================"
        echo "Running: $model"
        echo "  Skin tone: $skin_tone"
        echo "  Task: $task_definition"
        echo "  Layer: $layer"
        echo "================================================"

        # Construct results directory name (matches bootstrap script output)
        if [ "$skin_tone" == "All" ]; then
            skin_suffix=""
        else
            skin_suffix="_skin${skin_tone}"
        fi
        results_dir="${model}_DDI_R1${skin_suffix}_${task_definition}_detailed_medical"

        # =====================================================
        # STEP 1: Run VCR bootstrap experiment (5 seeds)
        # =====================================================
        echo ""
        echo "[Step 1/3] Running VCR bootstrap experiment..."
        python "${SCRIPT_DIR}/src/experiments/bootstrap_resample_for_pvalues_r1.py" \
            --model "$model" \
            --filter_skin_tone "$skin_tone" \
            --task_definition "$task_definition" \
            --layer "$layer" \
            --random_seeds 0 1 2 3 4 \
            --top_k_concepts 20

        if [ $? -ne 0 ]; then
            echo "✗ VCR bootstrap failed"
            continue
        fi
        echo "✓ VCR bootstrap completed"

        # =====================================================
        # STEP 2: Run pval_plots.py (cross-seed t-test analysis)
        # =====================================================
        echo ""
        echo "[Step 2/3] Running cross-seed statistical analysis..."
        python "${SCRIPT_DIR}/src/plot/pval_plots.py" \
            --results-dir "$results_dir" \
            --n-concepts 20

        if [ $? -ne 0 ]; then
            echo "✗ pval_plots.py failed"
            continue
        fi
        echo "✓ Statistical analysis completed"

        # =====================================================
        # STEP 3: Omitted features analysis
        # =====================================================
        echo ""
        echo "[Step 3/3] Running omitted features analysis..."
        python "${SCRIPT_DIR}/src/experiments/omitted_features_analysis.py" \
            --results_dir "$results_dir" \
            --output_dir "${results_dir}/omitted_features" \
            --n_images 7 \
            --n_visualize 20

        if [ $? -ne 0 ]; then
            echo "✗ Omitted features analysis failed"
        else
            echo "✓ Omitted features analysis completed"
        fi

        echo ""
        echo "================================================"
        echo "✓ Full pipeline completed for:"
        echo "  Model: $model"
        echo "  Skin tone: $skin_tone"
        echo "  Task: $task_definition"
        echo "  Results: $results_dir"
        echo "================================================"
        echo ""
    done
done

echo "================================================"
echo "All R1 experiments complete!"
echo "================================================"
