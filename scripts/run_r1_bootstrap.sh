#!/bin/bash
#SBATCH --job-name=r1_bootstrap
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Bootstrap-only script for R1 (seeds 0-4)

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME
mkdir -p logs

SCRIPT_DIR="/home/groups/roxanad/sonnet/vcr"

results_dir="R1-Onevision-7B_DDI_R1_malignant_prob_detailed_medical"

echo "================================================"
echo "R1-Onevision-7B Bootstrap (seeds 0-4)"
echo "================================================"

# Delete existing seeds 0-4 to force regeneration
echo "Deleting cached seeds 0-4..."
rm -rf "${results_dir}/seed_0" "${results_dir}/seed_1" "${results_dir}/seed_2" "${results_dir}/seed_3" "${results_dir}/seed_4"
echo "Done."

python "${SCRIPT_DIR}/src/experiments/bootstrap_resample_for_pvalues_r1.py" \
    --model "R1-Onevision-7B" \
    --filter_skin_tone "All" \
    --task_definition "malignant_prob" \
    --layer "model.language_model.layers.27" \
    --random_seeds 0 1 2 3 4 \
    --top_k_concepts 20

if [ $? -ne 0 ]; then
    echo "✗ R1 bootstrap failed"
    exit 1
fi

echo "================================================"
echo "✓ R1 bootstrap complete!"
echo "================================================"
