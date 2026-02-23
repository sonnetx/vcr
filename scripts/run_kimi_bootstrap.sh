#!/bin/bash
#SBATCH --job-name=kimi_bootstrap
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Bootstrap-only script for Kimi (seeds 0-4)

# Load modules
ml gcc/14.2.0
ml python/3.12.1
ml cuda/11.7.1

# Activate Kimi virtual environment
source /home/groups/roxanad/sonnet/vcr/kimi_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME
mkdir -p logs

SCRIPT_DIR="/home/groups/roxanad/sonnet/vcr"

results_dir="Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_detailed_medical"

echo "================================================"
echo "Kimi-VL-A3B-Thinking Bootstrap (seeds 0-4)"
echo "================================================"

# Delete existing seeds 0-4 to force regeneration
echo "Deleting cached seeds 0-4..."
rm -rf "${results_dir}/seed_0" "${results_dir}/seed_1" "${results_dir}/seed_2" "${results_dir}/seed_3" "${results_dir}/seed_4"
echo "Done."

python "${SCRIPT_DIR}/src/experiments/bootstrap_resample_for_pvalues_kimi.py" \
    --model "Kimi-VL-A3B-Thinking" \
    --filter_skin_tone "All" \
    --task_definition "malignant_prob" \
    --random_seeds 0 1 2 3 4 \
    --top_k_concepts 20

if [ $? -ne 0 ]; then
    echo "✗ Kimi bootstrap failed"
    exit 1
fi

echo "================================================"
echo "✓ Kimi bootstrap complete!"
echo "================================================"
