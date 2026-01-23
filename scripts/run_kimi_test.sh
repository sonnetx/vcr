#!/bin/bash
#SBATCH --job-name=kimi_test
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

# Run the test
python /home/groups/roxanad/sonnet/vcr/src/experiments/test_kimi_vl.py
