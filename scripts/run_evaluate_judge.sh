#!/bin/bash
#SBATCH --job-name=evaluate_judge
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# LLM Judge Validation Pipeline
# Tests the LLM judge accuracy against human annotations on a small set.
# Use this to validate judge quality BEFORE running large-scale analysis.
#
# Usage:
#   sbatch scripts/run_evaluate_judge.sh
#
# For production analysis on many traces, use run_judge_analysis.sh instead.

ml python/3.9.0
ml gcc/14.2.0

source /home/groups/roxanad/sonnet/vcr/qwen_vl_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
source ~/.secrets

mkdir -p logs

# =====================================================
# Configuration
# =====================================================
ANNOTATIONS_DIR="human_annotations"

RESULTS_DIRS=(
    "R1-Onevision-7B_DDI_R1_malignant_prob_detailed_medical"
    "GLM-4.1V-9B-Thinking_DDI_GLM4_malignant_prob_detailed_medical"
    "Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_detailed_medical"
)

OUTPUT_DIR="analysis_outputs/judge_evaluation"
JUDGE_BACKEND="gpt"  # "claude", "gpt", or "r1"

# =====================================================
# Process each model
# =====================================================
for results_dir in "${RESULTS_DIRS[@]}"; do
    echo "======================================================================"
    echo "Evaluating judge ($JUDGE_BACKEND) for: $results_dir"
    echo "======================================================================"

    model_name=$(echo "$results_dir" | cut -d'_' -f1-3 | sed 's/_DDI.*//')

    # Find annotation matrix
    if [ -d "$ANNOTATIONS_DIR" ]; then
        annotation_csv="${ANNOTATIONS_DIR}/annotation_matrix_${model_name}.csv"
    else
        annotation_csv="${results_dir}/analysis_outputs/significant_only/annotation_matrix_${model_name}.csv"
    fi

    if [ ! -f "$annotation_csv" ]; then
        echo "WARNING: Annotation CSV not found: $annotation_csv"
        echo "Skipping $model_name"
        continue
    fi

    # analyze_annotation_matrix.py appends model_name as subdirectory,
    # so pass OUTPUT_DIR directly (not OUTPUT_DIR/model_name)
    annotation_output_dir="${OUTPUT_DIR}"

    # Judge outputs go under a judge-specific subdirectory
    output_model_dir="${OUTPUT_DIR}/${model_name}/${JUDGE_BACKEND}"
    mkdir -p "$output_model_dir"

    # -------------------------------------------------
    # Step 1: Create inspection JSON from annotated traces only
    # -------------------------------------------------
    echo ""
    echo "Step 1: Exporting annotated traces for judge..."

    # analyze_annotation_matrix.py creates {output_dir}/{model_name}/llm_judge_input.json
    python /home/groups/roxanad/sonnet/vcr/src/experiments/analyze_annotation_matrix.py \
        --annotation_csv "$annotation_csv" \
        --results_dir "$results_dir" \
        --metadata_csv "/scratch/users/sonnet/ddi/ddi_metadata.csv" \
        --output_dir "$annotation_output_dir"

    judge_input="${OUTPUT_DIR}/${model_name}/llm_judge_input.json"

    if [ ! -f "$judge_input" ]; then
        echo "ERROR: Failed to create judge input at: $judge_input"
        continue
    fi

    # -------------------------------------------------
    # Step 2: Run LLM Judge on annotated traces
    # -------------------------------------------------
    echo ""
    echo "Step 2: Running LLM judge ($JUDGE_BACKEND) on annotated traces..."

    python /home/groups/roxanad/sonnet/vcr/src/experiments/concept_presence_analysis.py \
        --input_json "$judge_input" \
        --output_dir "$output_model_dir" \
        --judge_backend "$JUDGE_BACKEND"

    # Find the most recent judge output
    judge_json=$(ls -t "${output_model_dir}"/*_concept_presence_analysis_*.json 2>/dev/null | head -1)

    if [ -z "$judge_json" ]; then
        echo "ERROR: No judge output found"
        continue
    fi

    # -------------------------------------------------
    # Step 3: Evaluate judge against human ground truth
    # -------------------------------------------------
    echo ""
    echo "Step 3: Evaluating judge accuracy..."

    python /home/groups/roxanad/sonnet/vcr/src/experiments/evaluate_judge.py \
        --judge_json "$judge_json" \
        --ground_truth "$annotation_csv" \
        --output_csv "${output_model_dir}/judge_evaluation.csv"

    echo ""
    echo "Completed: $model_name (judge: $JUDGE_BACKEND)"
    echo "  Results: ${output_model_dir}/"
    echo ""
done

echo "======================================================================"
echo "Judge evaluation complete (judge: $JUDGE_BACKEND)."
echo ""
echo "Output structure:"
echo "  ${OUTPUT_DIR}/{model_name}/{judge_backend}/"
echo "    - *_concept_presence_analysis_*.json (judge predictions)"
echo "    - judge_evaluation.csv               (per-trace comparison)"
echo ""
echo "  ${OUTPUT_DIR}/{model_name}/"
echo "    - llm_judge_input.json               (shared input across judges)"
echo ""
echo "Review the console output above for:"
echo "  - Overall accuracy, precision, recall, F1"
echo "  - Per-concept breakdown"
echo "  - Common mismatches (FP/FN)"
echo ""
echo "If judge accuracy is acceptable, run production analysis:"
echo "  sbatch scripts/run_judge_analysis.sh"
echo "======================================================================"
