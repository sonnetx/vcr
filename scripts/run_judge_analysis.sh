#!/bin/bash
#SBATCH --job-name=judge_analysis
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Production LLM Judge Analysis Pipeline
# Runs the LLM judge on many reasoning traces across all seeds
# and computes frequency, co-occurrence, and consistency statistics.
#
# Usage:
#   sbatch scripts/run_judge_analysis.sh
#
# This is the PRODUCTION workflow for analyzing concept presence.
# For validating the judge against human annotations, use run_evaluate_judge.sh

ml python/3.9.0
ml gcc/14.2.0

source /home/groups/roxanad/sonnet/vcr/qwen_vl_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
source ~/.secrets

mkdir -p logs

# =====================================================
# Configuration
# =====================================================
RESULTS_DIRS=(
    "R1-Onevision-7B_DDI_R1_malignant_prob_detailed_medical"
    "GLM-4.1V-9B-Thinking_DDI_GLM4_malignant_prob_detailed_medical"
    "Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_detailed_medical"
)

# Number of traces to analyze per model (0 = all)
MAX_TRACES=300

# LLM Judge settings
JUDGE_BACKEND="claude"  # "claude", "gpt", or "r1"

# Set to true to force regeneration of inspection JSON (e.g., after dedup fix)
REGENERATE=true

# =====================================================
# Process each model
# =====================================================
for results_dir in "${RESULTS_DIRS[@]}"; do
    echo "======================================================================"
    echo "Processing ($JUDGE_BACKEND): $results_dir"
    echo "======================================================================"

    model_name=$(echo "$results_dir" | cut -d'_' -f1-3 | sed 's/_DDI.*//')

    # Shared input dir (inspection JSON is judge-agnostic)
    shared_dir="${results_dir}/analysis_outputs/judge_analysis"
    mkdir -p "$shared_dir"

    # Judge-specific output dir
    output_dir="${shared_dir}/${JUDGE_BACKEND}"
    mkdir -p "$output_dir"

    # -------------------------------------------------
    # Step 1: Create inspection JSON with traces from ALL seeds
    # -------------------------------------------------
    echo ""
    echo "Step 1: Creating inspection JSON (all seeds, max ${MAX_TRACES} traces)..."

    inspection_json="${shared_dir}/inspection_all_seeds.json"

    # Archive old inspection JSON if regenerating
    if [ "$REGENERATE" = true ] && [ -f "$inspection_json" ]; then
        archive_name="${inspection_json%.json}_$(date +%Y%m%d_%H%M%S).json.bak"
        mv "$inspection_json" "$archive_name"
        echo "  Archived old inspection JSON -> $(basename $archive_name)"
    fi

    if [ ! -f "$inspection_json" ]; then
        python /home/groups/roxanad/sonnet/vcr/src/experiments/create_inspection_json.py \
            --results_dir "$results_dir" \
            --annotations_dir "human_annotations" \
            --all_seeds \
            --max_traces "$MAX_TRACES" \
            --output "$inspection_json"
    else
        echo "  Using existing: $inspection_json"
    fi

    if [ ! -f "$inspection_json" ]; then
        echo "ERROR: Failed to create inspection JSON"
        continue
    fi

    # -------------------------------------------------
    # Step 2: Run LLM Judge
    # -------------------------------------------------
    echo ""
    echo "Step 2: Running LLM judge (${JUDGE_BACKEND})..."

    python /home/groups/roxanad/sonnet/vcr/src/experiments/concept_presence_analysis.py \
        --input_json "$inspection_json" \
        --output_dir "$output_dir" \
        --judge_backend "$JUDGE_BACKEND"

    # Find the most recent judge output
    judge_json=$(ls -t "${output_dir}"/*_concept_presence_analysis_*.json 2>/dev/null | head -1)

    if [ -z "$judge_json" ]; then
        echo "ERROR: No judge output found"
        continue
    fi

    # -------------------------------------------------
    # Step 3: Analyze judge output
    # -------------------------------------------------
    echo ""
    echo "Step 3: Analyzing judge output..."

    python /home/groups/roxanad/sonnet/vcr/src/experiments/analyze_judge_output.py \
        --judge_json "$judge_json" \
        --output_dir "$output_dir"

    echo ""
    echo "Completed: $model_name (judge: $JUDGE_BACKEND)"
    echo "  Outputs saved to: $output_dir"
    echo ""
done

echo "======================================================================"
echo "All models processed (judge: $JUDGE_BACKEND)."
echo ""
echo "Output structure for each model:"
echo "  {results_dir}/analysis_outputs/judge_analysis/"
echo "    - inspection_all_seeds.json              (shared input)"
echo "    - ${JUDGE_BACKEND}/"
echo "      - *_concept_presence_analysis_*.json   (judge predictions)"
echo "      - concept_frequency.csv/png            (frequency analysis)"
echo "      - concept_cooccurrence.csv/png         (co-occurrence matrix)"
echo "      - concept_pairs.csv                    (pairs with Jaccard > 0.3)"
echo "      - cross_seed_consistency.csv           (consistency across seeds)"
echo "      - summary_stats.json                   (overall statistics)"
echo "======================================================================"
