#!/bin/bash
#SBATCH --job-name=annot_analysis
#SBATCH --partition=roxanad
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ml python/3.9.0
ml gcc/14.2.0

source /home/groups/roxanad/sonnet/vcr/qwen_vl_env/bin/activate

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
source ~/.secrets

mkdir -p logs

# =====================================================
# Configuration
# =====================================================
# Option 1: Single directory with all annotation CSVs
ANNOTATIONS_DIR="human_annotations"

# Option 2: Individual CSVs in their results directories
RESULTS_DIRS=(
    "R1-Onevision-7B_DDI_R1_malignant_prob_detailed_medical"
    "GLM-4.1V-9B-Thinking_DDI_GLM4_malignant_prob_detailed_medical"
    "Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_detailed_medical"
)
METADATA_CSV="/scratch/users/sonnet/ddi/ddi_metadata.csv"
OUTPUT_DIR="analysis_outputs/annotation_analysis"

# LLM Judge settings
RUN_LLM_JUDGE=true  # Set to false to skip LLM judge
JUDGE_BACKEND="claude"  # "claude" or "r1"

# =====================================================
# Step 1: Run Analysis (frequency, co-occurrence, accuracy)
# =====================================================
echo "======================================================================"
echo "STEP 1: Analyzing annotations"
echo "======================================================================"

# Check if annotations directory exists with CSVs
if [ -d "$ANNOTATIONS_DIR" ] && ls "$ANNOTATIONS_DIR"/annotation_matrix_*.csv 1> /dev/null 2>&1; then
    echo "Using batch mode: annotations from $ANNOTATIONS_DIR"

    python /home/groups/roxanad/sonnet/vcr/src/experiments/analyze_annotation_matrix.py \
        --annotations_dir "$ANNOTATIONS_DIR" \
        --results_dirs "${RESULTS_DIRS[@]}" \
        --metadata_csv "$METADATA_CSV" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Using individual mode: annotations from results directories"

    for results_dir in "${RESULTS_DIRS[@]}"; do
        echo "----------------------------------------------------------------------"
        echo "Analyzing: $results_dir"
        echo "----------------------------------------------------------------------"

        model_name=$(echo "$results_dir" | cut -d'_' -f1-3 | sed 's/_DDI.*//')
        annotation_csv="${results_dir}/analysis_outputs/significant_only/annotation_matrix_${model_name}.csv"

        if [ ! -f "$annotation_csv" ]; then
            echo "WARNING: Annotation CSV not found: $annotation_csv"
            continue
        fi

        python /home/groups/roxanad/sonnet/vcr/src/experiments/analyze_annotation_matrix.py \
            --annotation_csv "$annotation_csv" \
            --results_dir "$results_dir" \
            --metadata_csv "$METADATA_CSV"
    done
fi

# =====================================================
# Step 2: Run LLM Judge on annotated traces
# =====================================================
if [ "$RUN_LLM_JUDGE" = true ]; then
    echo ""
    echo "======================================================================"
    echo "STEP 2: Running LLM Judge on annotated traces"
    echo "======================================================================"

    for results_dir in "${RESULTS_DIRS[@]}"; do
        model_name=$(echo "$results_dir" | cut -d'_' -f1-3 | sed 's/_DDI.*//')

        # Determine output directory based on mode
        if [ -d "$ANNOTATIONS_DIR" ]; then
            analysis_dir="${OUTPUT_DIR}/${model_name}"
        else
            analysis_dir="${results_dir}/analysis_outputs/annotation_analysis"
        fi

        judge_input="${analysis_dir}/llm_judge_input.json"
        judge_output_dir="${analysis_dir}/judge_output"

        if [ ! -f "$judge_input" ]; then
            echo "WARNING: LLM judge input not found: $judge_input"
            continue
        fi

        echo "----------------------------------------------------------------------"
        echo "Running LLM judge for: $model_name"
        echo "  Input: $judge_input"
        echo "----------------------------------------------------------------------"

        mkdir -p "$judge_output_dir"

        python /home/groups/roxanad/sonnet/vcr/src/experiments/concept_presence_analysis.py \
            --input_json "$judge_input" \
            --output_dir "$judge_output_dir" \
            --judge_backend "$JUDGE_BACKEND"

        echo ""
    done

    # =====================================================
    # Step 3: Evaluate LLM Judge against ground truth
    # =====================================================
    echo ""
    echo "======================================================================"
    echo "STEP 3: Evaluating LLM Judge accuracy"
    echo "======================================================================"

    for results_dir in "${RESULTS_DIRS[@]}"; do
        model_name=$(echo "$results_dir" | cut -d'_' -f1-3 | sed 's/_DDI.*//')

        # Determine paths
        if [ -d "$ANNOTATIONS_DIR" ]; then
            analysis_dir="${OUTPUT_DIR}/${model_name}"
            annotation_csv="${ANNOTATIONS_DIR}/annotation_matrix_${model_name}.csv"
        else
            analysis_dir="${results_dir}/analysis_outputs/annotation_analysis"
            annotation_csv="${results_dir}/analysis_outputs/significant_only/annotation_matrix_${model_name}.csv"
        fi

        judge_output_dir="${analysis_dir}/judge_output"

        # Find the most recent judge output JSON
        judge_json=$(ls -t "${judge_output_dir}"/*_concept_presence_analysis_*.json 2>/dev/null | head -1)

        if [ -z "$judge_json" ]; then
            echo "WARNING: No judge output found for $model_name"
            continue
        fi

        echo "----------------------------------------------------------------------"
        echo "Evaluating judge for: $model_name"
        echo "  Judge output: $judge_json"
        echo "  Ground truth: $annotation_csv"
        echo "----------------------------------------------------------------------"

        python /home/groups/roxanad/sonnet/vcr/src/experiments/evaluate_judge.py \
            --judge_json "$judge_json" \
            --ground_truth "$annotation_csv" \
            --output_csv "${analysis_dir}/judge_evaluation.csv"

        echo ""
    done
fi

echo "======================================================================"
echo "Done. Analysis outputs saved to:"
if [ -d "$ANNOTATIONS_DIR" ]; then
    echo "  - ${OUTPUT_DIR}/{model_name}/"
else
    for results_dir in "${RESULTS_DIRS[@]}"; do
        echo "  - ${results_dir}/analysis_outputs/annotation_analysis/"
    done
fi
echo ""
echo "Each folder contains:"
echo "  - concept_frequency.csv/png      (frequency analysis)"
echo "  - concept_cooccurrence.csv/png   (co-occurrence matrix)"
echo "  - concept_clusters.png           (hierarchical clustering)"
echo "  - concept_pairs.csv              (pairs with Jaccard > 0.5)"
echo "  - accuracy_by_concept.csv        (accuracy metrics)"
echo "  - accuracy_vs_count.png          (scatter plot)"
echo "  - summary_stats.json             (overall statistics)"
echo "  - llm_judge_input.json           (input for LLM judge)"
if [ "$RUN_LLM_JUDGE" = true ]; then
    echo "  - judge_output/                  (LLM judge predictions)"
    echo "  - judge_evaluation.csv           (judge accuracy vs ground truth)"
fi
echo "======================================================================"
