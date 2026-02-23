"""
Evaluate LLM judge accuracy against human ground-truth annotations.

Compares the judge's concept presence predictions against human annotations
and reports precision, recall, F1, and per-concept accuracy.

Supports two ground truth formats:
1. Annotation matrix (concepts as rows, traces as columns) - from create_annotation_matrix.py
2. Legacy format (traces as rows, concepts as columns) - from export_annotations.py

Usage:
    python src/experiments/evaluate_judge.py \
        --judge_json <path_to_concept_presence_analysis.json> \
        --ground_truth annotation_matrix.csv

    # Or with legacy format:
    python src/experiments/evaluate_judge.py \
        --judge_json <path_to_concept_presence_analysis.json> \
        --ground_truth annotations_corrected.csv \
        --legacy_format
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_trace_column(col_name):
    """Parse trace column name to extract seed, index.
    Format: s{seed}_i{index}_{image_name}
    """
    match = re.match(r'^s(\d+)_i(\d+)_(.+)$', col_name)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def load_annotation_matrix(csv_path):
    """Load annotation matrix (concepts as rows, traces as columns).

    Returns:
        dict: {(seed, index): {concept: 0/1}}
        list: concept names
    """
    df = pd.read_csv(csv_path)

    # Identify metadata columns vs trace columns
    metadata_cols = ['direction', 'concept', 'definition', 'mean_dd', 'p_value']
    trace_columns = [c for c in df.columns if c not in metadata_cols]

    # Filter to only annotated traces (columns with at least one 1)
    trace_data = df[trace_columns]
    col_sums = (trace_data.fillna(0) == 1).sum(axis=0)
    annotated_cols = [col for col, s in zip(trace_columns, col_sums) if s > 0]

    if len(annotated_cols) < len(trace_columns):
        print(f"  Filtering to {len(annotated_cols)}/{len(trace_columns)} annotated traces")
    trace_columns = annotated_cols

    # Get concept names (use 'definition' if available for matching with judge)
    if 'definition' in df.columns:
        raw_concepts = df['definition'].tolist()
    else:
        raw_concepts = df['concept'].tolist()

    raw_concept_names = df['concept'].tolist()  # Original names for display

    # Deduplicate concepts with identical definitions
    # Build mapping: definition -> list of original row indices
    seen = {}  # definition -> (new_index, display_name)
    old_to_new = {}
    deduped_concepts = []
    deduped_names = []

    for i, concept_def in enumerate(raw_concepts):
        key = concept_def.strip() if isinstance(concept_def, str) else str(concept_def)
        if key in seen:
            old_to_new[i] = seen[key]
        else:
            new_idx = len(deduped_concepts)
            seen[key] = new_idx
            old_to_new[i] = new_idx
            deduped_concepts.append(concept_def)
            deduped_names.append(raw_concept_names[i])

    n_removed = len(raw_concepts) - len(deduped_concepts)
    if n_removed > 0:
        print(f"  Deduped {len(raw_concepts)} -> {len(deduped_concepts)} ground truth concepts ({n_removed} duplicates)")

    concepts = deduped_concepts
    concept_names = deduped_names

    # Build ground truth dict, merging duplicate concept rows via OR
    gt_labels = {}
    for col in trace_columns:
        parsed = parse_trace_column(col)
        if not parsed:
            continue

        seed, idx, _ = parsed
        key = (seed, idx)

        gt_labels[key] = {c: 0 for c in concepts}
        for i, concept_def in enumerate(raw_concepts):
            val = 1 if df[col].iloc[i] == 1 else 0
            new_idx = old_to_new[i]
            # OR: if any duplicate row has a 1, the merged concept is 1
            if val == 1:
                gt_labels[key][concepts[new_idx]] = 1

    return gt_labels, concepts, concept_names


def load_legacy_csv(csv_path, concepts):
    """Load legacy format CSV (traces as rows, concepts as columns)."""
    gt_labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['trace_index'])
            gt_labels[idx] = {c: int(row.get(c, 0)) for c in concepts}
    return gt_labels


def load_judge_output(json_path):
    """Load judge JSON output.

    Returns:
        dict: {(seed, index) or index: {concept: 0/1}}
        list: concept definitions used by judge
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get concepts (could be definitions or names)
    concepts = [c['concept'] for c in data['concepts']]

    # Also get original names if available
    original_names = [c.get('original_name', c['concept']) for c in data['concepts']]

    judge_preds = {}
    for trace in data['trace_analyses']:
        # Try to get seed info
        seed = trace.get('seed', 0)
        idx = trace['trace_index']

        present = set(trace['reasoning_analysis'].get('concepts_present', []))
        preds = {c: (1 if c in present else 0) for c in concepts}

        # Store with both key formats for flexibility
        judge_preds[(seed, idx)] = preds
        judge_preds[idx] = preds  # Legacy format

    return judge_preds, concepts, original_names


def main():
    parser = argparse.ArgumentParser(description='Evaluate judge accuracy against ground truth')
    parser.add_argument('--judge_json', type=str, required=True,
                        help='Path to concept_presence_analysis JSON output')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth CSV (annotation matrix or legacy format)')
    parser.add_argument('--legacy_format', action='store_true',
                        help='Use legacy CSV format (traces as rows)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Save detailed comparison to CSV')
    args = parser.parse_args()

    print("=" * 70)
    print("LLM Judge Evaluation")
    print("=" * 70)
    print(f"Judge output: {args.judge_json}")
    print(f"Ground truth: {args.ground_truth}")
    print()

    # Load judge output
    judge_preds, judge_concepts, original_names = load_judge_output(args.judge_json)

    # Load ground truth
    if args.legacy_format:
        gt_labels = load_legacy_csv(args.ground_truth, judge_concepts)
        concepts = judge_concepts
        concept_names = original_names
        use_tuple_keys = False
    else:
        gt_labels, concepts, concept_names = load_annotation_matrix(args.ground_truth)
        use_tuple_keys = True

    print(f"Judge concepts: {len(judge_concepts)}")
    print(f"Ground truth concepts: {len(concepts)}")
    print()

    # Compare
    tp = fp = fn = tn = 0
    per_concept = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    mismatches = []
    detailed_results = []

    # Build a mapping from concept definition to display name
    concept_to_display = {}
    for i, concept in enumerate(concepts):
        display_name = concept_names[i] if i < len(concept_names) else concept
        concept_to_display[concept] = display_name

    matched_traces = 0
    for key in sorted(gt_labels.keys()):
        if key not in judge_preds:
            continue

        matched_traces += 1

        # Match by concept string, not by position
        for concept in concepts:
            # Both ground truth and judge use the same concept string (definition)
            pred = judge_preds[key].get(concept, 0)
            gold = gt_labels[key].get(concept, 0)

            # Use display name for reporting
            display_name = concept_to_display.get(concept, concept)

            if pred == 1 and gold == 1:
                tp += 1
                per_concept[display_name]['tp'] += 1
                result = 'TP'
            elif pred == 1 and gold == 0:
                fp += 1
                per_concept[display_name]['fp'] += 1
                mismatches.append((key, display_name, 'FP', 'judge=1, gold=0'))
                result = 'FP'
            elif pred == 0 and gold == 1:
                fn += 1
                per_concept[display_name]['fn'] += 1
                mismatches.append((key, display_name, 'FN', 'judge=0, gold=1'))
                result = 'FN'
            else:
                tn += 1
                per_concept[display_name]['tn'] += 1
                result = 'TN'

            detailed_results.append({
                'trace_key': str(key),
                'concept': display_name,
                'judge_pred': pred,
                'ground_truth': gold,
                'result': result
            })

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"Traces matched: {matched_traces}/{len(gt_labels)}")
    print(f"Concepts: {len(concepts)}")
    print(f"Total cells evaluated: {total}")
    print()
    print("=" * 50)
    print("OVERALL METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.1%}  ({tp + tn}/{total})")
    print(f"Precision: {precision:.1%}  ({tp}/{tp + fp}) - When judge says present, how often correct?")
    print(f"Recall:    {recall:.1%}  ({tp}/{tp + fn}) - Of actual present, how many did judge find?")
    print(f"F1:        {f1:.3f}")
    print()
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Per-concept breakdown
    print()
    print("=" * 50)
    print("PER-CONCEPT BREAKDOWN")
    print("=" * 50)
    print(f"  {'Concept':30s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s}")
    print("  " + "-" * 75)

    sorted_concepts = sorted(per_concept.keys(),
                             key=lambda c: per_concept[c]['fp'] + per_concept[c]['fn'],
                             reverse=True)

    for concept in sorted_concepts:
        s = per_concept[concept]
        t = s['tp'] + s['fp'] + s['fn'] + s['tn']
        acc = (s['tp'] + s['tn']) / t if t else 0
        prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) else float('nan')
        rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) else float('nan')

        prec_str = f"{prec:5.1%}" if not pd.isna(prec) else "  n/a"
        rec_str = f"{rec:5.1%}" if not pd.isna(rec) else "  n/a"

        name_display = concept[:30] if len(concept) <= 30 else concept[:27] + "..."
        print(f"  {name_display:30s}  {acc:5.1%}  {prec_str}  {rec_str}  {s['tp']:3d} {s['fp']:3d} {s['fn']:3d} {s['tn']:3d}")

    # Show mismatches
    if mismatches:
        print()
        print("=" * 50)
        print(f"MISMATCHES ({len(mismatches)} total)")
        print("=" * 50)
        for key, concept, mtype, desc in mismatches[:30]:
            key_str = f"s{key[0]}_i{key[1]}" if isinstance(key, tuple) else f"trace_{key}"
            name_display = concept[:25] if len(concept) <= 25 else concept[:22] + "..."
            print(f"  {key_str:15s} | {name_display:25s} | {mtype} ({desc})")
        if len(mismatches) > 30:
            print(f"  ... and {len(mismatches) - 30} more")

    # Save detailed CSV if requested
    if args.output_csv:
        output_df = pd.DataFrame(detailed_results)
        output_df.to_csv(args.output_csv, index=False)
        print()
        print(f"Detailed comparison saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
