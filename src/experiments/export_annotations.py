"""
Export LLM judge output as a CSV for manual annotation/correction.

Generates a spreadsheet where each row is a reasoning trace and each column
is a concept. Cells are pre-filled with the judge's 1/0 predictions.
Annotators correct mistakes, then run evaluate_judge.py to compute accuracy.

Usage:
    python src/experiments/export_annotations.py \
        --judge_json <path_to_concept_presence_analysis.json> \
        --output annotations.csv \
        --num_traces 20
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Export judge output as annotation CSV')
    parser.add_argument('--judge_json', type=str, required=True,
                        help='Path to concept_presence_analysis JSON output')
    parser.add_argument('--output', type=str, default='annotations.csv',
                        help='Output CSV path (default: annotations.csv)')
    parser.add_argument('--num_traces', type=int, default=0,
                        help='Number of traces to include (0 = all)')
    args = parser.parse_args()

    with open(args.judge_json, 'r') as f:
        data = json.load(f)

    concepts = [c['concept'] for c in data['concepts']]
    traces = data['trace_analyses']

    if args.num_traces > 0:
        traces = traces[:args.num_traces]

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header: trace_index, image_path, label, reasoning (truncated), then one column per concept
        header = ['trace_index', 'image_path', 'label', 'reasoning'] + concepts
        writer.writerow(header)

        for trace in traces:
            ra = trace['reasoning_analysis']
            present_set = set(ra.get('concepts_present', []))

            # Truncate reasoning for readability in spreadsheet
            reasoning_text = ra.get('raw_llm_response', '')
            # Get original reasoning from the trace if available
            # Use image_path as identifier since we don't store full reasoning in output
            reasoning_preview = reasoning_text[:200] + '...' if len(reasoning_text) > 200 else reasoning_text

            row = [
                trace['trace_index'],
                trace.get('image_path', ''),
                trace.get('label', ''),
                reasoning_preview,
            ]

            # Add 1/0 per concept based on judge prediction
            for concept in concepts:
                row.append(1 if concept in present_set else 0)

            writer.writerow(row)

    print(f"Exported {len(traces)} traces x {len(concepts)} concepts to {args.output}")
    print(f"Edit the CSV to correct any wrong labels, then run:")
    print(f"  python src/experiments/evaluate_judge.py \\")
    print(f"      --judge_json {args.judge_json} \\")
    print(f"      --ground_truth {args.output}")


if __name__ == '__main__':
    main()
