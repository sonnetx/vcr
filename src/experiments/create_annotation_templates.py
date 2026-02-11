"""
Generate blank annotation CSV templates for manual concept presence labeling.

Reads inspection JSONs and outputs CSVs with one row per trace,
metadata columns, and one empty column per concept for the annotator to fill (1/0).

Usage:
    # Single model + direction:
    python src/experiments/create_annotation_templates.py \
        --results_dir R1-Onevision-7B_DDI_R1_malignant_prob_simple_binary \
        --direction positive \
        --num_traces 20

    # All 3 models, both directions:
    python src/experiments/create_annotation_templates.py --all --num_traces 20
"""

import json
import argparse
import pandas as pd
from pathlib import Path


MODELS = [
    "R1-Onevision-7B_DDI_R1_malignant_prob_simple_binary",
    "GLM-4.1V-9B-Thinking_DDI_GLM4_malignant_prob_simple_binary",
    "Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_simple_binary",
]


def generate_template(results_dir, direction, num_traces, output_path=None):
    results_dir = Path(results_dir)
    inspection_path = results_dir / f'inspection_{direction}.json'

    if not inspection_path.exists():
        print(f"  SKIP: {inspection_path} not found")
        return None

    with open(inspection_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    concepts = [c['concept'] for c in data['top_concepts']]
    traces = data['reasoning_traces'][:num_traces]

    rows = []
    for trace in traces:
        row = {
            'trace_index': trace['index'],
            'image_path': trace['image_path'],
            'label': trace['label'],
            'reasoning': trace['reasoning'],
        }
        for concept in concepts:
            row[concept] = ''
        rows.append(row)

    df = pd.DataFrame(rows)

    if output_path is None:
        output_path = results_dir / f'annotations_{direction}.csv'

    df.to_csv(output_path, index=False)
    print(f"  Created: {output_path}")
    print(f"    {len(traces)} traces x {len(concepts)} concepts")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate blank annotation CSV templates for manual concept presence labeling'
    )
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Single results directory')
    parser.add_argument('--direction', type=str, default=None,
                        choices=['positive', 'negative'],
                        help='Concept direction')
    parser.add_argument('--num_traces', type=int, default=20,
                        help='Number of traces to include (default: 20)')
    parser.add_argument('--all', action='store_true',
                        help='Generate for all 3 models x both directions')
    args = parser.parse_args()

    if args.all:
        for model_dir in MODELS:
            for direction in ['positive', 'negative']:
                print(f"\n--- {model_dir} / {direction} ---")
                generate_template(model_dir, direction, args.num_traces)
    elif args.results_dir and args.direction:
        generate_template(args.results_dir, args.direction, args.num_traces)
    else:
        parser.error('Provide --results_dir and --direction, or use --all')


if __name__ == '__main__':
    main()
