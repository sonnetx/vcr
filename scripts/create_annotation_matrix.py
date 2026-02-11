#!/usr/bin/env python
"""
Create annotation matrix CSV for manual concept presence labeling.

Creates a CSV with:
- Rows: significant concepts (positive + negative combined)
- Columns: randomly sampled reasoning traces across all seeds
- Cells: empty for manual annotation (1 = concept present, 0 = absent)

The script:
1. Loads significant concepts from analysis_outputs/
2. Scans all seed_* folders for reasoning_traces.json
3. Randomly samples N traces across all seeds
4. Creates annotation matrix + trace text reference file
5. Outputs to analysis_outputs/significant_only/ for easy annotation alongside images

Usage:
    python scripts/create_annotation_matrix.py \
        --results_dir R1-Onevision-7B_DDI_R1_malignant_prob_detailed_medical \
        --n_traces 100
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def load_significant_concepts(results_dir):
    """Load significant concepts from analysis CSVs."""
    results_dir = Path(results_dir)
    analysis_dir = results_dir / "analysis_outputs"

    all_concepts = []

    for direction in ["positive", "negative"]:
        # Find the concepts CSV - use most recent if multiple exist
        csv_files = list(analysis_dir.glob(f"top_*_{direction}_concepts.csv"))
        if not csv_files:
            print(f"WARNING: No {direction} concepts CSV found")
            continue

        # Sort by modification time, most recent first
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        csv_path = csv_files[0]

        if len(csv_files) > 1:
            print(f"  Found {len(csv_files)} {direction} CSVs, using most recent: {csv_path.name}")

        df = pd.read_csv(csv_path)

        # Filter to significant only
        if 'significant' in df.columns:
            df_sig = df[df['significant'] == True].copy()
        else:
            print(f"WARNING: No 'significant' column in {csv_path.name}, using all concepts")
            df_sig = df.copy()

        # Add direction and collect
        df_sig['direction'] = direction
        all_concepts.append(df_sig)
        print(f"  {direction}: {len(df_sig)} significant concepts")

    if not all_concepts:
        return pd.DataFrame()

    combined = pd.concat(all_concepts, ignore_index=True)
    return combined


def load_reasoning_traces(results_dir):
    """Load all reasoning traces from all seed folders."""
    results_dir = Path(results_dir)

    all_traces = []

    # Find all seed folders
    seed_dirs = sorted([d for d in results_dir.iterdir()
                        if d.is_dir() and re.match(r'^seed_\d+$', d.name)])

    print(f"Found {len(seed_dirs)} seed folders")

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.replace('seed_', ''))

        # Load reasoning traces
        reasoning_file = seed_dir / 'reasoning_traces.json'
        if not reasoning_file.exists():
            print(f"  WARNING: No reasoning_traces.json in {seed_dir.name}")
            continue

        with open(reasoning_file, 'r') as f:
            traces = json.load(f)

        # Load image paths for reference
        image_paths_file = seed_dir / 'image_paths.json'
        if image_paths_file.exists():
            with open(image_paths_file, 'r') as f:
                image_paths = json.load(f)
        else:
            image_paths = [f"image_{i}" for i in range(len(traces))]

        # Add metadata to each trace
        for i, trace in enumerate(traces):
            if trace:  # Skip empty traces
                image_name = Path(image_paths[i]).stem if i < len(image_paths) else f"image_{i}"
                all_traces.append({
                    'seed': seed,
                    'index': i,
                    'image': image_name,
                    'trace': trace
                })

    print(f"Total traces loaded: {len(all_traces)}")
    return all_traces


def sample_traces(all_traces, n_traces, random_seed=42):
    """Randomly sample n traces from all available traces."""
    random.seed(random_seed)

    if len(all_traces) <= n_traces:
        print(f"WARNING: Only {len(all_traces)} traces available, using all")
        return all_traces

    sampled = random.sample(all_traces, n_traces)

    # Sort by seed then index for some organization
    sampled.sort(key=lambda x: (x['seed'], x['index']))

    return sampled


def create_annotation_matrix(concepts_df, sampled_traces, output_path):
    """Create the annotation matrix CSV."""

    # Create column headers for traces
    # Format: seed_S_img_IMAGENAME (truncated for readability)
    trace_columns = []
    trace_texts = {}  # Store full trace text separately

    for i, trace_info in enumerate(sampled_traces):
        col_name = f"s{trace_info['seed']}_i{trace_info['index']}_{trace_info['image'][:20]}"
        trace_columns.append(col_name)
        trace_texts[col_name] = trace_info['trace']

    # Create the matrix
    # First columns: concept metadata (including definition for LLM judge)
    # Remaining columns: one per trace (empty for annotation)

    rows = []
    for _, concept_row in concepts_df.iterrows():
        concept_name = concept_row['concept']
        row = {
            'direction': concept_row['direction'],
            'concept': concept_name,
            'definition': concept_name,  # Default to concept name, edit for LLM judge
            'mean_dd': concept_row.get('mean_dd', ''),
            'p_value': concept_row.get('p_value', ''),
        }
        # Add empty cells for each trace
        for col in trace_columns:
            row[col] = ''
        rows.append(row)

    matrix_df = pd.DataFrame(rows)

    # Save the matrix
    matrix_df.to_csv(output_path, index=False)
    print(f"\nSaved annotation matrix: {output_path}")
    print(f"  Rows (concepts): {len(matrix_df)}")
    print(f"  Trace columns: {len(trace_columns)}")

    # Save trace texts separately for reference during annotation
    trace_ref_path = output_path.parent / (output_path.stem + '_trace_texts.json')
    with open(trace_ref_path, 'w') as f:
        json.dump(trace_texts, f, indent=2)
    print(f"  Trace texts saved: {trace_ref_path}")

    return matrix_df


def main():
    parser = argparse.ArgumentParser(description='Create annotation matrix for concept presence')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Results directory with analysis_outputs folder')
    parser.add_argument('--n_traces', type=int, default=100,
                        help='Number of reasoning traces to sample (default: 100)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: {results_dir}/annotation_matrix.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trace sampling (default: 42)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        return 1

    print(f"Processing: {results_dir}")
    print()

    # Load significant concepts
    print("Loading significant concepts...")
    concepts_df = load_significant_concepts(results_dir)

    if concepts_df.empty:
        print("ERROR: No significant concepts found")
        return 1

    print(f"Total significant concepts: {len(concepts_df)}")
    print()

    # Load all reasoning traces
    print("Loading reasoning traces...")
    all_traces = load_reasoning_traces(results_dir)

    if not all_traces:
        print("ERROR: No reasoning traces found")
        return 1

    print()

    # Sample traces
    print(f"Sampling {args.n_traces} traces (seed={args.seed})...")
    sampled_traces = sample_traces(all_traces, args.n_traces, args.seed)
    print(f"Sampled {len(sampled_traces)} traces")
    print()

    # Create output path - default to significant_only folder for easy annotation
    if args.output:
        output_path = Path(args.output)
    else:
        sig_dir = results_dir / 'analysis_outputs' / 'significant_only'
        sig_dir.mkdir(parents=True, exist_ok=True)

        # Extract model name from results directory (e.g., "R1-Onevision-7B" from "R1-Onevision-7B_DDI_...")
        model_name = results_dir.name.split('_DDI_')[0]
        output_path = sig_dir / f'annotation_matrix_{model_name}.csv'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the matrix
    create_annotation_matrix(concepts_df, sampled_traces, output_path)

    print()
    print("=" * 60)
    print("Instructions for annotation:")
    print("  1. Open the CSV in a spreadsheet editor")
    print("  2. For each concept row, check each trace column")
    print("  3. Mark 1 if the concept is present in the trace, 0 if absent")
    print("  4. Use the _trace_texts.json file to see full reasoning traces")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
