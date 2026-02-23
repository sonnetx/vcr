"""
Create inspection JSON from existing bootstrap results.
Run this to prepare input for concept_presence_analysis.py without re-running VCR.

By default, loads concepts and definitions from the human annotation CSV in
  <annotations_dir>/annotations_positive_<model>.csv
Column headers encode concept name + definition via <MEANING> tags.

Falls back to per-seed sensitivity ranking if no annotation CSV is found.

Usage:
    python src/experiments/create_inspection_json.py \
        --results_dir Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_simple_binary \
        --annotations_dir human_annotations \
        --seed 0 \
        --output inspection_input.json
"""

import json
import random
import re
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_traces_from_all_seeds(results_dir, config, max_traces=0, random_seed=42):
    """Load reasoning traces from all seed directories.

    Args:
        results_dir: Path to results directory
        config: Experiment config dict
        max_traces: Maximum traces to sample (0 = all)
        random_seed: Random seed for sampling

    Returns:
        List of trace dicts with seed, index, reasoning, image_path, label
    """
    results_dir = Path(results_dir)

    # Find all seed directories
    seed_dirs = sorted([d for d in results_dir.iterdir()
                        if d.is_dir() and re.match(r'^seed_\d+$', d.name)])

    if not seed_dirs:
        raise ValueError(f"No seed directories found in {results_dir}")

    print(f"  Found {len(seed_dirs)} seed directories")

    # Load metadata for labels
    df = pd.read_csv(config['metadata_path'], index_col=0)
    df['label'] = df['malignant'].map({False: 'Benign', True: 'Malignant'})

    all_traces = []

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split('_')[1])

        # Load reasoning traces
        traces_path = seed_dir / 'reasoning_traces.json'
        if not traces_path.exists():
            print(f"  WARNING: No reasoning_traces.json in {seed_dir.name}")
            continue

        with open(traces_path, 'r') as f:
            traces = json.load(f)

        # Load image paths if available
        image_paths_file = seed_dir / 'image_paths.json'
        if image_paths_file.exists():
            with open(image_paths_file, 'r') as f:
                image_paths = json.load(f)
        else:
            # Fall back to computing from train split
            train_df, _ = train_test_split(
                df, test_size=config['test_size'], random_state=seed
            )
            if config.get('demo_size', 0) > 0 and config.get('prompt', {}).get('use_demos', False):
                train_df, _ = train_test_split(
                    train_df, test_size=config['demo_size'], random_state=seed
                )
            image_paths = [f"{config['ddi_base_dir']}/{path}" for path in train_df['DDI_file']]

        # Get labels for this seed's split
        train_df, _ = train_test_split(
            df, test_size=config['test_size'], random_state=seed
        )
        if config.get('demo_size', 0) > 0 and config.get('prompt', {}).get('use_demos', False):
            train_df, _ = train_test_split(
                train_df, test_size=config['demo_size'], random_state=seed
            )
        train_labels = train_df['label'].values

        # Add traces with metadata
        for idx, trace in enumerate(traces):
            if trace:  # Skip empty traces
                image_path = image_paths[idx] if idx < len(image_paths) else ""
                label = train_labels[idx] if idx < len(train_labels) else "Unknown"

                all_traces.append({
                    'seed': seed,
                    'index': idx,
                    'image_path': image_path,
                    'label': label,
                    'reasoning': trace,
                    'answer': '',
                })

    print(f"  Total traces loaded: {len(all_traces)}")

    # Sample if requested
    if max_traces > 0 and len(all_traces) > max_traces:
        random.seed(random_seed)
        all_traces = random.sample(all_traces, max_traces)
        print(f"  Sampled {max_traces} traces")

    return all_traces



def find_annotation_csv(results_dir, annotations_dir):
    """Find the human annotation CSV for a given results directory.

    Searches annotations_dir for annotations_positive_*.csv matching the model name.
    """
    annotations_dir = Path(annotations_dir)
    dir_name = Path(results_dir).name

    # Extract model identifier (part before _DDI_)
    model_id = dir_name.split('_DDI_')[0] if '_DDI_' in dir_name else dir_name

    all_csvs = sorted(annotations_dir.glob('annotations_positive_*.csv'))
    if not all_csvs:
        return None

    for csv_path in all_csvs:
        stem = csv_path.stem
        if model_id in stem:
            return csv_path
        csv_model_part = stem.replace('annotations_positive_', '')
        if csv_model_part in dir_name or dir_name in csv_model_part:
            return csv_path

    return None


def load_concepts_from_annotation_csv(results_dir, annotations_dir):
    """Load concepts and definitions from the human annotation CSV.

    Expects CSV with 'concept' and 'definition' columns.
    Returns list of concept dicts, or None if no annotation CSV found.
    """
    csv_path = find_annotation_csv(results_dir, annotations_dir)
    if csv_path is None:
        return None

    print(f"  Found annotation CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if 'concept' not in df.columns or 'definition' not in df.columns:
        print(f"  WARNING: annotation CSV missing 'concept' or 'definition' columns")
        return None

    concepts = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        concept_name = row['concept']
        definition = row['definition']
        if pd.isna(concept_name):
            continue
        concepts.append({
            'rank': rank,
            'concept': definition if pd.notna(definition) else concept_name,
            'original_name': concept_name,
            'direction': row.get('direction', 'positive'),
        })

    if not concepts:
        return None

    print(f"  Loaded {len(concepts)} concepts with definitions from annotation CSV")
    return concepts


def load_concepts_from_sensitivity(seed_dir, top_k):
    """Fallback: load concepts from per-seed sensitivity (old behavior)."""
    weighted_sens = np.load(seed_dir / 'weighted_sens.npy')
    raw_sens = np.load(seed_dir / 'raw_sens.npy')

    with open(seed_dir / 'concept_texts.json', 'r') as f:
        concept_texts = json.load(f)

    avg_weighted_sens = np.mean(weighted_sens, axis=0)
    avg_raw_sens = np.mean(raw_sens, axis=0)

    top_indices = np.argsort(np.abs(avg_weighted_sens))[-top_k:][::-1]

    concepts = [
        {
            'rank': rank,
            'concept': concept_texts[int(idx)],
            'weighted_sensitivity': float(avg_weighted_sens[idx]),
            'raw_sensitivity': float(avg_raw_sens[idx]),
            'direction': 'positive' if avg_weighted_sens[idx] >= 0 else 'negative',
        }
        for rank, idx in enumerate(top_indices, 1)
    ]
    return concepts


def main():
    parser = argparse.ArgumentParser(
        description='Create inspection JSON from existing bootstrap results for concept_presence_analysis.py'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory with bootstrap experiment results')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed folder for reasoning traces (default: 0)')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top concepts if falling back to per-seed ranking (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: <results_dir>/inspection.json)')
    parser.add_argument('--direction', type=str, default=None,
                        choices=['positive', 'negative'],
                        help='Only include concepts of this direction (default: both)')
    parser.add_argument('--concepts_csv', type=str, default=None,
                        help='Optional: explicit path to a concepts CSV (overrides auto-detect)')
    parser.add_argument('--max_traces', type=int, default=0,
                        help='Maximum number of traces to include (0 = all, default: 0)')
    parser.add_argument('--annotations_dir', type=str, default='human_annotations',
                        help='Directory containing human annotation CSVs (default: human_annotations)')
    parser.add_argument('--all_seeds', action='store_true',
                        help='Load traces from ALL seed directories instead of just one')
    parser.add_argument('--sample_seed', type=int, default=42,
                        help='Random seed for sampling traces when --all_seeds is used (default: 42)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seed_dir = results_dir / f'seed_{args.seed}'

    if args.output is None:
        suffix = f'_{args.direction}' if args.direction else ''
        args.output = str(results_dir / f'inspection{suffix}.json')

    print("=" * 60)
    print("Creating Inspection JSON from Bootstrap Results")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    if args.all_seeds:
        print(f"Traces: ALL seeds (sample_seed={args.sample_seed})")
    else:
        print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print()

    # --- Concept selection ---
    top_concepts = None
    concept_source = None

    # 1) Explicit CSV override
    if args.concepts_csv:
        print(f"Using explicit concepts CSV: {args.concepts_csv}")
        df = pd.read_csv(args.concepts_csv)
        top_concepts = []
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            top_concepts.append({
                'rank': rank,
                'concept': row['concept'],
                'mean_dd': float(row['mean_dd']),
                'p_value': float(row['p_value']),
                'direction': 'positive' if float(row['mean_dd']) >= 0 else 'negative',
            })
        concept_source = f'csv:{args.concepts_csv}'

    # 2) Auto-detect from human annotation CSV
    if top_concepts is None:
        print(f"Checking for human annotation CSV in {args.annotations_dir}...")
        top_concepts = load_concepts_from_annotation_csv(results_dir, args.annotations_dir)
        if top_concepts is not None:
            concept_source = 'human_annotation_csv'
        else:
            print("  No annotation CSV found")

    # 3) Fallback to per-seed sensitivity
    if top_concepts is None:
        print(f"  Falling back to per-seed sensitivity ranking (top_k={args.top_k})")
        top_concepts = load_concepts_from_sensitivity(seed_dir, args.top_k)
        concept_source = 'per_seed_sensitivity'

    # Filter by direction if requested
    if args.direction:
        top_concepts = [c for c in top_concepts if c.get('direction') == args.direction]
        print(f"Filtered to {args.direction} concepts only")

    # Deduplicate concepts with identical definitions
    from experiment_utils import deduplicate_concepts
    top_concepts = deduplicate_concepts(top_concepts)

    print(f"\nConcept source: {concept_source}")
    print(f"Total concepts: {len(top_concepts)}")
    pos_count = sum(1 for c in top_concepts if c.get('direction') == 'positive')
    neg_count = sum(1 for c in top_concepts if c.get('direction') == 'negative')
    print(f"  Positive: {pos_count}, Negative: {neg_count}")
    print()

    for c in top_concepts[:5]:
        direction_char = c['direction'][0].upper()
        rank = c['rank']
        # Show code word + definition if we have both
        if 'original_name' in c:
            code_word = c['original_name']
            definition = c['concept']
            print(f"  [{direction_char}] {rank:2d}. {code_word}: {definition}")
        else:
            print(f"  [{direction_char}] {rank:2d}. {c['concept']}")
    if len(top_concepts) > 5:
        print("  ...")

    # Load experiment config
    print("\nLoading experiment config...")
    with open(results_dir / 'experiment_config.json', 'r') as f:
        exp_config = json.load(f)
    config = exp_config['config']

    # --- Load reasoning traces ---
    if args.all_seeds:
        print("\nLoading reasoning traces from ALL seeds...")
        reasoning_traces = load_traces_from_all_seeds(
            results_dir, config, args.max_traces, args.sample_seed
        )
    else:
        print(f"\nLoading reasoning traces from seed_{args.seed}...")
        with open(seed_dir / 'reasoning_traces.json', 'r') as f:
            reasoning_traces_raw = json.load(f)
        print(f"  Loaded {len(reasoning_traces_raw)} traces")

        # Load metadata for image paths and labels
        print("Loading DDI metadata for image paths...")
        df = pd.read_csv(config['metadata_path'], index_col=0)
        df['label'] = df['malignant'].map({False: 'Benign', True: 'Malignant'})

        train_df, _ = train_test_split(
            df, test_size=config['test_size'], random_state=args.seed
        )
        if config.get('demo_size', 0) > 0 and config.get('prompt', {}).get('use_demos', False):
            train_df, _ = train_test_split(
                train_df, test_size=config['demo_size'], random_state=args.seed
            )

        train_paths = [f"{config['ddi_base_dir']}/{path}" for path in train_df['DDI_file']]
        train_labels = train_df['label'].values
        print(f"  Found {len(train_paths)} training images")

        # Build reasoning traces with metadata
        reasoning_traces = []
        num_traces = len(reasoning_traces_raw)
        if args.max_traces > 0:
            num_traces = min(num_traces, args.max_traces)

        for idx in range(num_traces):
            reasoning = reasoning_traces_raw[idx] if idx < len(reasoning_traces_raw) else ""
            image_path = train_paths[idx] if idx < len(train_paths) else ""
            label = train_labels[idx] if idx < len(train_labels) else "Unknown"

            reasoning_traces.append({
                'seed': args.seed,
                'index': idx,
                'image_path': image_path,
                'label': label,
                'reasoning': reasoning,
                'answer': '',
            })

    # Build output
    output_data = {
        'model': exp_config.get('model', config.get('model_name', 'Unknown')),
        'prompt_name': exp_config.get('prompt_name', 'unknown'),
        'task_definition': config.get('task_definition', 'malignant_prob'),
        'all_seeds': args.all_seeds,
        'random_seed': args.seed if not args.all_seeds else None,
        'sample_seed': args.sample_seed if args.all_seeds else None,
        'source_results_dir': str(results_dir),
        'concept_source': concept_source,
        'top_concepts': top_concepts,
        'reasoning_traces': reasoning_traces,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print()
    print("=" * 60)
    print(f"Created: {args.output}")
    print(f"  - {len(top_concepts)} concepts ({concept_source})")
    print(f"  - {len(reasoning_traces)} reasoning traces")
    if args.all_seeds:
        # Show seed distribution
        from collections import Counter
        seed_counts = Counter(t['seed'] for t in reasoning_traces)
        print(f"  - Seeds: {dict(sorted(seed_counts.items()))}")
    print("=" * 60)
    print()
    print("Next step:")
    print(f"  python src/experiments/concept_presence_analysis.py \\")
    print(f"      --input_json {args.output} \\")
    print(f"      --output_dir results/concept_presence_analysis")


if __name__ == '__main__':
    main()
