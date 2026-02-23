#!/usr/bin/env python
"""
Analyze LLM judge output for concept presence patterns.

This script analyzes the output from concept_presence_analysis.py:
1. Concept frequency analysis - which concepts are frequently/rarely mentioned
2. Concept co-occurrence analysis - which concepts appear together
3. Cross-seed consistency - how consistently concepts appear across seeds

Usage:
    python src/experiments/analyze_judge_output.py \
        --judge_json path/to/concept_presence_analysis.json \
        --output_dir analysis_outputs/judge_analysis
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


def load_judge_output(json_path):
    """Load and parse judge output JSON.

    Returns:
        concepts: list of concept dicts
        trace_analyses: list of trace analysis results
        metadata: dict with analysis metadata
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    concepts = data.get('concepts', data.get('top_concepts', []))
    trace_analyses = data.get('trace_analyses', [])
    metadata = data.get('metadata', {})

    return concepts, trace_analyses, metadata


def build_presence_matrix(concepts, trace_analyses):
    """Build binary presence matrix from judge output.

    Returns:
        presence_matrix: numpy array (n_concepts x n_traces)
        trace_info: list of dicts with trace metadata
    """
    concept_names = [c['concept'] for c in concepts]
    n_concepts = len(concept_names)
    n_traces = len(trace_analyses)

    presence_matrix = np.zeros((n_concepts, n_traces), dtype=int)
    trace_info = []

    for j, trace in enumerate(trace_analyses):
        # Get present concepts from reasoning analysis
        present = set(trace.get('reasoning_analysis', {}).get('concepts_present', []))

        for i, concept in enumerate(concept_names):
            if concept in present:
                presence_matrix[i, j] = 1

        # Store trace metadata
        trace_info.append({
            'trace_index': trace.get('trace_index', j),
            'seed': trace.get('seed', 0),
            'image_path': trace.get('image_path', ''),
            'label': trace.get('label', ''),
        })

    return presence_matrix, trace_info


def compute_frequency_stats(concepts, presence_matrix):
    """Compute concept frequency statistics."""
    n_traces = presence_matrix.shape[1]

    stats = []
    for i, concept in enumerate(concepts):
        count = presence_matrix[i, :].sum()
        rate = count / n_traces if n_traces > 0 else 0

        stats.append({
            'concept': concept['concept'],
            'direction': concept.get('direction', 'unknown'),
            'original_name': concept.get('original_name', ''),
            'all_original_names': ', '.join(concept.get('all_original_names', [])),
            'duplicate_count': concept.get('duplicate_count', 1),
            'presence_count': int(count),
            'presence_rate': round(rate, 3),
        })

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('presence_rate', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)

    return stats_df


def compute_cooccurrence(presence_matrix, concept_names):
    """Compute concept co-occurrence matrix using Jaccard similarity."""
    n_concepts = presence_matrix.shape[0]
    cooccurrence = np.zeros((n_concepts, n_concepts))

    for i in range(n_concepts):
        for j in range(n_concepts):
            a = presence_matrix[i, :]
            b = presence_matrix[j, :]

            intersection = np.sum(a & b)
            union = np.sum(a | b)

            if union > 0:
                cooccurrence[i, j] = intersection / union
            else:
                cooccurrence[i, j] = 0.0

    cooccurrence_df = pd.DataFrame(cooccurrence, index=concept_names, columns=concept_names)

    # Find pairs with high co-occurrence
    pairs = []
    for i in range(n_concepts):
        for j in range(i + 1, n_concepts):
            if cooccurrence[i, j] > 0.3:  # Lower threshold than annotation analysis
                pairs.append({
                    'concept_1': concept_names[i],
                    'concept_2': concept_names[j],
                    'jaccard': round(cooccurrence[i, j], 3)
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values('jaccard', ascending=False)

    return cooccurrence_df, pairs_df


def compute_cross_seed_consistency(presence_matrix, trace_info, concepts):
    """Analyze concept consistency across different seeds for the same image."""
    # Group traces by image
    image_to_traces = defaultdict(list)

    for j, info in enumerate(trace_info):
        # Extract image name from path
        img_path = info.get('image_path', '')
        if img_path:
            img_name = Path(img_path).stem
        else:
            img_name = f"trace_{info['trace_index']}"

        image_to_traces[img_name].append((j, info['seed']))

    # Find images with multiple traces across different seeds
    multi_seed_images = {}
    for img, trace_seed_pairs in image_to_traces.items():
        if len(trace_seed_pairs) > 1:
            seeds = set(seed for _, seed in trace_seed_pairs)
            if len(seeds) > 1:
                indices = [idx for idx, _ in trace_seed_pairs]
                multi_seed_images[img] = indices

    if not multi_seed_images:
        print("  No images with multiple seeds found for consistency analysis")
        return None

    print(f"  Found {len(multi_seed_images)} images with traces across multiple seeds")

    # Compute per-concept consistency
    concept_consistency = []
    for i, concept in enumerate(concepts):
        consistent_count = 0
        total_images = 0

        for img_name, indices in multi_seed_images.items():
            values = [presence_matrix[i, j] for j in indices]
            if len(set(values)) == 1:  # All same (all 0 or all 1)
                consistent_count += 1
            total_images += 1

        consistency_rate = consistent_count / total_images if total_images > 0 else 0

        concept_consistency.append({
            'concept': concept['concept'],
            'direction': concept.get('direction', 'unknown'),
            'consistency_rate': round(consistency_rate, 3),
            'consistent_images': consistent_count,
            'total_multi_seed_images': total_images
        })

    consistency_df = pd.DataFrame(concept_consistency)
    consistency_df = consistency_df.sort_values('consistency_rate', ascending=True)

    return consistency_df


def plot_frequency(stats_df, output_path):
    """Plot bar chart of concept presence rates."""
    fig, ax = plt.subplots(figsize=(max(10, len(stats_df) * 0.5), 6))

    colors = ['#2ecc71' if d == 'positive' else '#e74c3c'
              for d in stats_df['direction']]

    x = range(len(stats_df))
    ax.bar(x, stats_df['presence_rate'], color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Concept')
    ax.set_ylabel('Presence Rate (Judge)')
    ax.set_title('Concept Presence Frequency (LLM Judge)')
    ax.set_xticks(x)

    # Use original_name if available, else concept
    labels = []
    for _, row in stats_df.iterrows():
        name = row['original_name'] if row['original_name'] else row['concept']
        if len(name) > 15:
            name = name[:12] + '...'
        labels.append(name)

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Positive (malignant)'),
        Patch(facecolor='#e74c3c', label='Negative (benign)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cooccurrence_heatmap(cooccurrence_df, output_path):
    """Plot heatmap of concept co-occurrence."""
    fig, ax = plt.subplots(figsize=(max(10, len(cooccurrence_df) * 0.6),
                                     max(8, len(cooccurrence_df) * 0.5)))

    im = ax.imshow(cooccurrence_df.values, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(cooccurrence_df.columns)))
    ax.set_yticks(range(len(cooccurrence_df.index)))

    labels = [c[:12] + '...' if len(c) > 12 else c for c in cooccurrence_df.columns]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_title('Concept Co-occurrence (Jaccard Similarity)')
    plt.colorbar(im, ax=ax, label='Jaccard Similarity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_concept_clusters(cooccurrence_df, concepts, distance_threshold=0.6):
    """Print concept clusters as text."""
    if len(cooccurrence_df) < 2:
        print("  Need at least 2 concepts for clustering")
        return

    distance_matrix = 1 - cooccurrence_df.values
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    condensed = squareform(distance_matrix)
    linkage_matrix = linkage(condensed, method='average')
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

    # Build concept -> definition mapping
    concept_to_def = {}
    for c in concepts:
        name = c.get('original_name', '') or c['concept']
        concept_to_def[c['concept']] = name

    # Group concepts by cluster
    clusters = defaultdict(list)
    concept_names = cooccurrence_df.index.tolist()
    for concept, cluster_id in zip(concept_names, cluster_labels):
        clusters[cluster_id].append(concept)

    # Print clusters with more than one concept
    multi_concept_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    if not multi_concept_clusters:
        print("  No concept clusters found (all concepts appear independently)")
        return

    print(f"\nConcept Clusters (Jaccard threshold={1-distance_threshold:.1f}):")
    for cluster_id, concepts_in_cluster in sorted(multi_concept_clusters.items(),
                                                   key=lambda x: -len(x[1])):
        print(f"\n  Cluster {cluster_id} ({len(concepts_in_cluster)} concepts):")
        for concept in concepts_in_cluster:
            name = concept_to_def.get(concept, concept)
            if len(name) > 50:
                name = name[:47] + "..."
            print(f"    - {name}")


def main():
    parser = argparse.ArgumentParser(description='Analyze LLM judge output')
    parser.add_argument('--judge_json', type=str, required=True,
                        help='Path to concept_presence_analysis JSON output')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same dir as judge_json)')
    args = parser.parse_args()

    judge_json = Path(args.judge_json)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = judge_json.parent / 'judge_analysis'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LLM Judge Output Analysis")
    print("=" * 70)
    print(f"Input: {judge_json}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading judge output...")
    concepts, trace_analyses, metadata = load_judge_output(judge_json)
    print(f"  Concepts: {len(concepts)}")
    print(f"  Traces: {len(trace_analyses)}")
    print()

    # Build presence matrix
    print("Building presence matrix...")
    presence_matrix, trace_info = build_presence_matrix(concepts, trace_analyses)
    print(f"  Matrix shape: {presence_matrix.shape}")
    print(f"  Total concept mentions: {presence_matrix.sum()}")
    print()

    # =================================
    # 1. Frequency Analysis
    # =================================
    print("Computing frequency statistics...")
    freq_stats = compute_frequency_stats(concepts, presence_matrix)
    freq_stats.to_csv(output_dir / 'concept_frequency.csv', index=False)
    print(f"  Saved: concept_frequency.csv")

    plot_frequency(freq_stats, output_dir / 'concept_frequency.png')
    print()

    # Print top/bottom concepts
    print("Most frequently mentioned concepts:")
    for _, row in freq_stats.head(5).iterrows():
        name = row['original_name'] if row['original_name'] else row['concept']
        if len(name) > 40:
            name = name[:37] + "..."
        print(f"  {name}: {row['presence_rate']:.0%} ({row['presence_count']} times)")

    print("\nLeast frequently mentioned concepts:")
    for _, row in freq_stats.tail(5).iterrows():
        name = row['original_name'] if row['original_name'] else row['concept']
        if len(name) > 40:
            name = name[:37] + "..."
        print(f"  {name}: {row['presence_rate']:.0%} ({row['presence_count']} times)")
    print()

    # =================================
    # 2. Co-occurrence Analysis
    # =================================
    print("Computing co-occurrence matrix...")
    concept_names = [c['concept'] for c in concepts]
    cooccurrence_df, pairs_df = compute_cooccurrence(presence_matrix, concept_names)

    cooccurrence_df.to_csv(output_dir / 'concept_cooccurrence.csv')
    print(f"  Saved: concept_cooccurrence.csv")

    if len(pairs_df) > 0:
        pairs_df.to_csv(output_dir / 'concept_pairs.csv', index=False)
        print(f"  Saved: concept_pairs.csv ({len(pairs_df)} pairs with Jaccard > 0.3)")

        print("\nTop co-occurring concept pairs:")
        for _, row in pairs_df.head(10).iterrows():
            print(f"  {row['jaccard']:.2f}: {row['concept_1'][:25]} + {row['concept_2'][:25]}")
    else:
        print("  No concept pairs with Jaccard > 0.3 found")
    print()

    plot_cooccurrence_heatmap(cooccurrence_df, output_dir / 'concept_cooccurrence.png')

    # Print clusters as text
    print_concept_clusters(cooccurrence_df, concepts)
    print()

    # =================================
    # 3. Cross-Seed Consistency
    # =================================
    print("Computing cross-seed consistency...")
    consistency_df = compute_cross_seed_consistency(presence_matrix, trace_info, concepts)

    if consistency_df is not None:
        consistency_df.to_csv(output_dir / 'cross_seed_consistency.csv', index=False)
        print(f"  Saved: cross_seed_consistency.csv")

        mean_consistency = consistency_df['consistency_rate'].mean()
        print(f"\n  Mean concept consistency across seeds: {mean_consistency:.0%}")

        low_consistency = consistency_df[consistency_df['consistency_rate'] < 0.5]
        if len(low_consistency) > 0:
            print(f"\n  Concepts with low cross-seed consistency (<50%):")
            for _, row in low_consistency.iterrows():
                print(f"    {row['concept'][:40]}: {row['consistency_rate']:.0%}")
    print()

    # =================================
    # 4. Summary Statistics
    # =================================
    # Seed distribution
    seed_counts = Counter(t['seed'] for t in trace_info)
    label_counts = Counter(t['label'] for t in trace_info)

    summary = {
        'num_traces': len(trace_analyses),
        'num_concepts': len(concepts),
        'total_concept_mentions': int(presence_matrix.sum()),
        'avg_concepts_per_trace': round(presence_matrix.sum(axis=0).mean(), 2),
        'seed_distribution': dict(sorted(seed_counts.items())),
        'label_distribution': dict(label_counts),
    }

    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: summary_stats.json")

    print()
    print("=" * 70)
    print(f"Analysis complete. Outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
