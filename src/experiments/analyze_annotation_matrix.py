#!/usr/bin/env python
"""
Analyze manual concept presence annotations from annotation matrix CSVs.

This script analyzes hand-labeled concept presence in reasoning traces:
1. Concept frequency analysis - which concepts are frequently/rarely mentioned
2. Concept co-occurrence analysis - which concepts appear together (clusters)
3. Accuracy correlation - how concept usage relates to prediction accuracy

Input format (annotation_matrix_{model}.csv):
- Rows: concepts (with columns: direction, concept, definition, mean_dd, p_value)
- Trace columns: s{seed}_i{index}_{image_name} with 1=present, empty=absent

Usage:
    python src/experiments/analyze_annotation_matrix.py \
        --annotation_csv path/to/annotation_matrix_Model.csv \
        --results_dir path/to/results_dir \
        --metadata_csv /path/to/ddi_metadata.csv
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


# =============================================================================
# Data Loading
# =============================================================================

def parse_trace_column(col_name):
    """Parse trace column name to extract seed, index, and image name.

    Format: s{seed}_i{index}_{image_name}
    Example: s0_i42_ISIC_0024306 -> (0, 42, 'ISIC_0024306')

    Returns:
        tuple: (seed, index, image_name) or None if parsing fails
    """
    match = re.match(r'^s(\d+)_i(\d+)_(.+)$', col_name)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None


def load_annotation_matrix(csv_path):
    """Load annotation matrix CSV and parse structure.

    Returns:
        concepts_df: DataFrame with concept metadata (direction, concept, definition, mean_dd, p_value)
        presence_matrix: numpy array (n_concepts x n_traces) with 1/0 values
        trace_info: list of dicts with seed, index, image_name per trace column
        trace_columns: list of trace column names
    """
    df = pd.read_csv(csv_path)

    # Identify metadata columns - use known list
    metadata_cols = ['direction', 'concept', 'definition', 'mean_dd', 'p_value']

    # More robust: identify trace columns by regex pattern, not exclusion
    # This handles cases where extra columns might be added to the CSV
    trace_info = []
    valid_trace_cols = []
    for col in df.columns:
        parsed = parse_trace_column(col)
        if parsed:
            seed, idx, image = parsed
            trace_info.append({
                'column': col,
                'seed': seed,
                'index': idx,
                'image_name': image
            })
            valid_trace_cols.append(col)

    # Warn about any unexpected columns
    expected_cols = set(metadata_cols + valid_trace_cols)
    unexpected = [c for c in df.columns if c not in expected_cols]
    if unexpected:
        print(f"  Note: Ignoring unexpected columns: {unexpected}")

    # Extract concept metadata and reset index for consistent row access
    concepts_df = df[metadata_cols].copy().reset_index(drop=True)

    # Extract presence matrix - ensure binary 0/1 values
    presence_data = df[valid_trace_cols].copy()
    # Convert: empty/NaN -> 0, any non-zero value -> 1
    presence_matrix = (presence_data.fillna(0).values != 0).astype(int)

    # Validate presence matrix values
    max_val = presence_matrix.max() if presence_matrix.size > 0 else 0
    min_val = presence_matrix.min() if presence_matrix.size > 0 else 0
    if max_val > 1 or min_val < 0:
        print(f"  WARNING: Presence matrix had unexpected values (min={min_val}, max={max_val}), clipped to 0/1")
        presence_matrix = np.clip(presence_matrix, 0, 1)

    # Filter to only columns that have at least one annotation
    # (columns with all zeros are unannotated)
    col_sums = presence_matrix.sum(axis=0)
    annotated_mask = col_sums > 0
    n_annotated = annotated_mask.sum()
    n_total = len(valid_trace_cols)

    if n_annotated < n_total:
        print(f"  Filtering to {n_annotated}/{n_total} annotated traces (removing {n_total - n_annotated} unannotated)")
        presence_matrix = presence_matrix[:, annotated_mask]
        trace_info = [t for i, t in enumerate(trace_info) if annotated_mask[i]]
        valid_trace_cols = [c for i, c in enumerate(valid_trace_cols) if annotated_mask[i]]

    print(f"  Loaded {len(concepts_df)} concepts x {len(valid_trace_cols)} traces")
    print(f"  Total annotations: {presence_matrix.sum()}")

    return concepts_df, presence_matrix, trace_info, valid_trace_cols


def load_trace_texts(trace_texts_path):
    """Load reasoning trace texts from JSON file."""
    if not Path(trace_texts_path).exists():
        print(f"  Warning: Trace texts file not found: {trace_texts_path}")
        return {}

    with open(trace_texts_path, 'r') as f:
        return json.load(f)


def load_ground_truth_labels(metadata_csv, image_names):
    """Load ground truth labels from DDI metadata CSV.

    Args:
        metadata_csv: Path to ddi_metadata.csv
        image_names: List of image names to look up

    Returns:
        dict: image_name -> label ('malignant' or 'benign')
        Also returns skin_tone dict: image_name -> skin_tone (12 or 56)
    """
    if not Path(metadata_csv).exists():
        print(f"  Warning: Metadata CSV not found: {metadata_csv}")
        return {}, {}

    df = pd.read_csv(metadata_csv, index_col=0)

    # Debug: show what we're matching
    print(f"  Matching {len(image_names)} image names against {len(df)} metadata rows")
    if image_names:
        print(f"  Sample image names: {image_names[:3]}")
    if 'DDI_file' in df.columns:
        print(f"  Sample DDI_file values: {df['DDI_file'].head(3).tolist()}")

    labels = {}
    skin_tones = {}
    unmatched = []
    for img_name in image_names:
        # Try to match by DDI_file containing the image name (use regex=False for exact substring match)
        matches = df[df['DDI_file'].str.contains(img_name, na=False, regex=False)]
        if len(matches) > 0:
            is_malignant = matches.iloc[0]['malignant']
            labels[img_name] = 'malignant' if is_malignant else 'benign'
            # Also get skin tone for FST analysis
            if 'skin_tone' in df.columns:
                skin_tones[img_name] = matches.iloc[0]['skin_tone']
        else:
            unmatched.append(img_name)

    print(f"  Loaded labels for {len(labels)}/{len(image_names)} images")
    if skin_tones:
        fst12_count = sum(1 for v in skin_tones.values() if v == 12)
        fst56_count = sum(1 for v in skin_tones.values() if v == 56)
        print(f"  Skin tone distribution: FST12={fst12_count}, FST56={fst56_count}")

    # Always show unmatched if any
    if unmatched:
        print(f"  WARNING: {len(unmatched)} images could not be matched:")
        for u in unmatched:  # Show ALL unmatched
            print(f"    - '{u}'")

    return labels, skin_tones


def load_predictions_for_traces(results_dir, trace_info):
    """Load predictions for each trace from appropriate seed folders.

    Args:
        results_dir: Path to results directory containing seed_* folders
        trace_info: List of dicts with seed, index, image_name

    Returns:
        dict: column_name -> prediction ('malignant' or 'benign')
    """
    results_dir = Path(results_dir)
    predictions = {}

    # Group traces by seed to minimize file loading
    by_seed = {}
    for info in trace_info:
        seed = info['seed']
        if seed not in by_seed:
            by_seed[seed] = []
        by_seed[seed].append(info)

    for seed, infos in by_seed.items():
        pred_path = results_dir / f'seed_{seed}' / 'choice_differences.npy'
        if not pred_path.exists():
            print(f"  Warning: Predictions not found: {pred_path}")
            continue

        choice_diffs = np.load(pred_path)

        for info in infos:
            idx = info['index']
            if idx < len(choice_diffs):
                # choice_diff >= 0 -> malignant, < 0 -> benign
                pred = 'malignant' if choice_diffs[idx] >= 0 else 'benign'
                predictions[info['column']] = pred

    print(f"  Loaded predictions for {len(predictions)}/{len(trace_info)} traces")
    return predictions


# =============================================================================
# Frequency Analysis
# =============================================================================

def compute_frequency_stats(concepts_df, presence_matrix, trace_columns):
    """Compute concept frequency statistics.

    Returns:
        DataFrame with columns: concept, direction, definition, presence_count,
        presence_rate, rank
    """
    n_traces = presence_matrix.shape[1]

    stats = []
    for row_idx, (_, row) in enumerate(concepts_df.iterrows()):
        count = presence_matrix[row_idx, :].sum()
        rate = count / n_traces if n_traces > 0 else 0

        stats.append({
            'concept': row['concept'],
            'direction': row['direction'],
            'definition': row['definition'],
            'mean_dd': row['mean_dd'],
            'p_value': row['p_value'],
            'presence_count': int(count),
            'presence_rate': round(rate, 3),
        })

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('presence_rate', ascending=False)
    stats_df['rank'] = range(1, len(stats_df) + 1)

    return stats_df


def plot_frequency(stats_df, output_path, title="Concept Presence Frequency"):
    """Plot bar chart of concept presence rates."""
    fig, ax = plt.subplots(figsize=(max(10, len(stats_df) * 0.5), 6))

    # Color by direction
    colors = ['#2ecc71' if d == 'positive' else '#e74c3c'
              for d in stats_df['direction']]

    x = range(len(stats_df))
    ax.bar(x, stats_df['presence_rate'], color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Concept')
    ax.set_ylabel('Presence Rate')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['concept'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)

    # Add legend
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


# =============================================================================
# Co-occurrence Analysis
# =============================================================================

def compute_cooccurrence(presence_matrix, concept_names):
    """Compute concept co-occurrence matrix using Jaccard similarity.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns:
        cooccurrence_df: DataFrame with concept x concept Jaccard similarities
        pairs_df: DataFrame of concept pairs with Jaccard > 0.5
    """
    n_concepts = presence_matrix.shape[0]
    cooccurrence = np.zeros((n_concepts, n_concepts))

    for i in range(n_concepts):
        for j in range(n_concepts):
            a = presence_matrix[i, :]
            b = presence_matrix[j, :]

            intersection = np.sum((a == 1) & (b == 1))
            union = np.sum((a == 1) | (b == 1))

            if union > 0:
                cooccurrence[i, j] = intersection / union
            else:
                cooccurrence[i, j] = 0 if i != j else 1

    cooccurrence_df = pd.DataFrame(
        cooccurrence,
        index=concept_names,
        columns=concept_names
    )

    # Extract pairs with Jaccard > 0.5
    pairs = []
    for i in range(n_concepts):
        for j in range(i + 1, n_concepts):
            jaccard = cooccurrence[i, j]
            if jaccard > 0.5:
                pairs.append({
                    'concept_1': concept_names[i],
                    'concept_2': concept_names[j],
                    'jaccard': round(jaccard, 3),
                    'co_count': int(np.sum((presence_matrix[i, :] == 1) &
                                           (presence_matrix[j, :] == 1)))
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values('jaccard', ascending=False)

    return cooccurrence_df, pairs_df


def plot_cooccurrence_heatmap(cooccurrence_df, output_path):
    """Plot co-occurrence heatmap."""
    fig, ax = plt.subplots(figsize=(max(10, len(cooccurrence_df) * 0.6),
                                     max(8, len(cooccurrence_df) * 0.5)))

    im = ax.imshow(cooccurrence_df.values, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(cooccurrence_df.columns)))
    ax.set_yticks(range(len(cooccurrence_df.index)))
    ax.set_xticklabels(cooccurrence_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(cooccurrence_df.index)
    ax.set_title('Concept Co-occurrence (Jaccard Similarity)')

    plt.colorbar(im, ax=ax, label='Jaccard Similarity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_concept_clusters(cooccurrence_df, output_path):
    """Plot dendrogram of concept clusters."""
    if len(cooccurrence_df) < 2:
        print("  Skipping dendrogram: need at least 2 concepts")
        return

    # Convert similarity to distance
    distance_matrix = 1 - cooccurrence_df.values
    np.fill_diagonal(distance_matrix, 0)

    # Ensure symmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Condensed distance matrix for linkage
    condensed = squareform(distance_matrix)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed, method='average')

    fig, ax = plt.subplots(figsize=(max(10, len(cooccurrence_df) * 0.4), 6))

    dendrogram(
        linkage_matrix,
        labels=cooccurrence_df.index.tolist(),
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax
    )

    ax.set_title('Concept Clusters (Hierarchical Clustering)')
    ax.set_ylabel('Distance (1 - Jaccard)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_concept_clusters(cooccurrence_df, concepts_df, distance_threshold=0.5):
    """Print concept clusters as text.

    Groups concepts that co-occur frequently (Jaccard > 1 - distance_threshold).

    Args:
        cooccurrence_df: DataFrame with concept x concept Jaccard similarities
        concepts_df: DataFrame with concept metadata including definitions
        distance_threshold: Distance threshold for clustering (lower = tighter clusters)
    """
    if len(cooccurrence_df) < 2:
        print("  Need at least 2 concepts for clustering")
        return

    # Convert similarity to distance
    distance_matrix = 1 - cooccurrence_df.values
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Condensed distance matrix for linkage
    condensed = squareform(distance_matrix)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed, method='average')

    # Get cluster assignments
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

    # Build concept -> definition mapping
    concept_to_def = dict(zip(concepts_df['concept'], concepts_df['definition']))

    # Group concepts by cluster
    from collections import defaultdict
    clusters = defaultdict(list)
    concept_names = cooccurrence_df.index.tolist()
    for concept, cluster_id in zip(concept_names, cluster_labels):
        clusters[cluster_id].append(concept)

    # Print clusters with more than one concept (actual co-occurrence)
    multi_concept_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    if not multi_concept_clusters:
        print("  No concept clusters found (all concepts appear independently)")
        return

    print(f"\nConcept Clusters (concepts that frequently co-occur, Jaccard threshold={1-distance_threshold:.1f}):")
    for cluster_id, concepts in sorted(multi_concept_clusters.items(), key=lambda x: -len(x[1])):
        print(f"\n  Cluster {cluster_id} ({len(concepts)} concepts):")
        for concept in concepts:
            definition = concept_to_def.get(concept, '')
            # Truncate long definitions
            if len(definition) > 60:
                definition = definition[:57] + "..."
            print(f"    - {concept}: {definition}")


# =============================================================================
# Accuracy Analysis
# =============================================================================

def compute_accuracy_metrics(concepts_df, presence_matrix, trace_info,
                             predictions, labels):
    """Compute accuracy metrics per concept.

    Returns:
        accuracy_df: DataFrame with per-concept accuracy metrics
        overall_stats: dict with overall statistics
    """
    trace_columns = [t['column'] for t in trace_info]
    n_traces = len(trace_columns)

    # Compute correctness per trace
    correct = []
    for info in trace_info:
        col = info['column']
        img = info['image_name']

        pred = predictions.get(col)
        label = labels.get(img)

        if pred and label:
            correct.append(1 if pred == label else 0)
        else:
            correct.append(np.nan)

    correct = np.array(correct)
    valid_mask = ~np.isnan(correct)

    if valid_mask.sum() == 0:
        print("  Warning: No valid predictions/labels to compute accuracy")
        return None, None

    # Per-concept accuracy metrics
    accuracy_stats = []
    for row_idx, (_, row) in enumerate(concepts_df.iterrows()):
        concept_present = presence_matrix[row_idx, :] == 1
        concept_absent = presence_matrix[row_idx, :] == 0

        # Accuracy when concept is present
        present_and_valid = concept_present & valid_mask
        if present_and_valid.sum() > 0:
            acc_present = correct[present_and_valid].mean()
            n_present = present_and_valid.sum()
        else:
            acc_present = np.nan
            n_present = 0

        # Accuracy when concept is absent
        absent_and_valid = concept_absent & valid_mask
        if absent_and_valid.sum() > 0:
            acc_absent = correct[absent_and_valid].mean()
            n_absent = absent_and_valid.sum()
        else:
            acc_absent = np.nan
            n_absent = 0

        accuracy_stats.append({
            'concept': row['concept'],
            'direction': row['direction'],
            'n_present': int(n_present),
            'accuracy_when_present': round(acc_present, 3) if not np.isnan(acc_present) else None,
            'n_absent': int(n_absent),
            'accuracy_when_absent': round(acc_absent, 3) if not np.isnan(acc_absent) else None,
            'accuracy_diff': round(acc_present - acc_absent, 3) if not (np.isnan(acc_present) or np.isnan(acc_absent)) else None
        })

    accuracy_df = pd.DataFrame(accuracy_stats)

    # Overall statistics
    concept_counts = presence_matrix.sum(axis=0)  # concepts per trace
    overall_accuracy = correct[valid_mask].mean()

    # Correlation between concept count and accuracy
    valid_counts = concept_counts[valid_mask]
    valid_correct = correct[valid_mask]

    if len(valid_counts) > 1:
        correlation = np.corrcoef(valid_counts, valid_correct)[0, 1]
    else:
        correlation = np.nan

    overall_stats = {
        'n_traces': int(n_traces),
        'n_valid': int(valid_mask.sum()),
        'overall_accuracy': round(overall_accuracy, 3),
        'mean_concepts_per_trace': round(concept_counts.mean(), 2),
        'std_concepts_per_trace': round(concept_counts.std(), 2),
        'concept_count_accuracy_correlation': round(correlation, 3) if not np.isnan(correlation) else None
    }

    return accuracy_df, overall_stats


def plot_accuracy_vs_count(presence_matrix, trace_info, predictions, labels, output_path):
    """Plot scatter of concept count vs accuracy."""
    trace_columns = [t['column'] for t in trace_info]

    # Compute correctness per trace
    correct = []
    for info in trace_info:
        col = info['column']
        img = info['image_name']

        pred = predictions.get(col)
        label = labels.get(img)

        if pred and label:
            correct.append(1 if pred == label else 0)
        else:
            correct.append(np.nan)

    correct = np.array(correct)
    concept_counts = presence_matrix.sum(axis=0)

    valid_mask = ~np.isnan(correct)
    if valid_mask.sum() == 0:
        print("  Skipping accuracy plot: no valid data")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Jittered scatter
    jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=valid_mask.sum())
    colors = ['#2ecc71' if c == 1 else '#e74c3c' for c in correct[valid_mask]]

    ax.scatter(
        concept_counts[valid_mask] + jitter,
        correct[valid_mask] + jitter * 0.2,
        c=colors, s=80, alpha=0.7, edgecolors='black'
    )

    # Mean accuracy at each concept count
    unique_counts = sorted(set(concept_counts[valid_mask]))
    mean_acc = []
    for c in unique_counts:
        mask = (concept_counts == c) & valid_mask
        if mask.sum() > 0:
            mean_acc.append(correct[mask].mean())
        else:
            mean_acc.append(np.nan)

    ax.plot(unique_counts, mean_acc, 'b-o', linewidth=2, markersize=8,
            label='Mean accuracy', zorder=5)

    ax.set_xlabel('Number of VCR Concepts in Reasoning')
    ax.set_ylabel('Prediction Correct (0/1)')
    ax.set_title('Concept Count vs Prediction Accuracy')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Incorrect', 'Correct'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FST-Based Concept Analysis (Fitzpatrick Skin Type Fairness)
# =============================================================================

def compute_fst_concept_stats(presence_matrix, trace_info, skin_tones, concepts_df):
    """Analyze concept presence differences by Fitzpatrick skin type.

    Args:
        presence_matrix: numpy array (n_concepts x n_traces)
        trace_info: list of trace info dicts with image_name
        skin_tones: dict mapping image_name -> skin_tone (12 or 56)
        concepts_df: DataFrame with concept metadata

    Returns:
        fst_stats_df: DataFrame with per-concept FST statistics
    """
    if not skin_tones:
        print("  No skin tone data available for FST analysis")
        return None

    n_concepts, n_traces = presence_matrix.shape

    # Build masks for FST groups
    fst_12_mask = np.zeros(n_traces, dtype=bool)
    fst_56_mask = np.zeros(n_traces, dtype=bool)

    for j, info in enumerate(trace_info):
        img_name = info['image_name']
        if img_name in skin_tones:
            if skin_tones[img_name] == 12:
                fst_12_mask[j] = True
            elif skin_tones[img_name] == 56:
                fst_56_mask[j] = True

    n_fst12 = fst_12_mask.sum()
    n_fst56 = fst_56_mask.sum()

    if n_fst12 == 0 or n_fst56 == 0:
        print(f"  Insufficient FST data: FST12={n_fst12}, FST56={n_fst56}")
        return None

    print(f"  FST groups: FST12={n_fst12} traces, FST56={n_fst56} traces")

    # Compute per-concept statistics by FST group
    stats = []
    for row_idx, (_, row) in enumerate(concepts_df.iterrows()):
        concept_row = presence_matrix[row_idx, :]

        # Presence rates by FST
        rate_fst12 = concept_row[fst_12_mask].mean() if n_fst12 > 0 else 0
        rate_fst56 = concept_row[fst_56_mask].mean() if n_fst56 > 0 else 0

        # Disparity (FST12 - FST56)
        disparity = rate_fst12 - rate_fst56

        # Chi-squared test for significant difference
        # Contingency table: [present_fst12, absent_fst12], [present_fst56, absent_fst56]
        present_12 = concept_row[fst_12_mask].sum()
        absent_12 = n_fst12 - present_12
        present_56 = concept_row[fst_56_mask].sum()
        absent_56 = n_fst56 - present_56

        # Only compute p-value if we have enough data
        from scipy.stats import fisher_exact
        try:
            contingency = [[present_12, absent_12], [present_56, absent_56]]
            _, p_value = fisher_exact(contingency)
        except Exception:
            p_value = 1.0

        stats.append({
            'concept': row['concept'],
            'direction': row['direction'],
            'definition': row['definition'],
            'rate_fst12': round(rate_fst12, 3),
            'rate_fst56': round(rate_fst56, 3),
            'disparity': round(disparity, 3),
            'abs_disparity': round(abs(disparity), 3),
            'count_fst12': int(present_12),
            'count_fst56': int(present_56),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05
        })

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('abs_disparity', ascending=False)

    return stats_df


def plot_fst_disparity(fst_stats_df, output_path):
    """Plot grouped bar chart of concept presence by FST group."""
    if fst_stats_df is None or len(fst_stats_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Grouped bar chart
    ax = axes[0]
    x = np.arange(len(fst_stats_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, fst_stats_df['rate_fst12'], width,
                   label='FST I-II (Light)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, fst_stats_df['rate_fst56'], width,
                   label='FST V-VI (Dark)', color='#e67e22', alpha=0.8)

    ax.set_xlabel('Concept')
    ax.set_ylabel('Presence Rate')
    ax.set_title('Concept Presence by Skin Type')
    ax.set_xticks(x)
    ax.set_xticklabels([c[:15] + '...' if len(c) > 15 else c
                        for c in fst_stats_df['concept']], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Right plot: Disparity ranking
    ax2 = axes[1]
    colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in fst_stats_df['disparity']]
    ax2.barh(range(len(fst_stats_df)), fst_stats_df['disparity'], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(fst_stats_df)))
    ax2.set_yticklabels([c[:20] + '...' if len(c) > 20 else c
                         for c in fst_stats_df['concept']])
    ax2.set_xlabel('Disparity (FST12 - FST56)')
    ax2.set_title('Concept Disparity by Skin Type')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # Add significance markers
    for i, (_, row) in enumerate(fst_stats_df.iterrows()):
        if row['significant']:
            ax2.annotate('*', xy=(row['disparity'], i), fontsize=14, color='red',
                        ha='left' if row['disparity'] > 0 else 'right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Cross-Trace Consistency Analysis
# =============================================================================

def compute_cross_seed_consistency(presence_matrix, trace_info, concepts_df):
    """Analyze concept consistency across different seeds for the same image.

    For images that appear in multiple seeds, computes how consistently
    each concept is mentioned (or not mentioned) across seeds.

    Returns:
        consistency_df: DataFrame with per-concept consistency scores
        image_consistency: dict mapping image_name -> consistency score
    """
    # Group traces by image
    from collections import defaultdict
    image_to_traces = defaultdict(list)

    for j, info in enumerate(trace_info):
        img_name = info['image_name']
        image_to_traces[img_name].append((j, info['seed']))

    # Find images with multiple traces across DIFFERENT seeds
    # (not just multiple traces with same seed due to truncated names)
    multi_seed_images = {}
    for img, trace_seed_pairs in image_to_traces.items():
        if len(trace_seed_pairs) > 1:
            seeds = set(seed for _, seed in trace_seed_pairs)
            if len(seeds) > 1:  # Actually different seeds, not just same seed
                indices = [idx for idx, _ in trace_seed_pairs]
                multi_seed_images[img] = indices

    if not multi_seed_images:
        print("  No images with multiple seeds found for consistency analysis")
        print("  (Note: requires same image annotated across different seeds)")
        return None, None

    print(f"  Found {len(multi_seed_images)} images with traces across multiple seeds")

    n_concepts = presence_matrix.shape[0]

    # Compute per-concept consistency
    concept_consistency = []
    for row_idx, (_, row) in enumerate(concepts_df.iterrows()):
        # For each image with multiple traces, check if concept is consistent
        consistent_count = 0
        total_images = 0

        for img_name, indices in multi_seed_images.items():
            values = [presence_matrix[row_idx, j] for j in indices]
            # Consistent if all same (all 0 or all 1)
            if len(set(values)) == 1:
                consistent_count += 1
            total_images += 1

        consistency_rate = consistent_count / total_images if total_images > 0 else 0

        concept_consistency.append({
            'concept': row['concept'],
            'direction': row['direction'],
            'definition': row['definition'],
            'consistency_rate': round(consistency_rate, 3),
            'consistent_images': consistent_count,
            'total_multi_seed_images': total_images
        })

    consistency_df = pd.DataFrame(concept_consistency)
    consistency_df = consistency_df.sort_values('consistency_rate', ascending=True)

    # Compute per-image consistency (across all concepts)
    image_consistency = {}
    for img_name, indices in multi_seed_images.items():
        consistent_concepts = 0
        for row_idx in range(n_concepts):
            values = [presence_matrix[row_idx, j] for j in indices]
            if len(set(values)) == 1:
                consistent_concepts += 1
        image_consistency[img_name] = consistent_concepts / n_concepts if n_concepts > 0 else 0

    return consistency_df, image_consistency


def plot_consistency(consistency_df, output_path):
    """Plot concept consistency across seeds."""
    if consistency_df is None or len(consistency_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(consistency_df) * 0.4), 6))

    # Color by consistency level
    colors = ['#e74c3c' if r < 0.5 else '#f39c12' if r < 0.8 else '#2ecc71'
              for r in consistency_df['consistency_rate']]

    x = range(len(consistency_df))
    ax.bar(x, consistency_df['consistency_rate'], color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Concept')
    ax.set_ylabel('Cross-Seed Consistency Rate')
    ax.set_title('Concept Consistency Across Different Seeds (Same Image)')
    ax.set_xticks(x)
    ax.set_xticklabels([c[:15] + '...' if len(c) > 15 else c
                        for c in consistency_df['concept']], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)

    # Add threshold lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Low consistency')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Medium consistency')

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='High (≥80%)'),
        Patch(facecolor='#f39c12', label='Medium (50-80%)'),
        Patch(facecolor='#e74c3c', label='Low (<50%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# LLM Judge Export
# =============================================================================

def export_for_llm_judge(concepts_df, trace_info, trace_texts, presence_matrix, output_path,
                         annotated_only=True):
    """Export data in format compatible with concept_presence_analysis.py.

    Uses 'definition' column for concept descriptions instead of just names.

    Args:
        annotated_only: If True, only export traces that have at least one annotation (1).
                        This is useful for running the judge only on traces you've reviewed.
    """
    # Build concepts list with definitions
    concepts = []
    for _, row in concepts_df.iterrows():
        concepts.append({
            'concept': row['definition'],  # Use definition for LLM judge
            'original_name': row['concept'],
            'direction': row['direction'],
            'mean_dd': row['mean_dd'],
            'p_value': row['p_value']
        })

    # Deduplicate concepts with identical definitions
    from experiment_utils import deduplicate_concepts
    concepts, presence_matrix = deduplicate_concepts(concepts, presence_matrix)

    # Determine which traces to include
    if annotated_only:
        # Only include traces that have at least one 1 in any concept
        annotated_trace_indices = set()
        for j in range(presence_matrix.shape[1]):
            if presence_matrix[:, j].sum() > 0:
                annotated_trace_indices.add(j)
        print(f"  Found {len(annotated_trace_indices)} annotated traces (with at least one concept marked)")
    else:
        annotated_trace_indices = set(range(len(trace_info)))

    # Build reasoning traces list
    reasoning_traces = []
    for j, info in enumerate(trace_info):
        if j not in annotated_trace_indices:
            continue

        col = info['column']
        trace_text = trace_texts.get(col, '')

        reasoning_traces.append({
            'index': info['index'],
            'seed': info['seed'],
            'image_path': info['image_name'],
            'column': col,  # Include column name for matching back
            'reasoning': trace_text,
            'answer': ''  # Empty as we only analyze reasoning
        })

    output = {
        'top_concepts': concepts,
        'reasoning_traces': reasoning_traces,
        'metadata': {
            'source': 'analyze_annotation_matrix.py',
            'n_concepts': len(concepts),
            'n_traces': len(reasoning_traces),
            'annotated_only': annotated_only
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved LLM judge input: {output_path} ({len(reasoning_traces)} traces)")


# =============================================================================
# Main
# =============================================================================

def find_annotation_csvs(annotations_dir):
    """Find all annotation matrix CSVs in a directory."""
    annotations_dir = Path(annotations_dir)
    csvs = list(annotations_dir.glob('annotation_matrix_*.csv'))
    # Filter out trace_texts.json files
    csvs = [c for c in csvs if not c.name.endswith('_trace_texts.json')]
    return sorted(csvs)


def extract_model_from_csv(csv_path):
    """Extract model name from annotation_matrix_{model}.csv filename."""
    stem = Path(csv_path).stem
    if stem.startswith('annotation_matrix_'):
        return stem.replace('annotation_matrix_', '')
    return stem


def main():
    parser = argparse.ArgumentParser(
        description='Analyze manual concept presence annotations'
    )
    # Single file mode
    parser.add_argument('--annotation_csv', type=str, default=None,
                        help='Path to a single annotation_matrix_{model}.csv')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Results directory with seed_* folders (for single file mode)')

    # Batch mode - all CSVs in one directory
    parser.add_argument('--annotations_dir', type=str, default=None,
                        help='Directory containing all annotation_matrix_*.csv files')
    parser.add_argument('--results_dirs', type=str, nargs='+', default=None,
                        help='List of results directories (matched to CSVs by model name)')

    parser.add_argument('--metadata_csv', type=str,
                        default='/scratch/users/sonnet/ddi/ddi_metadata.csv',
                        help='Path to ddi_metadata.csv for ground truth labels')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: {results_dir}/analysis_outputs/annotation_analysis)')
    args = parser.parse_args()

    # Determine mode
    if args.annotations_dir:
        # Batch mode
        annotation_csvs = find_annotation_csvs(args.annotations_dir)
        if not annotation_csvs:
            print(f"ERROR: No annotation_matrix_*.csv files found in {args.annotations_dir}")
            return 1

        print(f"Found {len(annotation_csvs)} annotation CSVs in {args.annotations_dir}")
        for csv in annotation_csvs:
            print(f"  - {csv.name}")
        print()

        # Match CSVs to results dirs
        results_dir_map = {}
        if args.results_dirs:
            for rd in args.results_dirs:
                model_name = Path(rd).name.split('_DDI_')[0]
                results_dir_map[model_name] = rd

        for csv_path in annotation_csvs:
            model_name = extract_model_from_csv(csv_path)
            results_dir = results_dir_map.get(model_name)

            if not results_dir:
                # Try to find matching results dir
                for rd in (args.results_dirs or []):
                    if model_name in rd:
                        results_dir = rd
                        break

            if not results_dir:
                print(f"WARNING: No results_dir found for {model_name}, skipping...")
                continue

            run_analysis(csv_path, results_dir, args.metadata_csv, args.output_dir)

    elif args.annotation_csv and args.results_dir:
        # Single file mode
        run_analysis(args.annotation_csv, args.results_dir, args.metadata_csv, args.output_dir)

    else:
        parser.error("Must specify either --annotation_csv + --results_dir OR --annotations_dir + --results_dirs")


def run_analysis(annotation_csv, results_dir, metadata_csv, output_dir_arg):
    """Run analysis for a single annotation CSV."""
    annotation_csv = Path(annotation_csv)
    results_dir = Path(results_dir)

    if output_dir_arg:
        # If single output_dir specified, create subdirectory per model
        model_name = extract_model_from_csv(annotation_csv)
        output_dir = Path(output_dir_arg) / model_name
    else:
        output_dir = results_dir / 'analysis_outputs' / 'annotation_analysis'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Annotation Matrix Analysis")
    print("=" * 70)
    print(f"Annotation CSV: {annotation_csv}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print()

    # Load annotation matrix
    print("Loading annotation matrix...")
    concepts_df, presence_matrix, trace_info, trace_columns = load_annotation_matrix(annotation_csv)
    print()

    # Load trace texts - check multiple locations
    # 1. Same directory as annotation CSV
    trace_texts_path = annotation_csv.parent / (annotation_csv.stem + '_trace_texts.json')

    # 2. Fall back to original location in results_dir
    if not trace_texts_path.exists():
        model_name = extract_model_from_csv(annotation_csv)
        alt_path = results_dir / 'analysis_outputs' / 'significant_only' / f'annotation_matrix_{model_name}_trace_texts.json'
        if alt_path.exists():
            trace_texts_path = alt_path

    print(f"Loading trace texts from {trace_texts_path}...")
    trace_texts = load_trace_texts(trace_texts_path)
    print()

    # Load ground truth labels and skin tones
    print("Loading ground truth labels...")
    image_names = [t['image_name'] for t in trace_info]
    labels, skin_tones = load_ground_truth_labels(metadata_csv, image_names)
    print()

    # Load predictions
    print("Loading predictions...")
    predictions = load_predictions_for_traces(results_dir, trace_info)
    print()

    # =================================
    # 1. Frequency Analysis
    # =================================
    print("Computing frequency statistics...")
    freq_stats = compute_frequency_stats(concepts_df, presence_matrix, trace_columns)
    freq_stats.to_csv(output_dir / 'concept_frequency.csv', index=False)
    print(f"  Saved: concept_frequency.csv")

    plot_frequency(freq_stats, output_dir / 'concept_frequency.png')
    print()

    # Print top/bottom concepts with definitions
    print("Most frequently mentioned concepts:")
    for _, row in freq_stats.head(5).iterrows():
        print(f"  {row['concept']}: {row['presence_rate']:.0%} ({row['presence_count']} times)")
        print(f"    Definition: {row['definition']}")

    print("\nLeast frequently mentioned concepts:")
    for _, row in freq_stats.tail(5).iterrows():
        print(f"  {row['concept']}: {row['presence_rate']:.0%} ({row['presence_count']} times)")
        print(f"    Definition: {row['definition']}")
    print()

    # =================================
    # 2. Co-occurrence Analysis
    # =================================
    print("Computing co-occurrence matrix...")
    concept_names = concepts_df['concept'].tolist()
    cooccurrence_df, pairs_df = compute_cooccurrence(presence_matrix, concept_names)

    cooccurrence_df.to_csv(output_dir / 'concept_cooccurrence.csv')
    print(f"  Saved: concept_cooccurrence.csv")

    if len(pairs_df) > 0:
        pairs_df.to_csv(output_dir / 'concept_pairs.csv', index=False)
        print(f"  Saved: concept_pairs.csv ({len(pairs_df)} pairs with Jaccard > 0.5)")

        print("\nTop co-occurring concept pairs (Jaccard > 0.5):")
        for _, row in pairs_df.head(10).iterrows():
            print(f"  {row['concept_1']} + {row['concept_2']}: {row['jaccard']:.2f}")
    else:
        print("  No concept pairs with Jaccard > 0.5 found")
    print()

    plot_cooccurrence_heatmap(cooccurrence_df, output_dir / 'concept_cooccurrence.png')
    plot_concept_clusters(cooccurrence_df, output_dir / 'concept_clusters.png')

    # Print clusters as text (easier to read than dendrogram)
    print_concept_clusters(cooccurrence_df, concepts_df)
    print()

    # =================================
    # 3. Accuracy Analysis
    # =================================
    print("Computing accuracy metrics...")
    accuracy_df, overall_stats = compute_accuracy_metrics(
        concepts_df, presence_matrix, trace_info, predictions, labels
    )

    if accuracy_df is not None:
        accuracy_df.to_csv(output_dir / 'accuracy_by_concept.csv', index=False)
        print(f"  Saved: accuracy_by_concept.csv")

        plot_accuracy_vs_count(presence_matrix, trace_info, predictions, labels,
                               output_dir / 'accuracy_vs_count.png')

        print(f"\nOverall Statistics:")
        print(f"  Traces analyzed: {overall_stats['n_valid']}/{overall_stats['n_traces']}")
        print(f"  Overall accuracy: {overall_stats['overall_accuracy']:.0%}")
        print(f"  Mean concepts per trace: {overall_stats['mean_concepts_per_trace']:.1f}")
        print(f"  Concept count vs accuracy correlation: {overall_stats['concept_count_accuracy_correlation']}")

        # Concepts with biggest accuracy difference
        if 'accuracy_diff' in accuracy_df.columns:
            sorted_by_diff = accuracy_df.dropna(subset=['accuracy_diff']).sort_values(
                'accuracy_diff', ascending=False
            )
            if len(sorted_by_diff) > 0:
                print("\nConcepts most associated with correct predictions:")
                for _, row in sorted_by_diff.head(5).iterrows():
                    print(f"  {row['concept']}: +{row['accuracy_diff']:.0%} when present")

        # Save overall stats
        with open(output_dir / 'summary_stats.json', 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"\n  Saved: summary_stats.json")
    print()

    # =================================
    # 4. FST-Based Fairness Analysis
    # =================================
    if skin_tones:
        print("Computing FST-based concept analysis...")
        fst_stats = compute_fst_concept_stats(presence_matrix, trace_info, skin_tones, concepts_df)

        if fst_stats is not None:
            fst_stats.to_csv(output_dir / 'fst_concept_disparity.csv', index=False)
            print(f"  Saved: fst_concept_disparity.csv")

            plot_fst_disparity(fst_stats, output_dir / 'fst_disparity.png')

            # Print concepts with significant FST differences
            significant = fst_stats[fst_stats['significant']]
            if len(significant) > 0:
                print(f"\n  Concepts with significant FST disparity (p < 0.05):")
                for _, row in significant.iterrows():
                    direction = "more in light skin" if row['disparity'] > 0 else "more in dark skin"
                    print(f"    {row['concept']}: {abs(row['disparity']):.0%} {direction} (p={row['p_value']:.3f})")
            else:
                print("  No concepts with statistically significant FST disparity")
        print()
    else:
        print("Skipping FST analysis (no skin tone data available)")
        print()

    # =================================
    # 5. Cross-Seed Consistency Analysis
    # =================================
    print("Computing cross-seed consistency...")
    consistency_df, image_consistency = compute_cross_seed_consistency(
        presence_matrix, trace_info, concepts_df
    )

    if consistency_df is not None:
        consistency_df.to_csv(output_dir / 'cross_seed_consistency.csv', index=False)
        print(f"  Saved: cross_seed_consistency.csv")

        plot_consistency(consistency_df, output_dir / 'cross_seed_consistency.png')

        # Summary statistics
        mean_consistency = consistency_df['consistency_rate'].mean()
        low_consistency = consistency_df[consistency_df['consistency_rate'] < 0.5]

        print(f"\n  Mean concept consistency across seeds: {mean_consistency:.0%}")

        if len(low_consistency) > 0:
            print(f"\n  Concepts with low cross-seed consistency (<50%):")
            for _, row in low_consistency.iterrows():
                definition = row['definition']
                if len(definition) > 50:
                    definition = definition[:47] + "..."
                print(f"    {row['concept']}: {row['consistency_rate']:.0%}")
                print(f"      Definition: {definition}")

        # Also show high consistency concepts
        high_consistency = consistency_df[consistency_df['consistency_rate'] >= 0.8]
        if len(high_consistency) > 0:
            print(f"\n  Concepts with high cross-seed consistency (>=80%):")
            for _, row in high_consistency.head(10).iterrows():
                definition = row['definition']
                if len(definition) > 50:
                    definition = definition[:47] + "..."
                print(f"    {row['concept']}: {row['consistency_rate']:.0%}")
                print(f"      Definition: {definition}")
    print()

    # =================================
    # 6. LLM Judge Export
    # =================================
    print("Exporting for LLM judge (annotated traces only)...")
    export_for_llm_judge(
        concepts_df, trace_info, trace_texts, presence_matrix,
        output_dir / 'llm_judge_input.json',
        annotated_only=True  # Only export traces you've annotated
    )
    print()

    print("=" * 70)
    print(f"Analysis complete. Outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
