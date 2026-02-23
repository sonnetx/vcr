"""
Cross-seed agreement visualizations for VCR experiments.

Reads per-seed VCR sensitivities and concept presence analysis results
to visualize how consistent the findings are across random seeds.

Generates:
1. Concept sensitivity rank heatmap across seeds
2. Pairwise Jaccard similarity matrix between seeds
3. Concept stability bar chart (how many seeds each concept appears in top-K)
4. LLM judge presence rate comparison across seeds
5. Sensitivity vs. reasoning presence scatter plot

Usage:
    python src/experiments/visualize_cross_seed.py \
        --results_dir Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_simple_binary \
        --top_k 20
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_seed_data(results_dir: Path, seeds: list, top_k: int):
    """Load per-seed sensitivities, concept texts, and concept presence results."""
    seed_data = []

    for seed in seeds:
        seed_dir = results_dir / f'seed_{seed}'
        if not seed_dir.exists():
            continue

        entry = {'seed': seed, 'seed_dir': seed_dir}

        # Load sensitivities and concepts
        sens_path = seed_dir / 'weighted_sens.npy'
        concepts_path = seed_dir / 'concept_texts.json'
        if not sens_path.exists() or not concepts_path.exists():
            continue

        weighted_sens = np.load(sens_path)
        with open(concepts_path, 'r') as f:
            concept_texts = json.load(f)

        avg_sens = np.mean(weighted_sens, axis=0)
        top_indices = np.argsort(np.abs(avg_sens))[-top_k:][::-1]
        top_concepts = [concept_texts[int(i)] for i in top_indices]
        top_sensitivities = [float(avg_sens[i]) for i in top_indices]

        entry['concept_texts'] = concept_texts
        entry['avg_sens'] = avg_sens
        entry['top_indices'] = top_indices
        entry['top_concepts'] = top_concepts
        entry['top_sensitivities'] = top_sensitivities

        # Load concept presence analysis (LLM judge) if available
        presence_dir = seed_dir / 'concept_presence_analysis'
        if presence_dir.exists():
            json_files = sorted(presence_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            if json_files:
                with open(json_files[0], 'r') as f:
                    entry['concept_presence'] = json.load(f)

        seed_data.append(entry)

    return seed_data


def plot_sensitivity_rank_heatmap(seed_data, top_k, output_dir):
    """
    Heatmap showing each concept's rank across seeds.
    Rows = union of all top-K concepts, columns = seeds.
    Cell color = rank in that seed (darker = higher rank).
    """
    # Collect all concepts that appear in any seed's top-K
    all_top_concepts = []
    concept_seed_count = Counter()
    for sd in seed_data:
        concept_seed_count.update(sd['top_concepts'])
        for c in sd['top_concepts']:
            if c not in all_top_concepts:
                all_top_concepts.append(c)

    # Sort by number of seeds they appear in (most stable first)
    all_top_concepts.sort(key=lambda c: (-concept_seed_count[c], c))

    # Build rank matrix (NaN if concept not in that seed's top-K)
    n_concepts = len(all_top_concepts)
    n_seeds = len(seed_data)
    rank_matrix = np.full((n_concepts, n_seeds), np.nan)

    for j, sd in enumerate(seed_data):
        for rank, concept in enumerate(sd['top_concepts'], 1):
            if concept in all_top_concepts:
                i = all_top_concepts.index(concept)
                rank_matrix[i, j] = rank

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, n_seeds * 1.5), max(10, n_concepts * 0.4)))

    # Use a reversed colormap so rank 1 = darkest
    cmap = sns.color_palette("YlOrRd_r", as_cmap=True)
    cmap.set_bad(color='white')

    sns.heatmap(
        rank_matrix, ax=ax, cmap=cmap, vmin=1, vmax=top_k,
        xticklabels=[f"Seed {sd['seed']}" for sd in seed_data],
        yticklabels=all_top_concepts,
        annot=True, fmt='.0f', annot_kws={'size': 8},
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': f'Rank (1 = most sensitive)'}
    )

    # Add seed count annotations on the right
    for i, concept in enumerate(all_top_concepts):
        count = concept_seed_count[concept]
        ax.text(n_seeds + 0.3, i + 0.5, f'{count}/{n_seeds}',
                va='center', ha='left', fontsize=8, color='#333333')

    ax.set_title(f'Top-{top_k} Concept Sensitivity Ranks Across Seeds\n(blank = not in top-{top_k} for that seed)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Random Seed', fontsize=11)
    ax.set_ylabel('Concept', fontsize=11)

    plt.tight_layout()
    out = output_dir / 'cross_seed_rank_heatmap.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_jaccard_matrix(seed_data, top_k, output_dir):
    """Pairwise Jaccard similarity heatmap between seeds' top-K concept sets."""
    n = len(seed_data)
    jaccard = np.ones((n, n))

    for i, j in combinations(range(n), 2):
        set_i = set(seed_data[i]['top_concepts'][:top_k])
        set_j = set(seed_data[j]['top_concepts'][:top_k])
        inter = len(set_i & set_j)
        union = len(set_i | set_j)
        j_val = inter / union if union > 0 else 0.0
        jaccard[i, j] = j_val
        jaccard[j, i] = j_val

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [f"Seed {sd['seed']}" for sd in seed_data]

    sns.heatmap(
        jaccard, ax=ax, cmap='Blues', vmin=0, vmax=1,
        xticklabels=labels, yticklabels=labels,
        annot=True, fmt='.2f', linewidths=1, linecolor='white',
        cbar_kws={'label': 'Jaccard Similarity'}
    )

    mean_j = np.mean([jaccard[i, j] for i, j in combinations(range(n), 2)])
    ax.set_title(f'Pairwise Jaccard Similarity of Top-{top_k} Concepts\n(mean = {mean_j:.3f})',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    out = output_dir / 'cross_seed_jaccard_matrix.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_concept_stability(seed_data, top_k, output_dir):
    """
    Bar chart: for each concept in the union of top-K across seeds,
    how many seeds include it. Colored by stability tier.
    """
    n_seeds = len(seed_data)
    concept_seed_count = Counter()
    for sd in seed_data:
        concept_seed_count.update(sd['top_concepts'][:top_k])

    # Sort by count descending
    sorted_concepts = concept_seed_count.most_common()
    concepts = [c for c, _ in sorted_concepts]
    counts = [n for _, n in sorted_concepts]

    # Color by stability
    colors = []
    for count in counts:
        if count == n_seeds:
            colors.append('#2ecc71')   # green: in all seeds
        elif count >= n_seeds * 0.75:
            colors.append('#3498db')   # blue: >=75%
        elif count >= n_seeds * 0.5:
            colors.append('#f39c12')   # orange: >=50%
        else:
            colors.append('#e74c3c')   # red: <50%

    fig, ax = plt.subplots(figsize=(14, max(6, len(concepts) * 0.3)))
    y_pos = np.arange(len(concepts))
    ax.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(concepts, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Seeds in Top-K', fontsize=11)
    ax.set_title(f'Concept Stability: How Many Seeds Include Each Concept in Top-{top_k}\n'
                 f'(green={n_seeds}/{n_seeds}, blue>=75%, orange>=50%, red<50%)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, n_seeds + 0.5)
    ax.axvline(x=n_seeds, color='green', linestyle='--', alpha=0.5, label=f'All {n_seeds} seeds')
    ax.axvline(x=n_seeds * 0.75, color='blue', linestyle='--', alpha=0.5, label='75%')
    ax.axvline(x=n_seeds * 0.5, color='orange', linestyle='--', alpha=0.5, label='50%')

    # Add count labels
    for i, count in enumerate(counts):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=8)

    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    out = output_dir / 'cross_seed_concept_stability.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_presence_rate_across_seeds(seed_data, top_k, output_dir):
    """
    For seeds that have LLM judge results, show concept presence rate
    (fraction of traces where judge found concept) across seeds.
    """
    seeds_with_presence = [sd for sd in seed_data if 'concept_presence' in sd]
    if not seeds_with_presence:
        print("  Skipping presence rate plot: no LLM judge results found")
        return

    # Collect all concepts across seeds
    all_concepts = set()
    for sd in seeds_with_presence:
        per_concept = sd['concept_presence']['summary_statistics']['per_concept']
        all_concepts.update(per_concept.keys())

    # Build presence rate matrix
    concepts_list = sorted(all_concepts)
    n_concepts = len(concepts_list)
    n_seeds = len(seeds_with_presence)

    rate_matrix = np.zeros((n_concepts, n_seeds))
    for j, sd in enumerate(seeds_with_presence):
        per_concept = sd['concept_presence']['summary_statistics']['per_concept']
        for i, concept in enumerate(concepts_list):
            if concept in per_concept:
                rate_matrix[i, j] = per_concept[concept]['presence_rate_reasoning']

    # Sort by mean presence rate
    mean_rates = np.mean(rate_matrix, axis=1)
    sort_idx = np.argsort(-mean_rates)
    rate_matrix = rate_matrix[sort_idx]
    concepts_list = [concepts_list[i] for i in sort_idx]

    # Take top concepts by mean rate
    show_k = min(top_k, n_concepts)
    rate_matrix = rate_matrix[:show_k]
    concepts_list = concepts_list[:show_k]

    fig, ax = plt.subplots(figsize=(max(8, n_seeds * 1.5), max(8, show_k * 0.4)))

    sns.heatmap(
        rate_matrix, ax=ax, cmap='YlGnBu', vmin=0, vmax=1,
        xticklabels=[f"Seed {sd['seed']}" for sd in seeds_with_presence],
        yticklabels=concepts_list,
        annot=True, fmt='.0%', annot_kws={'size': 8},
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'Presence Rate (LLM Judge)'}
    )

    ax.set_title(f'LLM Judge Concept Presence Rate Across Seeds\n(fraction of traces where concept was found in reasoning)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Random Seed', fontsize=11)

    plt.tight_layout()
    out = output_dir / 'cross_seed_presence_rate_heatmap.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_sensitivity_vs_presence(seed_data, top_k, output_dir):
    """
    Scatter plot: average sensitivity (x) vs LLM judge presence rate (y)
    for each concept, pooled across seeds. Shows whether sensitive concepts
    are actually discussed in reasoning.
    """
    seeds_with_presence = [sd for sd in seed_data if 'concept_presence' in sd]
    if not seeds_with_presence:
        print("  Skipping sensitivity vs presence plot: no LLM judge results found")
        return

    # Collect per-concept data across seeds
    concept_sens = {}
    concept_presence = {}

    for sd in seeds_with_presence:
        per_concept = sd['concept_presence']['summary_statistics']['per_concept']
        for rank, concept in enumerate(sd['top_concepts'][:top_k]):
            idx = sd['top_indices'][rank]
            sens_val = abs(float(sd['avg_sens'][idx]))

            if concept not in concept_sens:
                concept_sens[concept] = []
                concept_presence[concept] = []

            concept_sens[concept].append(sens_val)
            if concept in per_concept:
                concept_presence[concept].append(per_concept[concept]['presence_rate_reasoning'])
            else:
                concept_presence[concept].append(0.0)

    # Average across seeds
    concepts = list(concept_sens.keys())
    avg_sens = [np.mean(concept_sens[c]) for c in concepts]
    avg_pres = [np.mean(concept_presence[c]) for c in concepts]
    n_seeds_per = [len(concept_sens[c]) for c in concepts]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(avg_sens, avg_pres, c=n_seeds_per, cmap='viridis',
                         s=80, edgecolors='black', linewidth=0.5, vmin=1, vmax=len(seed_data))

    # Label points
    for i, concept in enumerate(concepts):
        ax.annotate(concept, (avg_sens[i], avg_pres[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel('Mean |Sensitivity| (across seeds)', fontsize=11)
    ax.set_ylabel('Mean LLM Judge Presence Rate (across seeds)', fontsize=11)
    ax.set_title('VCR Sensitivity vs. Reasoning Presence\n'
                 '(color = number of seeds concept appears in top-K)',
                 fontsize=13, fontweight='bold')

    plt.colorbar(scatter, ax=ax, label='# Seeds in Top-K')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'cross_seed_sensitivity_vs_presence.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate cross-seed agreement visualizations')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Results directory containing seed_* subdirectories')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to include (default: auto-detect)')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top concepts to analyze')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        return

    # Auto-detect seeds
    if args.seeds is None:
        seed_dirs = sorted(results_dir.glob('seed_*'))
        seeds = [int(d.name.split('_')[1]) for d in seed_dirs]
    else:
        seeds = args.seeds

    print(f"Loading data for seeds: {seeds}")
    seed_data = load_seed_data(results_dir, seeds, args.top_k)
    print(f"Loaded {len(seed_data)} seeds with data")

    if len(seed_data) < 2:
        print("ERROR: Need at least 2 seeds for cross-seed analysis")
        return

    # Output dir: inside the results dir
    output_dir = results_dir / 'cross_seed_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations in: {output_dir}")
    plot_sensitivity_rank_heatmap(seed_data, args.top_k, output_dir)
    plot_jaccard_matrix(seed_data, args.top_k, output_dir)
    plot_concept_stability(seed_data, args.top_k, output_dir)
    plot_presence_rate_across_seeds(seed_data, args.top_k, output_dir)
    plot_sensitivity_vs_presence(seed_data, args.top_k, output_dir)

    print(f"\nAll cross-seed visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
