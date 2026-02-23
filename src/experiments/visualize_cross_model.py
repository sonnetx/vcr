"""
Cross-model comparison visualizations for VCR experiments.

Compares VCR sensitivity findings, concept stability, and LLM judge presence
rates across multiple models (e.g., Kimi-VL, R1-Onevision, GLM-4.1V).

Generates:
1. Top concept comparison: grouped bar chart of sensitivities across models
2. Concept overlap Venn/UpSet: which concepts are shared vs model-specific
3. Presence rate comparison: LLM judge findings across models
4. Sensitivity correlation: pairwise scatter between models
5. Summary table: key metrics per model

Usage:
    python src/experiments/visualize_cross_model.py \
        --results_dirs \
            Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_simple_binary \
            R1-Onevision-7B_DDI_R1_malignant_prob_simple_binary \
        --model_names "Kimi-VL" "R1-Onevision" \
        --top_k 20 \
        --output_dir cross_model_comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, OrderedDict
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_model_data(results_dir: Path, model_name: str, top_k: int, seeds: list = None):
    """Load aggregated data for a single model across seeds."""
    if not results_dir.exists():
        print(f"  WARNING: {results_dir} does not exist")
        return None

    # Auto-detect seeds
    if seeds is None:
        seed_dirs = sorted(results_dir.glob('seed_*'))
        seeds = [int(d.name.split('_')[1]) for d in seed_dirs]

    if not seeds:
        print(f"  WARNING: No seeds found in {results_dir}")
        return None

    per_seed_top_concepts = []
    per_seed_sensitivities = {}  # concept -> list of sens values
    per_seed_presence = {}       # concept -> list of presence rates

    for seed in seeds:
        seed_dir = results_dir / f'seed_{seed}'
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

        per_seed_top_concepts.append(top_concepts)

        for rank, idx in enumerate(top_indices):
            concept = concept_texts[int(idx)]
            if concept not in per_seed_sensitivities:
                per_seed_sensitivities[concept] = []
            per_seed_sensitivities[concept].append(float(avg_sens[idx]))

        # LLM judge presence
        presence_dir = seed_dir / 'concept_presence_analysis'
        if presence_dir.exists():
            json_files = sorted(presence_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            if json_files:
                with open(json_files[0], 'r') as f:
                    presence_data = json.load(f)
                per_concept = presence_data.get('summary_statistics', {}).get('per_concept', {})
                for concept, stats in per_concept.items():
                    if concept not in per_seed_presence:
                        per_seed_presence[concept] = []
                    per_seed_presence[concept].append(stats.get('presence_rate_reasoning', 0.0))

    if not per_seed_top_concepts:
        return None

    # Compute stable concepts (appear in >= 50% of seeds)
    n_seeds = len(per_seed_top_concepts)
    concept_seed_count = Counter()
    for tc in per_seed_top_concepts:
        concept_seed_count.update(tc)

    stable_concepts = [c for c, count in concept_seed_count.most_common()
                       if count >= max(1, n_seeds * 0.5)]

    return {
        'model_name': model_name,
        'results_dir': results_dir,
        'n_seeds': n_seeds,
        'per_seed_top_concepts': per_seed_top_concepts,
        'per_seed_sensitivities': per_seed_sensitivities,
        'per_seed_presence': per_seed_presence,
        'concept_seed_count': concept_seed_count,
        'stable_concepts': stable_concepts,
    }


def plot_top_concepts_comparison(models_data, top_k, output_dir):
    """
    Grouped bar chart comparing mean |sensitivity| for the union of
    stable concepts across all models.
    """
    # Get union of stable concepts across models
    all_stable = OrderedDict()
    for md in models_data:
        for c in md['stable_concepts'][:top_k]:
            if c not in all_stable:
                all_stable[c] = {}

    concepts = list(all_stable.keys())
    if not concepts:
        print("  Skipping top concepts comparison: no stable concepts found")
        return

    n_models = len(models_data)
    n_concepts = min(len(concepts), top_k)
    concepts = concepts[:n_concepts]

    # Build data: mean |sensitivity| per model per concept
    sens_data = np.zeros((n_concepts, n_models))
    sens_err = np.zeros((n_concepts, n_models))

    for j, md in enumerate(models_data):
        for i, concept in enumerate(concepts):
            vals = md['per_seed_sensitivities'].get(concept, [])
            if vals:
                sens_data[i, j] = np.mean(np.abs(vals))
                sens_err[i, j] = np.std(np.abs(vals)) if len(vals) > 1 else 0

    # Sort concepts by max sensitivity across models
    max_sens = np.max(sens_data, axis=1)
    sort_idx = np.argsort(-max_sens)
    concepts = [concepts[i] for i in sort_idx]
    sens_data = sens_data[sort_idx]
    sens_err = sens_err[sort_idx]

    fig, ax = plt.subplots(figsize=(14, max(6, n_concepts * 0.4)))
    y = np.arange(n_concepts)
    bar_height = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    for j, md in enumerate(models_data):
        offset = (j - n_models / 2 + 0.5) * bar_height
        ax.barh(y + offset, sens_data[:, j], bar_height * 0.9,
                xerr=sens_err[:, j], label=md['model_name'],
                color=colors[j], edgecolor='black', linewidth=0.3,
                capsize=2)

    ax.set_yticks(y)
    ax.set_yticklabels(concepts, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |Sensitivity| (across seeds)', fontsize=11)
    ax.set_title(f'Top Concept Sensitivities: Cross-Model Comparison\n'
                 f'(stable concepts from ≥50% of seeds per model)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out = output_dir / 'cross_model_sensitivity_comparison.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_concept_overlap(models_data, top_k, output_dir):
    """
    Heatmap showing which concepts are in each model's stable set.
    Rows = union of concepts, columns = models.
    """
    all_concepts = OrderedDict()
    for md in models_data:
        for c in md['stable_concepts'][:top_k]:
            if c not in all_concepts:
                all_concepts[c] = Counter()
            all_concepts[c][md['model_name']] = 1

    concepts = list(all_concepts.keys())
    if not concepts:
        print("  Skipping concept overlap: no concepts found")
        return

    n_concepts = len(concepts)
    n_models = len(models_data)
    model_names = [md['model_name'] for md in models_data]

    # Build presence matrix
    presence = np.zeros((n_concepts, n_models))
    for i, concept in enumerate(concepts):
        for j, md in enumerate(models_data):
            if concept in md['stable_concepts'][:top_k]:
                presence[i, j] = 1.0

    # Sort by number of models sharing the concept
    shared_count = np.sum(presence, axis=1)
    sort_idx = np.argsort(-shared_count)
    presence = presence[sort_idx]
    concepts = [concepts[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2), max(8, n_concepts * 0.35)))

    cmap = sns.color_palette(["#f0f0f0", "#2ecc71"], as_cmap=True)
    sns.heatmap(
        presence, ax=ax, cmap=cmap, vmin=0, vmax=1,
        xticklabels=model_names, yticklabels=concepts,
        linewidths=0.5, linecolor='gray', cbar=False,
        annot=np.where(presence == 1, '✓', ''), fmt='',
        annot_kws={'size': 12, 'color': 'black'}
    )

    # Shared count on right
    for i in range(len(concepts)):
        count = int(np.sum(presence[i]))
        ax.text(n_models + 0.3, i + 0.5, f'{count}/{n_models}',
                va='center', ha='left', fontsize=9)

    ax.set_title(f'Concept Overlap Across Models\n'
                 f'(stable concepts from ≥50% of seeds, top {top_k})',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    out = output_dir / 'cross_model_concept_overlap.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # Print summary
    shared_all = [concepts[i] for i in range(len(concepts)) if np.sum(presence[i]) == n_models]
    print(f"  Concepts shared by ALL models: {shared_all if shared_all else 'none'}")


def plot_presence_rate_comparison(models_data, top_k, output_dir):
    """
    Grouped bar chart of mean LLM judge presence rate per concept across models.
    """
    # Collect concepts that have presence data
    all_concepts = OrderedDict()
    for md in models_data:
        for concept in md['stable_concepts'][:top_k]:
            if concept in md['per_seed_presence']:
                if concept not in all_concepts:
                    all_concepts[concept] = {}
                all_concepts[concept][md['model_name']] = np.mean(md['per_seed_presence'][concept])

    concepts = list(all_concepts.keys())
    if not concepts:
        print("  Skipping presence rate comparison: no LLM judge data found")
        return

    n_concepts = min(len(concepts), top_k)
    concepts = concepts[:n_concepts]
    n_models = len(models_data)
    model_names = [md['model_name'] for md in models_data]

    rate_data = np.zeros((n_concepts, n_models))
    for i, concept in enumerate(concepts):
        for j, md in enumerate(models_data):
            vals = md['per_seed_presence'].get(concept, [])
            if vals:
                rate_data[i, j] = np.mean(vals)

    # Sort by max presence rate
    max_rate = np.max(rate_data, axis=1)
    sort_idx = np.argsort(-max_rate)
    concepts = [concepts[i] for i in sort_idx]
    rate_data = rate_data[sort_idx]

    fig, ax = plt.subplots(figsize=(14, max(6, n_concepts * 0.4)))
    y = np.arange(n_concepts)
    bar_height = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    for j, md in enumerate(models_data):
        offset = (j - n_models / 2 + 0.5) * bar_height
        ax.barh(y + offset, rate_data[:, j], bar_height * 0.9,
                label=md['model_name'], color=colors[j],
                edgecolor='black', linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(concepts, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Presence Rate (LLM Judge)', fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title(f'LLM Judge Presence Rate: Cross-Model Comparison',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out = output_dir / 'cross_model_presence_rate.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_sensitivity_correlation(models_data, top_k, output_dir):
    """
    Pairwise scatter plots of concept sensitivities between models.
    Uses the union of all concepts, with 0 for concepts not in a model's top-K.
    """
    if len(models_data) < 2:
        return

    # Build concept -> mean |sensitivity| per model
    all_concepts = set()
    for md in models_data:
        all_concepts.update(md['per_seed_sensitivities'].keys())

    concepts = sorted(all_concepts)
    model_sens = {}
    for md in models_data:
        model_sens[md['model_name']] = np.array([
            np.mean(np.abs(md['per_seed_sensitivities'].get(c, [0])))
            for c in concepts
        ])

    pairs = list(combinations(range(len(models_data)), 2))
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[0, idx]
        name_i = models_data[i]['model_name']
        name_j = models_data[j]['model_name']
        x = model_sens[name_i]
        y = model_sens[name_j]

        # Only plot concepts that are nonzero in at least one model
        mask = (x > 0) | (y > 0)
        x_plot, y_plot = x[mask], y[mask]
        concepts_plot = [c for c, m in zip(concepts, mask) if m]

        ax.scatter(x_plot, y_plot, s=30, alpha=0.6, edgecolors='black', linewidth=0.3)

        # Label top points
        top_idx = np.argsort(-(x_plot + y_plot))[:8]
        for ti in top_idx:
            ax.annotate(concepts_plot[ti], (x_plot[ti], y_plot[ti]),
                        textcoords="offset points", xytext=(4, 4), fontsize=6, alpha=0.7)

        # Correlation
        if len(x_plot) > 2:
            corr = np.corrcoef(x_plot, y_plot)[0, 1]
            ax.set_title(f'{name_i} vs {name_j}\n(r = {corr:.3f})', fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'{name_i} vs {name_j}', fontsize=11, fontweight='bold')

        ax.set_xlabel(f'{name_i} |sensitivity|', fontsize=10)
        ax.set_ylabel(f'{name_j} |sensitivity|', fontsize=10)

        # Diagonal reference
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.2)

    plt.suptitle('Cross-Model Sensitivity Correlation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = output_dir / 'cross_model_sensitivity_correlation.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def generate_summary_table(models_data, top_k, output_dir):
    """Generate a text summary table with key metrics per model."""
    lines = []
    lines.append("=" * 80)
    lines.append("Cross-Model Summary")
    lines.append("=" * 80)

    header = f"{'Model':<25} {'Seeds':>5} {'Stable':>7} {'Mean |Sens|':>12} {'Mean Pres':>10} {'Jaccard':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for md in models_data:
        n_seeds = md['n_seeds']
        n_stable = len(md['stable_concepts'])

        # Mean sensitivity of stable concepts
        sens_vals = []
        for c in md['stable_concepts'][:top_k]:
            vals = md['per_seed_sensitivities'].get(c, [])
            if vals:
                sens_vals.append(np.mean(np.abs(vals)))
        mean_sens = np.mean(sens_vals) if sens_vals else 0

        # Mean presence rate
        pres_vals = []
        for c in md['stable_concepts'][:top_k]:
            vals = md['per_seed_presence'].get(c, [])
            if vals:
                pres_vals.append(np.mean(vals))
        mean_pres = np.mean(pres_vals) if pres_vals else float('nan')

        # Mean pairwise Jaccard across seeds
        jaccard_vals = []
        for si, sj in combinations(range(len(md['per_seed_top_concepts'])), 2):
            set_i = set(md['per_seed_top_concepts'][si][:top_k])
            set_j = set(md['per_seed_top_concepts'][sj][:top_k])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard_vals.append(inter / union if union > 0 else 0)
        mean_jaccard = np.mean(jaccard_vals) if jaccard_vals else float('nan')

        pres_str = f"{mean_pres:.3f}" if not np.isnan(mean_pres) else "N/A"
        jacc_str = f"{mean_jaccard:.3f}" if not np.isnan(mean_jaccard) else "N/A"

        lines.append(f"{md['model_name']:<25} {n_seeds:>5} {n_stable:>7} {mean_sens:>12.6f} {pres_str:>10} {jacc_str:>8}")

    lines.append("=" * 80)

    # Cross-model overlap
    lines.append("")
    lines.append("Cross-Model Concept Overlap:")
    for i, j in combinations(range(len(models_data)), 2):
        set_i = set(models_data[i]['stable_concepts'][:top_k])
        set_j = set(models_data[j]['stable_concepts'][:top_k])
        inter = len(set_i & set_j)
        union = len(set_i | set_j)
        jaccard = inter / union if union > 0 else 0
        lines.append(f"  {models_data[i]['model_name']} ∩ {models_data[j]['model_name']}: "
                     f"{inter} shared, Jaccard={jaccard:.3f}")

    shared_all = set(models_data[0]['stable_concepts'][:top_k])
    for md in models_data[1:]:
        shared_all &= set(md['stable_concepts'][:top_k])
    lines.append(f"  Shared by ALL models: {sorted(shared_all) if shared_all else 'none'}")

    summary_text = "\n".join(lines)
    print(summary_text)

    out = output_dir / 'cross_model_summary.txt'
    with open(out, 'w') as f:
        f.write(summary_text)
    print(f"\n  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description='Generate cross-model comparison visualizations')
    parser.add_argument('--results_dirs', type=str, nargs='+', required=True,
                        help='Result directories for each model')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='Display names for each model (default: infer from directory)')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top concepts to compare')
    parser.add_argument('--output_dir', type=str, default='cross_model_comparison',
                        help='Output directory for visualizations')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to include (default: auto-detect per model)')
    args = parser.parse_args()

    if args.model_names and len(args.model_names) != len(args.results_dirs):
        print("ERROR: --model_names must have same length as --results_dirs")
        return

    # Infer model names from directory names if not provided
    if args.model_names is None:
        args.model_names = []
        for rd in args.results_dirs:
            name = Path(rd).name.split('_DDI_')[0]
            args.model_names.append(name)

    print("=" * 60)
    print("Cross-Model Comparison")
    print("=" * 60)

    models_data = []
    for rd, name in zip(args.results_dirs, args.model_names):
        print(f"\nLoading: {name} ({rd})")
        data = load_model_data(Path(rd), name, args.top_k, args.seeds)
        if data:
            models_data.append(data)
            print(f"  {data['n_seeds']} seeds, {len(data['stable_concepts'])} stable concepts")
        else:
            print(f"  FAILED to load")

    if len(models_data) < 2:
        print("\nERROR: Need at least 2 models with data for comparison")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations in: {output_dir}")
    plot_top_concepts_comparison(models_data, args.top_k, output_dir)
    plot_concept_overlap(models_data, args.top_k, output_dir)
    plot_presence_rate_comparison(models_data, args.top_k, output_dir)
    plot_sensitivity_correlation(models_data, args.top_k, output_dir)
    generate_summary_table(models_data, args.top_k, output_dir)

    print(f"\nAll cross-model visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
