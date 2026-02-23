"""
Analyze manual concept presence annotations from VCR experiments.

Reads annotation CSVs (one per model), parses concept metadata from column headers,
loads model predictions from choice_differences.npy, and produces:
  1. Concept presence rate bar chart (per model)
  2. Trace x Concept heatmap sorted by prediction correctness
  3. Concept count vs prediction accuracy (per model + pooled)
  4. Cross-model summary: mean concept usage vs accuracy
  5. Concept summary table (CSV) with accuracy-when-present/absent
  6. Summary statistics with median-split accuracy analysis

Usage:
    python src/experiments/analyze_annotations.py \
        --results_dirs \
            R1-Onevision-7B_DDI_R1_malignant_prob_simple_binary \
            GLM-4.1V-9B-Thinking_DDI_GLM4_malignant_prob_simple_binary \
            Kimi-VL-A3B-Thinking_DDI_KimiVL_malignant_prob_simple_binary \
        --output_dir analysis_outputs/annotation_analysis
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob


def parse_column_header(header):
    """Parse concept name and semantic tags from column header.

    Header format: 'concept_name <MEANING> ... </MEANING> <NEGATIVE> ... </NEGATIVE>'
    Returns (concept_name, meaning, negative).
    """
    meaning_match = re.search(r'<MEANING>\s*(.*?)\s*</MEANING>', header, re.IGNORECASE)
    negative_match = re.search(r'<NEGATIVE>\s*(.*?)\s*</NEGATIVE>', header, re.IGNORECASE)

    # Concept name is everything before the first tag
    name = re.split(r'<', header)[0].strip()

    meaning = meaning_match.group(1).strip() if meaning_match else ''
    negative = negative_match.group(1).strip() if negative_match else ''

    return name, meaning, negative


def load_annotation_csv(csv_path):
    """Load annotation CSV and parse column headers.

    Returns:
        df: DataFrame with standardized concept column names
        concept_meta: list of dicts with keys: name, meaning, negative, original_col
        metadata_cols: list of non-concept column names
    """
    df = pd.read_csv(csv_path)

    metadata_cols = ['trace_index', 'image_path', 'label', 'reasoning']
    concept_meta = []
    rename_map = {}

    for col in df.columns:
        if col in metadata_cols:
            continue
        name, meaning, negative = parse_column_header(col)
        concept_meta.append({
            'name': name,
            'meaning': meaning,
            'negative': negative,
            'original_col': col,
        })
        rename_map[col] = name

    df = df.rename(columns=rename_map)

    # Convert concept columns to numeric: empty cells → 0, non-numeric → NaN
    concept_names = [c['name'] for c in concept_meta]
    for c in concept_names:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # Drop concepts that were not annotated (all zeros across all traces)
    annotated = [c for c in concept_names if df[c].sum() > 0 or df[c].notna().all()]
    # Keep columns where at least one row has a 1, OR the column had explicit 0s
    # Since empty→0, we check if the original CSV had any non-empty values
    # Re-read to check: if original column was entirely empty, drop it
    df_raw = pd.read_csv(csv_path)
    raw_rename = {col: parse_column_header(col)[0] for col in df_raw.columns if col not in metadata_cols}
    df_raw = df_raw.rename(columns=raw_rename)

    kept_concepts = []
    kept_meta = []
    for c, meta in zip(concept_names, concept_meta):
        if c in df_raw.columns:
            # Keep if any cell in the original CSV was non-empty
            has_any_value = df_raw[c].notna().any() and (df_raw[c].astype(str).str.strip() != '').any()
            if has_any_value:
                kept_concepts.append(c)
                kept_meta.append(meta)

    if len(kept_concepts) < len(concept_names):
        dropped = set(concept_names) - set(kept_concepts)
        print(f"  Dropped {len(dropped)} unannotated concepts: {dropped}")

    return df, kept_meta, metadata_cols


def find_annotation_csv(results_dir, annotations_dir=None):
    """Find annotation CSV for a given results dir.

    Searches in annotations_dir (if provided) or in results_dir itself.
    Matches by checking if the model name portion (before _DDI_) appears in the CSV filename.
    """
    search_dir = Path(annotations_dir) if annotations_dir else Path(results_dir)
    dir_name = Path(results_dir).name

    # Extract the model identifier (part before _DDI_)
    model_id = dir_name.split('_DDI_')[0] if '_DDI_' in dir_name else dir_name

    all_csvs = sorted(search_dir.glob('annotations_positive_*.csv'))
    if not all_csvs:
        print(f"  No annotations_positive_*.csv files found in {search_dir}")
        return None

    print(f"  Looking for model_id='{model_id}' in {len(all_csvs)} CSVs in {search_dir}")

    for csv_path in all_csvs:
        stem = csv_path.stem
        # Check if model_id appears anywhere in the filename
        if model_id in stem:
            return str(csv_path)
        # Also check if the CSV's model part (after annotations_positive_) overlaps
        csv_model_part = stem.replace('annotations_positive_', '')
        if csv_model_part in dir_name or dir_name in csv_model_part:
            return str(csv_path)

    # Last resort: list what was found for debugging
    print(f"  Available CSVs: {[p.name for p in all_csvs]}")
    return None


def load_predictions(results_dir, trace_indices):
    """Load model predictions from choice_differences.npy for given trace indices.

    Returns dict with:
        'predicted_labels': list of 'Malignant'/'Benign' per trace (thresholded at 0)
        'log_probs': numpy array of log-probs
    """
    result = {'predicted_labels': None, 'log_probs': None}

    pred_path = Path(results_dir) / 'seed_0' / 'choice_differences.npy'
    if not pred_path.exists():
        print(f"  Warning: {pred_path} not found")
        return result

    all_preds = np.load(pred_path)
    valid = [i for i in trace_indices if i < len(all_preds)]
    if not valid:
        return result

    log_probs = all_preds[valid]
    result['log_probs'] = log_probs
    # Threshold at 0: positive log-prob → Malignant, negative → Benign
    result['predicted_labels'] = ['Malignant' if lp >= 0 else 'Benign' for lp in log_probs]

    return result


def extract_model_name(results_dir):
    """Extract short model name from results directory name."""
    name = Path(results_dir).name
    # Take the part before _DDI_
    parts = name.split('_DDI_')
    return parts[0] if parts else name


# ---- Plotting ----

def plot_presence_rates(model_data, output_dir):
    """Bar chart of concept presence rates, one subplot per model."""
    models = list(model_data.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[0, i]
        concepts = model_data[model]['concept_names']
        rates = model_data[model]['presence_rates']
        x = np.arange(len(concepts))

        ax.bar(x, rates, color=f'C{i}')
        ax.set_xlabel('Concept')
        ax.set_ylabel('Presence Rate')
        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels(concepts, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)

    plt.suptitle('Concept Presence Rate in Reasoning Traces', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'concept_presence_rates.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: concept_presence_rates.png")


def add_correctness_column(df, predictions):
    """Add 'correct' column to df based on predicted vs ground truth labels."""
    pred_labels = predictions.get('predicted_labels')
    if pred_labels is None:
        df['correct'] = np.nan
        return df
    n = min(len(df), len(pred_labels))
    correct = []
    for i in range(n):
        gt = df['label'].iloc[i]
        pred = pred_labels[i]
        correct.append(1 if gt == pred else 0)
    df['correct'] = correct[:n] if n == len(df) else correct + [np.nan] * (len(df) - n)
    return df


def plot_concept_heatmap(model_data, output_dir):
    """Trace x Concept heatmap per model, rows sorted by correctness."""
    from matplotlib.colors import ListedColormap

    for model, data in model_data.items():
        df = data['df'].copy()
        concept_names = data['concept_names']

        # Sort: correct predictions first, then incorrect
        df = df.sort_values('correct', ascending=False).reset_index(drop=True)

        matrix = df[concept_names].values.astype(float)
        correct_vals = df['correct'].values

        fig, ax = plt.subplots(figsize=(max(8, len(concept_names) * 0.8), max(4, len(df) * 0.4)))

        # Heatmap
        cmap = ListedColormap(['#f0f0f0', '#2c7bb6'])
        ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')

        # Correctness sidebar
        for i, c in enumerate(correct_vals):
            color = '#4daf4a' if c == 1 else '#e41a1c'
            ax.add_patch(plt.Rectangle((-1.2, i - 0.5), 0.8, 1, color=color, clip_on=False))

        ax.set_xticks(range(len(concept_names)))
        ax.set_xticklabels(concept_names, rotation=45, ha='right')
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"trace {int(r)}" for r in df['trace_index'].values], fontsize=8)
        ax.set_title(f'{model}')
        ax.set_xlim(-0.5, len(concept_names) - 0.5)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4daf4a', label='Correct prediction'),
            Patch(facecolor='#e41a1c', label='Incorrect prediction'),
            Patch(facecolor='#2c7bb6', label='Concept present'),
            Patch(facecolor='#f0f0f0', edgecolor='gray', label='Concept absent'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)

        plt.tight_layout()
        safe_name = model.replace(' ', '_')
        plt.savefig(output_dir / f'concept_heatmap_{safe_name}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: concept_heatmap_{safe_name}.png")


def plot_concept_count_vs_accuracy(model_data, output_dir):
    """Concept count vs accuracy, pooled across models with per-model panels."""
    models = list(model_data.keys())
    n_models = len(models)

    # --- Per-model panels ---
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[0, i]
        df = model_data[model]['df']
        concept_names = model_data[model]['concept_names']

        counts = df[concept_names].sum(axis=1).values
        correct = df['correct'].values

        # Jittered scatter of individual points
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(counts))
        colors = ['#4daf4a' if c == 1 else '#e41a1c' for c in correct]
        ax.scatter(counts + jitter, correct + jitter * 0.3, c=colors, s=60,
                   edgecolors='black', alpha=0.8, zorder=3)

        # Mean accuracy at each concept count
        unique_counts = sorted(set(counts))
        mean_acc = [correct[counts == c].mean() for c in unique_counts]
        n_at_count = [int((counts == c).sum()) for c in unique_counts]
        ax.plot(unique_counts, mean_acc, 'k-o', markersize=6, zorder=4, label='Mean accuracy')

        # Annotate counts
        for uc, ma, n in zip(unique_counts, mean_acc, n_at_count):
            ax.annotate(f'n={n}', (uc, ma), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=7, color='gray')

        ax.set_xlabel('# VCR Concepts in Reasoning')
        ax.set_ylabel('Prediction Correct')
        ax.set_title(model)
        ax.set_ylim(-0.15, 1.25)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Incorrect', 'Correct'])

    plt.suptitle('Concept Usage vs Prediction Accuracy', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / 'concept_count_vs_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: concept_count_vs_accuracy.png")


def plot_cross_model_summary(model_data, output_dir):
    """Cross-model bar chart: mean concept count and accuracy side by side."""
    models = list(model_data.keys())
    mean_counts = []
    accuracies = []

    for model in models:
        df = model_data[model]['df']
        concept_names = model_data[model]['concept_names']
        counts = df[concept_names].sum(axis=1)
        mean_counts.append(counts.mean())
        correct = df['correct']
        accuracies.append(correct.mean() if correct.notna().any() else 0)

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(6, len(models) * 2.5), 5))

    bars1 = ax1.bar(x - width / 2, mean_counts, width, label='Mean # Concepts', color='#2c7bb6', alpha=0.8)
    ax1.set_ylabel('Mean # Concepts in Reasoning', color='#2c7bb6')
    ax1.tick_params(axis='y', labelcolor='#2c7bb6')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, accuracies, width, label='Accuracy', color='#4daf4a', alpha=0.8)
    ax2.set_ylabel('Prediction Accuracy', color='#4daf4a')
    ax2.tick_params(axis='y', labelcolor='#4daf4a')
    ax2.set_ylim(0, 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.set_title('Cross-Model: Concept Usage vs Accuracy')

    # Combined legend
    ax1.legend(handles=[bars1, bars2], labels=['Mean # Concepts', 'Accuracy'],
               loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_model_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cross_model_summary.png")


def save_concept_summary(model_data, output_dir):
    """Save qualitative concept summary table as CSV."""
    rows = []

    for model, data in model_data.items():
        df = data['df']
        concept_meta = data['concept_meta']
        concept_names = data['concept_names']

        for meta in concept_meta:
            name = meta['name']
            rate = df[name].mean()

            # Accuracy when concept present vs absent
            present = df[df[name] == 1]
            absent = df[df[name] == 0]
            acc_present = present['correct'].mean() if len(present) > 0 else float('nan')
            acc_absent = absent['correct'].mean() if len(absent) > 0 else float('nan')

            rows.append({
                'model': model,
                'concept': name,
                'meaning': meta['meaning'],
                'negative': meta['negative'],
                'presence_rate': round(rate, 3),
                'n_present': int(len(present)),
                'accuracy_when_present': round(acc_present, 3) if not np.isnan(acc_present) else '',
                'n_absent': int(len(absent)),
                'accuracy_when_absent': round(acc_absent, 3) if not np.isnan(acc_absent) else '',
            })

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / 'concept_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: concept_summary.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("Concept Summary")
    print("=" * 80)
    for model in model_data:
        model_rows = summary_df[summary_df['model'] == model]
        print(f"\n--- {model} ---")
        print(f"{'Concept':20s} {'Rate':>6s} {'Acc(pres)':>10s} {'Acc(abs)':>10s}  Meaning")
        print("-" * 80)
        for _, r in model_rows.iterrows():
            meaning_short = r['meaning'][:40] + '...' if len(str(r['meaning'])) > 40 else r['meaning']
            acc_p = f"{r['accuracy_when_present']:.0%}" if r['accuracy_when_present'] != '' else 'n/a'
            acc_a = f"{r['accuracy_when_absent']:.0%}" if r['accuracy_when_absent'] != '' else 'n/a'
            print(f"{r['concept']:20s} {r['presence_rate']:6.1%} {acc_p:>10s} {acc_a:>10s}  {meaning_short}")


def print_summary_stats(model_data):
    """Print overall summary statistics."""
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for model, data in model_data.items():
        df = data['df']
        concept_names = data['concept_names']
        log_probs = data['predictions']['log_probs']

        concept_counts = df[concept_names].sum(axis=1)

        print(f"\n--- {model} ---")
        print(f"  Traces: {len(df)}")
        print(f"  Concepts: {len(concept_names)}")
        print(f"  Mean concepts per trace: {concept_counts.mean():.1f} (std={concept_counts.std():.1f})")
        print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
        if 'correct' in df.columns and df['correct'].notna().any():
            n_correct = int((df['correct'] == 1).sum())
            n_incorrect = int((df['correct'] == 0).sum())
            print(f"  Prediction accuracy: {n_correct}/{n_correct + n_incorrect} ({n_correct/(n_correct+n_incorrect):.0%})")

        if 'correct' in df.columns and df['correct'].notna().any():
            median_count = concept_counts.median()
            high = df[concept_counts >= median_count]
            low = df[concept_counts < median_count]
            if len(high) > 0 and len(low) > 0:
                acc_high = high['correct'].mean()
                acc_low = low['correct'].mean()
                print(f"  Accuracy (>= {median_count:.0f} concepts): {acc_high:.0%} (n={len(high)})")
                print(f"  Accuracy (< {median_count:.0f} concepts):  {acc_low:.0%} (n={len(low)})")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze manual concept presence annotations'
    )
    parser.add_argument('--results_dirs', nargs='+', required=True,
                        help='Model results directories (for predictions and model names)')
    parser.add_argument('--annotations_dir', type=str, default=None,
                        help='Directory containing annotation CSVs (default: look inside each results_dir)')
    parser.add_argument('--output_dir', type=str, default='analysis_outputs/annotation_analysis',
                        help='Output directory for plots and tables')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_data = {}

    for results_dir in args.results_dirs:
        csv_path = find_annotation_csv(results_dir, args.annotations_dir)
        if csv_path is None:
            print(f"No annotation CSV found in {results_dir}, skipping.")
            continue

        model_name = extract_model_name(results_dir)
        print(f"\nLoading {model_name}: {csv_path}")

        df, concept_meta, metadata_cols = load_annotation_csv(csv_path)
        concept_names = [c['name'] for c in concept_meta]

        # Load predictions and compute correctness
        trace_indices = df['trace_index'].values.astype(int)
        preds = load_predictions(results_dir, trace_indices)
        df = add_correctness_column(df, preds)

        presence_rates = df[concept_names].mean().values

        model_data[model_name] = {
            'df': df,
            'concept_meta': concept_meta,
            'concept_names': concept_names,
            'predictions': preds,
            'presence_rates': presence_rates,
        }

        print(f"  {len(df)} traces, {len(concept_names)} concepts")
        if preds['log_probs'] is not None:
            print(f"  Log-probs loaded ({len(preds['log_probs'])} values)")

    if not model_data:
        print("No annotation data found.")
        return

    print(f"\nGenerating analyses in {output_dir}...")
    plot_presence_rates(model_data, output_dir)
    plot_concept_heatmap(model_data, output_dir)
    plot_concept_count_vs_accuracy(model_data, output_dir)
    plot_cross_model_summary(model_data, output_dir)
    save_concept_summary(model_data, output_dir)
    print_summary_stats(model_data)

    print(f"\nDone. All outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
