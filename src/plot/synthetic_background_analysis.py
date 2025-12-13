#!/usr/bin/env python
# coding: utf-8

"""
Analysis and visualization for synthetic background experiment results.
Compares lesion predictions across different synthetically generated background groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
import json
from scipy import stats


def parse_group_from_filename(filename: str) -> str:
    """
    Extract group name from filename (first word before underscore or space).

    Args:
        filename: Image filename

    Returns:
        Group name (e.g., 'ear', 'abdomen', 'scalp')
    """
    # Remove extension
    stem = Path(filename).stem
    # Get first word (split on underscore or space)
    group = stem.split('_')[0].split()[0].lower()
    return group


def load_experiment_results(results_dir: str) -> tuple:
    """
    Load experiment results from saved files.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        Tuple of (config, df_results, df_summary, df_differences)
    """
    results_path = Path(results_dir)

    # Load config
    with open(results_path / 'experiment_config.json', 'r') as f:
        config = json.load(f)

    # Load results
    df_results = pd.read_csv(results_path / 'all_overlay_results.csv')
    df_summary = pd.read_csv(results_path / 'summary_by_background.csv')
    df_differences = pd.read_csv(results_path / 'image_differences.csv')

    # Add group column based on background filename
    df_results['group'] = df_results['background'].apply(parse_group_from_filename)

    # Filter to only lesion differences (no background comparison)
    df_differences = df_differences[df_differences['image_type'] == 'lesion']

    return config, df_results, df_summary, df_differences


def plot_group_comparison_violin(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Violin plot comparing log probability distributions across background groups.

    Args:
        df_results: DataFrame with all overlay results (must have 'group' column)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique groups sorted by median
    group_medians = df_results.groupby('group')['log_prob'].median().sort_values()
    groups_sorted = group_medians.index.tolist()

    # Create violin plot
    parts = ax.violinplot(
        [df_results[df_results['group'] == g]['log_prob'].values for g in groups_sorted],
        positions=range(len(groups_sorted)),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )

    # Customize violin colors
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    ax.set_xlabel('Background Group', fontsize=12)
    ax.set_ylabel('Log Probability', fontsize=12)
    ax.set_title('Distribution of Lesion Predictions Across Background Groups',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(groups_sorted)))
    ax.set_xticklabels(groups_sorted, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_group_comparison_boxplot(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Boxplot comparing log probabilities across background groups.

    Args:
        df_results: DataFrame with all overlay results (must have 'group' column)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique groups sorted by median
    group_medians = df_results.groupby('group')['log_prob'].median().sort_values()
    groups_sorted = group_medians.index.tolist()

    # Create boxplot
    bp = ax.boxplot(
        [df_results[df_results['group'] == g]['log_prob'].values for g in groups_sorted],
        labels=groups_sorted,
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
    )

    # Customize box colors
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.6)

    ax.set_xlabel('Background Group', fontsize=12)
    ax.set_ylabel('Log Probability', fontsize=12)
    ax.set_title('Lesion Prediction Distributions by Background Group',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(groups_sorted, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_group_statistics(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Bar plot showing mean and std for each background group.

    Args:
        df_results: DataFrame with all overlay results (must have 'group' column)
        save_path: Optional path to save the figure
    """
    # Calculate statistics by group
    group_stats = df_results.groupby('group')['log_prob'].agg(['mean', 'std', 'count']).reset_index()
    group_stats = group_stats.sort_values('mean')

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(group_stats))
    ax.errorbar(x_pos, group_stats['mean'],
                yerr=group_stats['std'],
                fmt='o', capsize=5, capthick=2, markersize=10, color='darkgreen',
                ecolor='darkgreen', elinewidth=2)

    # Add count annotations
    for i, (_, row) in enumerate(group_stats.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.05,
               f"n={int(row['count'])}",
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Background Group', fontsize=12)
    ax.set_ylabel('Mean Log Probability', fontsize=12)
    ax.set_title('Mean Lesion Predictions by Background Group (with std error bars)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_stats['group'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pairwise_group_comparison(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Heatmap showing statistical significance of pairwise group comparisons.

    Args:
        df_results: DataFrame with all overlay results (must have 'group' column)
        save_path: Optional path to save the figure
    """
    groups = sorted(df_results['group'].unique())
    n_groups = len(groups)

    # Initialize matrices
    p_values = np.ones((n_groups, n_groups))
    mean_diff = np.zeros((n_groups, n_groups))

    # Perform pairwise t-tests
    for i, group1 in enumerate(groups):
        data1 = df_results[df_results['group'] == group1]['log_prob'].values
        for j, group2 in enumerate(groups):
            if i != j:
                data2 = df_results[df_results['group'] == group2]['log_prob'].values
                t_stat, p_val = stats.ttest_ind(data1, data2)
                p_values[i, j] = p_val
                mean_diff[i, j] = data1.mean() - data2.mean()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: P-values heatmap
    sns.heatmap(-np.log10(p_values), annot=p_values, fmt='.4f', cmap='RdYlGn_r',
                xticklabels=groups, yticklabels=groups, ax=ax1,
                cbar_kws={'label': '-log10(p-value)'}, linewidths=0.5)
    ax1.set_title('Pairwise T-Test P-Values\n(darker = more significant)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Group', fontsize=11)
    ax1.set_ylabel('Group', fontsize=11)

    # Plot 2: Mean differences heatmap
    sns.heatmap(mean_diff, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=groups, yticklabels=groups, ax=ax2,
                cbar_kws={'label': 'Mean Difference'}, linewidths=0.5)
    ax2.set_title('Pairwise Mean Differences\n(row - column)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Group', fontsize=11)
    ax2.set_ylabel('Group', fontsize=11)

    plt.suptitle('Statistical Comparison Between Background Groups',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lesion_consistency_across_groups(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot showing how consistent each lesion's predictions are across different background groups.

    Args:
        df_results: DataFrame with all overlay results
        save_path: Optional path to save the figure
    """
    # Calculate variance for each lesion across groups
    lesion_variance = df_results.groupby('lesion')['log_prob'].var().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(lesion_variance))
    bars = ax.bar(x_pos, lesion_variance.values, color='coral', edgecolor='black', alpha=0.7)

    # Highlight high-variance lesions
    threshold = lesion_variance.median()
    for i, (bar, var) in enumerate(zip(bars, lesion_variance.values)):
        if var > threshold:
            bar.set_color('darkred')

    ax.axhline(threshold, color='blue', linestyle='--', linewidth=2,
               label=f'Median Variance: {threshold:.4f}')

    ax.set_xlabel('Lesion Image', fontsize=12)
    ax.set_ylabel('Prediction Variance Across Groups', fontsize=12)
    ax.set_title('Lesion Prediction Consistency\n(lower = more consistent across background groups)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lesion_variance.index, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_group_heatmap(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Heatmap showing log probabilities: background groups × lesions.

    Args:
        df_results: DataFrame with all overlay results
        save_path: Optional path to save the figure
    """
    # Create pivot table with mean log prob for each group-lesion combination
    pivot_table = df_results.groupby(['group', 'lesion'])['log_prob'].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Mean Log Probability'},
                linewidths=0.5, ax=ax, cbar=True)

    ax.set_title('Mean Log Probabilities: Background Group × Lesion',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Lesion Image', fontsize=12)
    ax.set_ylabel('Background Group', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lesion_differences(df_differences: pd.DataFrame, df_results: pd.DataFrame,
                           save_path: Optional[str] = None):
    """
    Plot pixel-level differences for lesions across different groups.

    Args:
        df_differences: DataFrame with image difference statistics (lesions only)
        df_results: DataFrame with all overlay results (for group info)
        save_path: Optional path to save the figure
    """
    # Merge to get group information
    df_diff_with_group = df_differences.merge(
        df_results[['background', 'lesion', 'group']].drop_duplicates(),
        on=['background', 'lesion'],
        how='left'
    )

    # Group by background group
    group_stats = df_diff_with_group.groupby('group')[['mean_diff', 'max_diff', 'total_diff']].mean()
    group_stats = group_stats.sort_values('mean_diff')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mean difference
    axes[0].bar(range(len(group_stats)), group_stats['mean_diff'].values,
                color='coral', edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Mean Pixel Difference', fontsize=11)
    axes[0].set_title('Avg Lesion Difference from Original', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(group_stats)))
    axes[0].set_xticklabels(group_stats.index, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Max difference
    axes[1].bar(range(len(group_stats)), group_stats['max_diff'].values,
                color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('Max Pixel Difference', fontsize=11)
    axes[1].set_title('Max Lesion Difference from Original', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(group_stats)))
    axes[1].set_xticklabels(group_stats.index, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Total difference
    axes[2].bar(range(len(group_stats)), group_stats['total_diff'].values,
                color='coral', edgecolor='black', alpha=0.7)
    axes[2].set_ylabel('Total Pixel Difference', fontsize=11)
    axes[2].set_title('Total Lesion Difference', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(group_stats)))
    axes[2].set_xticklabels(group_stats.index, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Pixel-Level Lesion Modifications Across Groups',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_sample_size_by_group(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Bar plot showing number of samples per background group.

    Args:
        df_results: DataFrame with all overlay results
        save_path: Optional path to save the figure
    """
    group_counts = df_results['group'].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(group_counts)), group_counts.values,
                  color='steelblue', edgecolor='black', alpha=0.7)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, group_counts.values)):
        ax.text(i, count + max(group_counts) * 0.01, str(count),
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Background Group', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Sample Size by Background Group', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(group_counts)))
    ax.set_xticklabels(group_counts.index, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_summary_statistics(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics for each background group.

    Args:
        df_results: DataFrame with all overlay results

    Returns:
        DataFrame with summary statistics
    """
    stats_list = []

    for group in sorted(df_results['group'].unique()):
        group_data = df_results[df_results['group'] == group]['log_prob']

        stats_list.append({
            'group': group,
            'count': len(group_data),
            'mean': group_data.mean(),
            'std': group_data.std(),
            'median': group_data.median(),
            'min': group_data.min(),
            'max': group_data.max(),
            'q25': group_data.quantile(0.25),
            'q75': group_data.quantile(0.75),
            'iqr': group_data.quantile(0.75) - group_data.quantile(0.25)
        })

    return pd.DataFrame(stats_list)


def generate_all_plots(results_dir: str):
    """
    Generate all visualization plots for the synthetic background experiment.

    Args:
        results_dir: Directory containing experiment results
    """
    results_path = Path(results_dir)
    plots_dir = results_path / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {results_dir}")
    config, df_results, df_summary, df_differences = load_experiment_results(results_dir)

    print(f"\nFound {len(df_results['group'].unique())} background groups:")
    for group in sorted(df_results['group'].unique()):
        count = len(df_results[df_results['group'] == group])
        print(f"  - {group}: {count} samples")

    print("\nGenerating plots...")

    # 1. Sample size by group
    print("  1. Sample size by group...")
    plot_sample_size_by_group(df_results, plots_dir / 'sample_size_by_group.png')

    # 2. Group comparison - violin plot
    print("  2. Group comparison (violin plot)...")
    plot_group_comparison_violin(df_results, plots_dir / 'group_comparison_violin.png')

    # 3. Group comparison - boxplot
    print("  3. Group comparison (boxplot)...")
    plot_group_comparison_boxplot(df_results, plots_dir / 'group_comparison_boxplot.png')

    # 4. Group statistics
    print("  4. Group statistics...")
    plot_group_statistics(df_results, plots_dir / 'group_statistics.png')

    # 5. Pairwise group comparison
    print("  5. Pairwise statistical comparisons...")
    plot_pairwise_group_comparison(df_results, plots_dir / 'pairwise_comparison.png')

    # 6. Group heatmap
    print("  6. Group × Lesion heatmap...")
    plot_group_heatmap(df_results, plots_dir / 'group_lesion_heatmap.png')

    # 7. Lesion consistency
    print("  7. Lesion consistency across groups...")
    plot_lesion_consistency_across_groups(df_results, plots_dir / 'lesion_consistency.png')

    # 8. Lesion differences
    print("  8. Lesion pixel differences...")
    plot_lesion_differences(df_differences, df_results, plots_dir / 'lesion_differences_by_group.png')

    # Generate and save summary statistics
    print("  9. Summary statistics table...")
    summary_stats = generate_summary_statistics(df_results)
    summary_stats.to_csv(plots_dir / 'group_summary_statistics.csv', index=False)
    print(f"    Saved: {plots_dir / 'group_summary_statistics.csv'}")

    print(f"\nAll plots saved to: {plots_dir}")
    print("\nSummary Statistics:")
    print(summary_stats.to_string(index=False))
    print("="*80)


def print_interpretation_guide():
    """Print a guide for interpreting the results."""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║          SYNTHETIC BACKGROUND EXPERIMENT - INTERPRETATION GUIDE              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 PLOT DESCRIPTIONS:

1. SAMPLE SIZE BY GROUP
   - Shows number of synthetic backgrounds generated for each group
   - Interpretation:
     * Ensures balanced sampling across groups
     * Unequal sizes may affect statistical power

2. GROUP COMPARISON (VIOLIN & BOXPLOT)
   - Shows distribution of lesion predictions for each background group
   - Interpretation:
     * Violin width → Distribution spread
     * Boxplot notches → 95% confidence interval for median
     * Non-overlapping notches → Significantly different medians
     * Compare shapes: symmetric vs skewed distributions

3. GROUP STATISTICS
   - Mean predictions with error bars (std) for each group
   - Interpretation:
     * Higher mean → Lesions appear more malignant on this background type
     * Large error bars → High variability within group
     * Sample size (n) shown above each point

4. PAIRWISE STATISTICAL COMPARISONS
   - Left: P-values from t-tests (darker = more significant)
   - Right: Mean differences between groups
   - Interpretation:
     * P-value < 0.05 → Statistically significant difference
     * Mean difference magnitude → Clinical significance
     * Look for systematic patterns

5. GROUP × LESION HEATMAP
   - Shows mean prediction for each group-lesion combination
   - Interpretation:
     * Each cell = average across all backgrounds in that group
     * Row patterns → Group effects
     * Column patterns → Lesion effects
     * Outlier cells → Specific interactions

6. LESION CONSISTENCY
   - Variance of each lesion's predictions across groups
   - Interpretation:
     * Low variance → Robust lesion, similar prediction regardless of background
     * High variance → Background-dependent lesion
     * Red bars (above median) → Inconsistent lesions

7. LESION PIXEL DIFFERENCES BY GROUP
   - Pixel-level modifications to lesions across groups
   - Interpretation:
     * Shows magnitude of editing applied
     * Should be similar across groups for fair comparison
     * Large differences may confound results

═══════════════════════════════════════════════════════════════════════════════

🔍 KEY QUESTIONS TO ASK:

1. GROUP EFFECTS:
   Q: Do certain background groups systematically affect lesion predictions?
   → Check violin/boxplot and statistics plots
   → Look for clear separation between groups

2. STATISTICAL SIGNIFICANCE:
   Q: Are group differences statistically meaningful?
   → Check pairwise comparison heatmap
   → P-values < 0.05 indicate significant differences

3. LESION ROBUSTNESS:
   Q: Are some lesions more sensitive to background type?
   → Check lesion consistency plot
   → High-variance lesions are problematic

4. INTERACTION EFFECTS:
   Q: Do certain lesion-group combinations behave unexpectedly?
   → Check group × lesion heatmap
   → Look for outlier cells

5. FAIRNESS OF COMPARISON:
   Q: Are all groups edited similarly?
   → Check lesion differences plot
   → Groups should have similar pixel modifications

═══════════════════════════════════════════════════════════════════════════════

⚠️  WARNING SIGNS (Potential Issues):

❌ Large mean differences between groups (>0.5 log prob)
   → Indicates strong background bias

❌ Many significant p-values in pairwise comparisons
   → Systematic group effects detected

❌ High lesion variance across groups
   → Predictions not robust to background changes

❌ Unequal pixel modifications across groups
   → Unfair comparison, confounding variable

❌ Skewed sample sizes across groups
   → May need to balance or use weighted statistics

✅ HEALTHY PATTERNS:

✓ Overlapping violin plots/boxplots across groups
✓ Few significant pairwise differences (p > 0.05)
✓ Low lesion variance in consistency plot
✓ Similar pixel modifications across groups
✓ Heatmap shows column structure (lesion effects) not row structure (group effects)

═══════════════════════════════════════════════════════════════════════════════

💡 EXPECTED FINDINGS:

If model is NOT biased by background:
  - Group means should be similar
  - Pairwise comparisons mostly non-significant
  - Lesion consistency should be high (low variance)
  - Heatmap should show lesion patterns, not group patterns

If model IS biased by background:
  - Clear separation in group distributions
  - Significant pairwise differences
  - Some lesions highly variable across groups
  - Heatmap shows strong row (group) effects

═══════════════════════════════════════════════════════════════════════════════
"""
    print(guide)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python synthetic_background_analysis.py <results_dir>")
        print("\nExample:")
        print("  python synthetic_background_analysis.py ./synthetic_background_results")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Generate all plots
    generate_all_plots(results_dir)

    # Print interpretation guide
    print("\n")
    print_interpretation_guide()
