#!/usr/bin/env python
# coding: utf-8

"""
Plotting and visualization functions for lesion overlay experiment results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
import json


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

    return config, df_results, df_summary, df_differences


def plot_log_prob_distribution(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot histogram of log probabilities across all combinations.

    Args:
        df_results: DataFrame with all overlay results
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df_results['log_prob'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Log Probabilities Across All Overlays', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_val = df_results['log_prob'].mean()
    median_val = df_results['log_prob'].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_background_effects(df_summary: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot mean log probabilities by background image with error bars.

    Args:
        df_summary: DataFrame with summary statistics by background
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by mean log prob for better visualization
    df_sorted = df_summary.sort_values('mean_log_prob')

    x_pos = np.arange(len(df_sorted))
    ax.errorbar(x_pos, df_sorted['mean_log_prob'],
                yerr=df_sorted['std_log_prob'],
                fmt='o', capsize=5, capthick=2, markersize=8)

    ax.set_xlabel('Background Image', fontsize=12)
    ax.set_ylabel('Mean Log Probability', fontsize=12)
    ax.set_title('Effect of Background Image on Model Predictions', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['background'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lesion_effects(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot mean log probabilities by lesion image.

    Args:
        df_results: DataFrame with all overlay results
        save_path: Optional path to save the figure
    """
    # Calculate mean and std for each lesion
    lesion_stats = df_results.groupby('lesion')['log_prob'].agg(['mean', 'std']).reset_index()
    lesion_stats = lesion_stats.sort_values('mean')

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(lesion_stats))
    ax.errorbar(x_pos, lesion_stats['mean'],
                yerr=lesion_stats['std'],
                fmt='o', capsize=5, capthick=2, markersize=8, color='coral')

    ax.set_xlabel('Lesion Image', fontsize=12)
    ax.set_ylabel('Mean Log Probability', fontsize=12)
    ax.set_title('Effect of Lesion Image on Model Predictions', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lesion_stats['lesion'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_heatmap(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot heatmap of log probabilities: backgrounds × lesions.

    Args:
        df_results: DataFrame with all overlay results
        save_path: Optional path to save the figure
    """
    # Create pivot table
    pivot_table = df_results.pivot(index='background', columns='lesion', values='log_prob')

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Log Probability'},
                linewidths=0.5, ax=ax)

    ax.set_title('Log Probabilities: Background × Lesion Combinations',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Lesion Image', fontsize=12)
    ax.set_ylabel('Background Image', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_image_differences(df_differences: pd.DataFrame, save_path: Optional[str] = None):
    """
    Compare pixel differences between lesions and backgrounds vs originals.

    Args:
        df_differences: DataFrame with image difference statistics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    lesion_diffs = df_differences[df_differences['image_type'] == 'lesion']
    background_diffs = df_differences[df_differences['image_type'] == 'background']

    # Mean difference comparison
    axes[0].bar(['Lesion', 'Background'],
                [lesion_diffs['mean_diff'].mean(), background_diffs['mean_diff'].mean()],
                color=['coral', 'skyblue'], edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Mean Pixel Difference', fontsize=11)
    axes[0].set_title('Average Difference from Original', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Max difference comparison
    axes[1].bar(['Lesion', 'Background'],
                [lesion_diffs['max_diff'].mean(), background_diffs['max_diff'].mean()],
                color=['coral', 'skyblue'], edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('Max Pixel Difference', fontsize=11)
    axes[1].set_title('Maximum Difference from Original', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Total difference comparison
    axes[2].bar(['Lesion', 'Background'],
                [lesion_diffs['total_diff'].mean(), background_diffs['total_diff'].mean()],
                color=['coral', 'skyblue'], edgecolor='black', alpha=0.7)
    axes[2].set_ylabel('Total Pixel Difference', fontsize=11)
    axes[2].set_title('Total Cumulative Difference', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Pixel-Level Differences: Lesions vs Backgrounds (compared to originals)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_variance_decomposition(df_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Decompose variance in predictions: how much comes from background vs lesion?

    Args:
        df_results: DataFrame with all overlay results
        save_path: Optional path to save the figure
    """
    # Variance by background (averaging over lesions)
    var_by_background = df_results.groupby('background')['log_prob'].var().mean()

    # Variance by lesion (averaging over backgrounds)
    var_by_lesion = df_results.groupby('lesion')['log_prob'].var().mean()

    # Total variance
    total_var = df_results['log_prob'].var()

    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [var_by_background, var_by_lesion]
    labels = [f'Background\nVariance: {var_by_background:.4f}',
              f'Lesion\nVariance: {var_by_lesion:.4f}']
    colors = ['skyblue', 'coral']
    explode = (0.05, 0.05)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12})
    ax.set_title(f'Variance Decomposition\n(Total Variance: {total_var:.4f})',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_example_overlays(results_dir: str, n_examples: int = 6, save_path: Optional[str] = None):
    """
    Display example overlay images with their predictions.

    Args:
        results_dir: Directory containing experiment results
        n_examples: Number of examples to show
        save_path: Optional path to save the figure
    """
    results_path = Path(results_dir)
    df_results = pd.read_csv(results_path / 'all_overlay_results.csv')

    # Sample diverse examples (highest, lowest, median predictions)
    sorted_df = df_results.sort_values('log_prob')
    indices = [
        0,  # Lowest
        len(sorted_df) // 4,  # Q1
        len(sorted_df) // 2,  # Median
        3 * len(sorted_df) // 4,  # Q3
        len(sorted_df) - 1,  # Highest
    ]

    if n_examples < len(indices):
        indices = indices[:n_examples]

    examples = sorted_df.iloc[indices]

    n_cols = min(3, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_examples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    for idx, (_, row) in enumerate(examples.iterrows()):
        if idx >= len(axes):
            break

        try:
            img = Image.open(row['overlay_path']).convert('RGB')
            axes[idx].imshow(img)
            axes[idx].set_title(f"Log Prob: {row['log_prob']:.3f}\n" +
                               f"BG: {row['background'][:20]}\n" +
                               f"Lesion: {row['lesion'][:20]}",
                               fontsize=10)
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error loading image',
                          ha='center', va='center')
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(examples), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Example Overlay Images and Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_all_plots(results_dir: str):
    """
    Generate all visualization plots for the experiment.

    Args:
        results_dir: Directory containing experiment results
    """
    results_path = Path(results_dir)
    plots_dir = results_path / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {results_dir}")
    config, df_results, df_summary, df_differences = load_experiment_results(results_dir)

    print("\nGenerating plots...")

    # 1. Log probability distribution
    print("  1. Log probability distribution...")
    plot_log_prob_distribution(df_results, plots_dir / 'log_prob_distribution.png')

    # 2. Background effects
    print("  2. Background effects...")
    plot_background_effects(df_summary, plots_dir / 'background_effects.png')

    # 3. Lesion effects
    print("  3. Lesion effects...")
    plot_lesion_effects(df_results, plots_dir / 'lesion_effects.png')

    # 4. Heatmap
    print("  4. Heatmap...")
    plot_heatmap(df_results, plots_dir / 'heatmap.png')

    # 5. Image differences
    print("  5. Image differences...")
    plot_image_differences(df_differences, plots_dir / 'image_differences.png')

    # 6. Variance decomposition
    print("  6. Variance decomposition...")
    plot_variance_decomposition(df_results, plots_dir / 'variance_decomposition.png')

    # 7. Example overlays
    print("  7. Example overlays...")
    plot_example_overlays(results_dir, n_examples=6, save_path=plots_dir / 'example_overlays.png')

    print(f"\nAll plots saved to: {plots_dir}")
    print("="*60)


def print_interpretation_guide():
    """Print a guide for interpreting the results."""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LESION OVERLAY EXPERIMENT - INTERPRETATION GUIDE          ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 PLOT DESCRIPTIONS:

1. LOG PROBABILITY DISTRIBUTION
   - Shows the distribution of model predictions across all overlay combinations
   - Interpretation:
     * Higher values → Model predicts "malignant" with higher confidence
     * Lower values → Model predicts "benign" or is uncertain
     * Spread indicates variability in predictions

2. BACKGROUND EFFECTS
   - Shows how different background images affect predictions (averaged over lesions)
   - Interpretation:
     * Each point represents one background image
     * Error bars show variability across different lesions on that background
     * Large variation → Background strongly influences predictions
     * Backgrounds with high means → Make lesions appear more malignant

3. LESION EFFECTS
   - Shows how different lesion images affect predictions (averaged over backgrounds)
   - Interpretation:
     * Each point represents one lesion image
     * Error bars show variability across different backgrounds
     * Large variation → Lesion interacts strongly with background
     * Lesions with high means → Appear more malignant regardless of background

4. HEATMAP (Background × Lesion)
   - Complete matrix of all combinations
   - Interpretation:
     * Each cell = one specific background + lesion combination
     * Rows/columns with similar colors → Consistent effects
     * Checkerboard pattern → Strong interaction effects
     * Look for "hotspots" (red) and "coldspots" (green)

5. IMAGE DIFFERENCES
   - Compares pixel-level changes between lesions/backgrounds vs originals
   - Interpretation:
     * Higher bars → More pixel-level modification from original
     * If lesions differ more than backgrounds → Lesion modification is more severe
     * Helps assess whether background/lesion changes are comparable

6. VARIANCE DECOMPOSITION
   - Shows what drives prediction variability: background or lesion?
   - Interpretation:
     * Larger slice → Greater contribution to prediction variance
     * ~50/50 split → Both equally important
     * Skewed split → One factor dominates model behavior

7. EXAMPLE OVERLAYS
   - Visual examples across the prediction spectrum
   - Interpretation:
     * Low log prob → Model sees as benign
     * High log prob → Model sees as malignant
     * Compare visual appearance with predictions

═══════════════════════════════════════════════════════════════════════════════

🔍 KEY QUESTIONS TO ASK:

1. BACKGROUND SPURIOUS CORRELATION:
   Q: Do certain backgrounds systematically increase/decrease predictions?
   → Check "Background Effects" plot
   → Large differences suggest background bias

2. LESION ROBUSTNESS:
   Q: Are lesion predictions stable across different backgrounds?
   → Check "Lesion Effects" plot - look at error bars
   → Small error bars → Robust predictions

3. INTERACTION EFFECTS:
   Q: Do certain background-lesion pairs create unexpected predictions?
   → Check "Heatmap" for surprising cells
   → Compare with individual effects

4. WHAT DRIVES PREDICTIONS?
   Q: Is the model responding more to background or lesion content?
   → Check "Variance Decomposition"
   → Ideally, lesion should dominate

5. MAGNITUDE OF CHANGES:
   Q: Are we modifying backgrounds/lesions equally?
   → Check "Image Differences"
   → Ensures fair comparison

═══════════════════════════════════════════════════════════════════════════════

⚠️  WARNING SIGNS (Potential Issues):

❌ Background dominates variance decomposition (>60%)
   → Model may rely on spurious background features

❌ Specific backgrounds always produce high/low predictions
   → Systematic background bias detected

❌ Lesions have large error bars in "Lesion Effects"
   → Lesion predictions not robust to background changes

❌ Extreme values in heatmap for specific combinations
   → Unexpected interaction effects

✅ HEALTHY PATTERNS:

✓ Lesion dominates variance decomposition (>60%)
✓ Background error bars are small in "Background Effects"
✓ Lesion effects show clear separation with small error bars
✓ Heatmap shows row/column structure (not checkerboard)

═══════════════════════════════════════════════════════════════════════════════
"""
    print(guide)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lesion_overlay_plots.py <results_dir>")
        print("\nExample:")
        print("  python lesion_overlay_plots.py ./lesion_overlay_results")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Generate all plots
    generate_all_plots(results_dir)

    # Print interpretation guide
    print("\n")
    print_interpretation_guide()
