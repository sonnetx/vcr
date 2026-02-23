"""
Omitted Features Analysis Script

Identifies features (concepts) that are important to the model (high sensitivity scores from VCR)
but tend to be left out of reasoning traces. Provides validation via top-activating images.

Key insight: VCR identifies concepts that influence predictions via gradient-based sensitivity,
while concept presence analysis tracks which concepts the model mentions in reasoning.
Features with high sensitivity but low mention rate are "omitted" - potentially important but not verbalized.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


def load_experiment_config(results_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration from experiment_config.json"""
    config_path = results_dir / "experiment_config.json"
    with open(config_path, "r") as f:
        full_config = json.load(f)
    return full_config["config"]


def load_sensitivity_data(results_dir: Path, seed: int = 0) -> Tuple[np.ndarray, List[str]]:
    """
    Load sensitivity data from VCR results.

    Args:
        results_dir: Directory containing VCR experiment results
        seed: Seed folder to load from (default: 0)

    Returns:
        Tuple of (weighted_sens array [n_samples, n_concepts], concept_texts list)
    """
    seed_dir = results_dir / f"seed_{seed}"

    # Load weighted sensitivity
    weighted_sens_path = seed_dir / "weighted_sens.npy"
    weighted_sens = np.load(weighted_sens_path)

    # Load concept texts
    concept_texts_path = seed_dir / "concept_texts.json"
    with open(concept_texts_path, "r") as f:
        concept_texts = json.load(f)

    return weighted_sens, concept_texts


def load_similarity_matrix(results_dir: Path, seed: int = 0) -> np.ndarray:
    """
    Load similarity matrix from VCR results.

    Args:
        results_dir: Directory containing VCR experiment results
        seed: Seed folder to load from (default: 0)

    Returns:
        Similarity matrix [n_concepts, n_images]
    """
    sim_matrix_path = results_dir / f"seed_{seed}" / "similarity_matrix.npy"
    return np.load(sim_matrix_path)


def find_latest_presence_json(presence_dir: Path = Path("results/concept_presence_analysis")) -> Optional[Path]:
    """
    Find the most recently modified JSON file in the presence analysis directory.

    Args:
        presence_dir: Directory to search for JSON files

    Returns:
        Path to the latest JSON file, or None if no JSON files found
    """
    if not presence_dir.exists():
        return None

    json_files = list(presence_dir.glob("*.json"))
    if not json_files:
        return None

    # Sort by modification time, most recent first
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files[0]


def load_presence_analysis(json_path: Path) -> Dict[str, Any]:
    """
    Load concept presence analysis JSON output.

    Args:
        json_path: Path to concept presence analysis JSON file

    Returns:
        Dict with summary_statistics.per_concept containing presence rates
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_image_paths(config: Dict[str, Any], seed: int = 0) -> List[Path]:
    """
    Prepare train image paths based on config.
    Must match the DDIDataLoader logic from the experiment script.

    Args:
        config: Experiment configuration
        seed: Random seed to use (default 0 for seed_0 folder similarity matrix)

    Returns:
        List of image paths in the order matching similarity matrix
    """
    metadata_path = config['metadata_path']
    base_dir = config['ddi_base_dir']
    test_size = config['test_size']
    demo_size = config['demo_size']
    use_demos = config['prompt'].get('use_demos', False)

    # Load metadata
    df = pd.read_csv(metadata_path, index_col=0)

    # First split: train/test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed
    )

    # Second split: demos (if needed)
    if use_demos:
        train_df, demo_df = train_test_split(
            train_df,
            test_size=demo_size,
            random_state=seed
        )

    # Create probe paths (train set after demo split)
    probe_paths = [Path(base_dir) / file for file in train_df.DDI_file]

    return probe_paths


def compute_omission_metrics(
    weighted_sens: np.ndarray,
    concept_texts: List[str],
    presence_data: Dict[str, Any],
    top_k: Optional[int] = None,
    use_presence_concepts_only: bool = True
) -> pd.DataFrame:
    """
    Compute omission metrics by merging sensitivity and presence data.

    Args:
        weighted_sens: Sensitivity array [n_samples, n_concepts]
        concept_texts: List of concept names
        presence_data: Concept presence analysis output
        top_k: Number of top concepts by sensitivity to analyze. If None and
               use_presence_concepts_only=True, uses all concepts from presence_data.
        use_presence_concepts_only: If True, only analyze concepts that exist in
               the presence_data (ensures fair comparison across analyses with
               different top-K values). Default: True.

    Returns:
        DataFrame with columns: concept, sens_mean, sens_rank, presence_rate, omission_rate, omission_score
    """
    # Compute mean absolute sensitivity per concept
    sens_mean = np.mean(np.abs(weighted_sens), axis=0)

    # Get sensitivity ranks (1 = highest sensitivity)
    sens_ranks_array = np.argsort(-sens_mean)
    sens_ranks = np.zeros(len(concept_texts), dtype=int)
    for rank, idx in enumerate(sens_ranks_array):
        sens_ranks[idx] = rank + 1  # 1-indexed

    # Get concepts from presence data
    per_concept_stats = presence_data.get('summary_statistics', {}).get('per_concept', {})
    presence_concepts = set(per_concept_stats.keys())

    # Also check the 'concepts' list in presence_data (contains original top concepts)
    if 'concepts' in presence_data:
        for c in presence_data['concepts']:
            if isinstance(c, dict):
                presence_concepts.add(c.get('concept', ''))
            else:
                presence_concepts.add(c)

    # Build dataframe
    records = []
    concept_to_idx = {c: i for i, c in enumerate(concept_texts)}

    for idx, concept in enumerate(concept_texts):
        rank = int(sens_ranks[idx])
        presence_rate = 0.0
        in_presence_data = False

        # Try to find presence rate (case-insensitive matching)
        if concept in per_concept_stats:
            presence_rate = per_concept_stats[concept].get('presence_rate_reasoning', 0.0)
            in_presence_data = True
        else:
            # Try case-insensitive match
            for key, stats in per_concept_stats.items():
                if key.lower() == concept.lower():
                    presence_rate = stats.get('presence_rate_reasoning', 0.0)
                    in_presence_data = True
                    break

        # Skip if using presence concepts only and this concept isn't in presence data
        if use_presence_concepts_only and not in_presence_data:
            continue

        records.append({
            'concept': concept,
            'concept_idx': idx,
            'sens_mean': sens_mean[idx],
            'sens_rank': rank,
            'presence_rate': presence_rate,
            'omission_rate': 1.0 - presence_rate,
            'in_presence_data': in_presence_data,
        })

    df = pd.DataFrame(records)

    # Apply top-K filter if specified
    if top_k is not None and len(df) > 0:
        df = df[df['sens_rank'] <= top_k].copy()

    if len(df) == 0:
        print("WARNING: No concepts found after filtering. Check that presence_data contains matching concepts.")
        return df

    # Compute omission score: sensitivity_percentile * omission_rate
    # Higher score = more important AND more omitted
    # Use the rank within the filtered set for fair comparison
    n_concepts = len(concept_texts)
    df['sensitivity_percentile'] = 1.0 - (df['sens_rank'] / n_concepts)
    df['omission_score'] = df['sensitivity_percentile'] * df['omission_rate']

    # Sort by omission score (highest = most concerning)
    df = df.sort_values('omission_score', ascending=False).reset_index(drop=True)

    return df


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """Center crop image to a square using the smaller dimension"""
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def create_omitted_concept_visualization(
    concept_idx: int,
    concept_name: str,
    omission_rate: float,
    sens_rank: int,
    sim_matrix: np.ndarray,
    image_paths: List[Path],
    n_images: int,
    save_path: Path
) -> None:
    """
    Create visualization showing top/bottom activating images for a concept.
    Follows the pattern from pval_plots.py.

    Args:
        concept_idx: Index of concept in similarity matrix
        concept_name: Name of the concept
        omission_rate: Omission rate for title
        sens_rank: Sensitivity rank for title
        sim_matrix: Similarity matrix [n_concepts, n_images]
        image_paths: List of image paths
        n_images: Number of images per row
        save_path: Path to save the visualization
    """
    # Get activation scores for this concept
    concept_activations = sim_matrix[concept_idx, :]

    # Get indices sorted by activation
    sorted_indices = np.argsort(-concept_activations)
    top_indices = sorted_indices[:n_images]
    bottom_indices = sorted_indices[-n_images:]

    # Create figure
    fig, axes = plt.subplots(2, n_images, figsize=(16, 10))
    fig.suptitle(
        f'Concept: {concept_name}\n'
        f'Sensitivity Rank: {sens_rank} | Omission Rate: {omission_rate:.1%}',
        fontsize=14, fontweight='bold', y=0.78
    )

    # Ensure axes is 2D
    if n_images == 1:
        axes = axes.reshape(2, 1)

    # Top activated images
    for i, img_idx in enumerate(top_indices):
        try:
            img_path = image_paths[img_idx]
            img = Image.open(img_path).convert('RGB')
            img_cropped = center_crop_to_square(img)
            axes[0, i].imshow(img_cropped)
            axes[0, i].axis('off')
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f'Error loading\nimage',
                          ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].axis('off')

    # Bottom activated images
    for i, img_idx in enumerate(bottom_indices):
        try:
            img_path = image_paths[img_idx]
            img = Image.open(img_path).convert('RGB')
            img_cropped = center_crop_to_square(img)
            axes[1, i].imshow(img_cropped)
            axes[1, i].axis('off')
        except Exception as e:
            axes[1, i].text(0.5, 0.5, f'Error loading\nimage',
                          ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')

    # Add row labels
    fig.text(0.06, 0.60, 'Most Activating\nImages', ha='center', va='center',
             fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.06, 0.35, 'Least Activating\nImages', ha='center', va='center',
             fontsize=12, fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, top=0.88, bottom=0.05, hspace=-0.5)

    # Save with 300 DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_omission_scatter_plot(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create scatter plot: sensitivity rank vs presence rate.
    Highlights the "omission zone" - high sensitivity, low presence.

    Args:
        df: DataFrame with sens_rank, presence_rate, omission_score columns
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        df['sens_rank'],
        df['presence_rate'],
        c=df['omission_score'],
        cmap='RdYlGn',  # Green = low omission, Red = high omission
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Omission Score', fontsize=11)

    # Highlight omission zone (top-left: low rank = high sensitivity, low presence)
    max_rank = df['sens_rank'].max()
    ax.axvspan(0, max_rank * 0.3, alpha=0.05, color='red', label='High Sensitivity Zone')
    ax.axhspan(0, 0.3, alpha=0.05, color='blue', label='Low Presence Zone')

    # Annotate top omitted concepts
    top_omitted = df.nlargest(10, 'omission_score')
    for _, row in top_omitted.iterrows():
        ax.annotate(
            row['concept'],
            (row['sens_rank'], row['presence_rate']),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords='offset points'
        )

    ax.set_xlabel('Sensitivity Rank (1 = Most Important)', fontsize=12)
    ax.set_ylabel('Presence Rate in Reasoning', fontsize=12)
    ax.set_title('Concept Sensitivity vs Presence in Reasoning\n(Top-left = Important but Omitted)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0, max_rank + 5)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot: {output_path}")


def create_omission_ranking_chart(df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """
    Create horizontal bar chart showing top N most-omitted concepts.

    Args:
        df: DataFrame with concept, omission_score columns (should be sorted)
        output_path: Path to save the plot
        top_n: Number of concepts to show
    """
    top_omitted = df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

    y_pos = np.arange(len(top_omitted))
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(top_omitted)))

    bars = ax.barh(y_pos, top_omitted['omission_score'], color=colors, alpha=0.85)

    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, top_omitted.itertuples())):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{row.omission_score:.3f} (rank {row.sens_rank}, {row.omission_rate:.0%} omitted)',
            va='center', fontsize=9
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_omitted['concept'])
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Omission Score (Sensitivity Percentile × Omission Rate)', fontsize=11)
    ax.set_title(f'Top {top_n} Most-Omitted High-Sensitivity Concepts', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(top_omitted['omission_score']) * 1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ranking chart: {output_path}")


def generate_validation_report(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    image_paths: List[Path],
    output_dir: Path,
    n_images: int = 7,
    top_n_visualize: int = 20
) -> Dict[str, Any]:
    """
    Generate comprehensive validation report with visualizations.

    Args:
        df: DataFrame with omission metrics (sorted by omission_score)
        sim_matrix: Similarity matrix for finding top-activating images
        image_paths: List of image paths
        output_dir: Output directory
        n_images: Number of images per concept visualization
        top_n_visualize: Number of top concepts to visualize

    Returns:
        Dict with report data for JSON output
    """
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations for top omitted concepts
    print(f"\nGenerating visualizations for top {top_n_visualize} omitted concepts...")
    top_omitted = df.head(top_n_visualize)

    visualization_paths = []
    for idx, row in enumerate(top_omitted.itertuples(), 1):
        # Create safe filename
        safe_name = "".join(c for c in row.concept if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name[:40]
        save_path = viz_dir / f"omitted_{idx:02d}_{safe_name}.png"

        create_omitted_concept_visualization(
            concept_idx=row.concept_idx,
            concept_name=row.concept,
            omission_rate=row.omission_rate,
            sens_rank=row.sens_rank,
            sim_matrix=sim_matrix,
            image_paths=image_paths,
            n_images=n_images,
            save_path=save_path
        )

        visualization_paths.append(str(save_path.relative_to(output_dir)))
        print(f"  [{idx}/{top_n_visualize}] {row.concept} - omission rate: {row.omission_rate:.1%}")

    # Create summary plots
    print("\nGenerating summary plots...")
    create_omission_scatter_plot(df, output_dir / "omission_scatter_plot.png")
    create_omission_ranking_chart(df, output_dir / "omission_ranking_chart.png", top_n=min(20, len(df)))

    # Build report data
    report_data = {
        'top_omitted_concepts': [],
        'visualization_paths': visualization_paths
    }

    for idx, row in enumerate(df.itertuples(), 1):
        # Get top activating image paths for this concept
        concept_activations = sim_matrix[row.concept_idx, :]
        sorted_indices = np.argsort(-concept_activations)
        top_img_indices = sorted_indices[:n_images]

        top_images = [
            {
                'path': str(image_paths[i]),
                'similarity': float(concept_activations[i])
            }
            for i in top_img_indices
        ]

        report_data['top_omitted_concepts'].append({
            'rank': idx,
            'concept': row.concept,
            'concept_idx': row.concept_idx,
            'sensitivity_mean': float(row.sens_mean),
            'sensitivity_rank': int(row.sens_rank),
            'presence_rate': float(row.presence_rate),
            'omission_rate': float(row.omission_rate),
            'omission_score': float(row.omission_score),
            'top_activating_images': top_images
        })

    return report_data


def generate_markdown_report(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20
) -> None:
    """
    Generate markdown report for manual validation.

    Args:
        df: DataFrame with omission metrics
        output_path: Path to save markdown file
        top_n: Number of concepts to include in detailed section
    """
    lines = [
        "# Omitted Features Analysis Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total concepts analyzed**: {len(df)}",
        f"- **Concepts with >50% omission rate**: {(df['omission_rate'] > 0.5).sum()}",
        f"- **Concepts with >70% omission rate**: {(df['omission_rate'] > 0.7).sum()}",
        f"- **Average omission rate**: {df['omission_rate'].mean():.1%}",
        f"- **Average presence rate**: {df['presence_rate'].mean():.1%}",
        "",
        "## Interpretation",
        "",
        "**Omission Score** = Sensitivity Percentile × Omission Rate",
        "",
        "High omission score indicates a concept that is:",
        "1. Important to the model (high sensitivity rank)",
        "2. Rarely mentioned in reasoning (high omission rate)",
        "",
        "These concepts may represent implicit features the model uses but doesn't verbalize.",
        "",
        f"## Top {top_n} Most-Omitted Features",
        "",
    ]

    top_omitted = df.head(top_n)
    for idx, row in enumerate(top_omitted.itertuples(), 1):
        lines.extend([
            f"### {idx}. {row.concept}",
            "",
            f"- **Sensitivity Rank**: {row.sens_rank} (top {row.sens_rank/len(df)*100:.1f}%)",
            f"- **Presence in Reasoning**: {row.presence_rate:.1%}",
            f"- **Omission Rate**: {row.omission_rate:.1%}",
            f"- **Omission Score**: {row.omission_score:.4f}",
            f"- **Validation**: See `visualizations/omitted_{idx:02d}_*.png`",
            "",
        ])

    lines.extend([
        "## Manual Validation Checklist",
        "",
        "For each omitted concept, verify:",
        "",
        "- [ ] **Label accuracy**: Do top-activating images actually show this feature?",
        "- [ ] **Implicit usage**: Is the feature mentioned via synonyms or related terms?",
        "- [ ] **Interpretability impact**: Is this omission meaningful for understanding model behavior?",
        "",
        "## All Analyzed Concepts",
        "",
        "| Rank | Concept | Sens Rank | Presence | Omission | Score |",
        "|------|---------|-----------|----------|----------|-------|",
    ])

    for idx, row in enumerate(df.itertuples(), 1):
        lines.append(
            f"| {idx} | {row.concept} | {row.sens_rank} | {row.presence_rate:.1%} | {row.omission_rate:.1%} | {row.omission_score:.4f} |"
        )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Saved markdown report: {output_path}")


def load_concepts_from_pval_csvs(results_dir: Path) -> List[Dict[str, Any]]:
    """Load significant concepts from pval_plots.py output CSVs."""
    analysis_dir = results_dir / "analysis_outputs"
    concepts = []

    for direction in ['positive', 'negative']:
        csv_files = list(analysis_dir.glob(f"top_*_{direction}_concepts.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            for _, row in df.iterrows():
                concepts.append({
                    'concept': row['concept'],
                    'mean_dd': row['mean_dd'],
                    'p_value': row['p_value'],
                    'direction': direction,
                })

    return concepts


def load_presence_from_annotations(csv_path: Path, concept_texts: List[str]) -> Dict[str, float]:
    """Load presence rates from human annotation CSV.

    Parses column headers (strips <MEANING>/<NEGATIVE> tags), computes
    per-concept presence rate (fraction of traces where concept=1).
    Returns {concept_name: presence_rate}.
    """
    import re
    df = pd.read_csv(csv_path)
    metadata_cols = ['trace_index', 'image_path', 'label', 'reasoning']
    presence_rates = {}

    for col in df.columns:
        if col in metadata_cols:
            continue
        # Parse concept name from header (strip tags)
        name = re.split(r'<', col)[0].strip()
        values = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Check if column was actually annotated (not all empty in raw)
        raw_values = df[col].astype(str).str.strip()
        if (raw_values == '').all() or (raw_values == 'nan').all():
            continue
        presence_rates[name] = float(values.mean())

    return presence_rates


def main():
    parser = argparse.ArgumentParser(
        description='Analyze omitted features - concepts with high sensitivity but low mention rate in reasoning'
    )

    # Required inputs
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory with VCR experiment results (weighted_sens.npy, similarity_matrix.npy, etc.)'
    )

    # Presence data sources (optional — if neither provided, uses pval CSVs for concept list only)
    parser.add_argument(
        '--presence_json',
        type=str,
        default=None,
        help='Path to concept presence analysis JSON output (from LLM judge)'
    )
    parser.add_argument(
        '--annotations_csv',
        type=str,
        default=None,
        help='Path to human annotation CSV (from create_annotation_templates.py, filled in)'
    )

    # Optional parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/omitted_features',
        help='Output directory for analysis results (default: results/omitted_features)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Analyze top K concepts by sensitivity (default: from pval CSVs or presence data)'
    )
    parser.add_argument(
        '--n_images',
        type=int,
        default=7,
        help='Number of top-activating images per concept visualization (default: 7)'
    )
    parser.add_argument(
        '--n_visualize',
        type=int,
        default=20,
        help='Number of top concepts to visualize (default: 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed folder to use for loading data (default: 0)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Omitted Features Analysis")
    print("=" * 80)
    print(f"VCR Results: {results_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Images per visualization: {args.n_images}")
    print(f"Concepts to visualize: {args.n_visualize}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("1. Loading experiment configuration...")
    config = load_experiment_config(results_dir)

    print("2. Loading sensitivity data...")
    weighted_sens, concept_texts = load_sensitivity_data(results_dir, seed=args.seed)
    print(f"   Loaded {len(concept_texts)} concepts, {weighted_sens.shape[0]} samples")

    print("3. Loading similarity matrix...")
    sim_matrix = load_similarity_matrix(results_dir, seed=args.seed)
    print(f"   Shape: {sim_matrix.shape}")

    print("4. Preparing image paths...")
    image_paths = prepare_image_paths(config, seed=args.seed)
    print(f"   Loaded {len(image_paths)} image paths")

    # Determine presence data source
    presence_rates = {}  # concept_name -> presence_rate
    presence_source = 'none'

    if args.annotations_csv:
        print(f"5. Loading human annotations from {args.annotations_csv}...")
        presence_rates = load_presence_from_annotations(Path(args.annotations_csv), concept_texts)
        presence_source = 'human_annotations'
        print(f"   Found presence rates for {len(presence_rates)} concepts")
    elif args.presence_json:
        print(f"5. Loading LLM judge presence from {args.presence_json}...")
        presence_data = load_presence_analysis(Path(args.presence_json))
        per_concept = presence_data.get('summary_statistics', {}).get('per_concept', {})
        for concept, stats in per_concept.items():
            presence_rates[concept] = stats.get('presence_rate_reasoning', 0.0)
        presence_source = 'llm_judge'
        print(f"   Found presence rates for {len(presence_rates)} concepts")
    else:
        print("5. No presence data provided. Loading concepts from pval CSVs...")
        pval_concepts = load_concepts_from_pval_csvs(results_dir)
        if pval_concepts:
            # Use pval concepts with presence_rate=0 (unknown)
            for c in pval_concepts:
                presence_rates[c['concept']] = 0.0
            presence_source = 'pval_csvs_only'
            print(f"   Loaded {len(pval_concepts)} significant concepts (no presence data)")
        else:
            print("   WARNING: No pval CSVs found either. Using top-K by sensitivity.")
            presence_source = 'sensitivity_only'

    print(f"   Presence source: {presence_source}")

    # Build omission metrics dataframe
    print("6. Computing omission metrics...")
    sens_mean = np.mean(np.abs(weighted_sens), axis=0)
    sens_ranks_array = np.argsort(-sens_mean)
    sens_ranks = np.zeros(len(concept_texts), dtype=int)
    for rank, idx in enumerate(sens_ranks_array):
        sens_ranks[idx] = rank + 1

    records = []
    concept_to_idx = {c: i for i, c in enumerate(concept_texts)}

    if presence_rates:
        # Use concepts from presence data
        for concept_name, rate in presence_rates.items():
            if concept_name in concept_to_idx:
                idx = concept_to_idx[concept_name]
                records.append({
                    'concept': concept_name,
                    'concept_idx': idx,
                    'sens_mean': sens_mean[idx],
                    'sens_rank': int(sens_ranks[idx]),
                    'presence_rate': rate,
                    'omission_rate': 1.0 - rate,
                })
    else:
        # No presence data at all — use top K by sensitivity
        top_k = args.top_k or 20
        for idx in sens_ranks_array[:top_k]:
            records.append({
                'concept': concept_texts[idx],
                'concept_idx': int(idx),
                'sens_mean': sens_mean[idx],
                'sens_rank': int(sens_ranks[idx]),
                'presence_rate': 0.0,
                'omission_rate': 1.0,
            })

    df = pd.DataFrame(records)

    if args.top_k and len(df) > args.top_k:
        df = df.nsmallest(args.top_k, 'sens_rank')

    if len(df) == 0:
        print("WARNING: No concepts found. Check data.")
        return

    # Compute omission score
    n_concepts = len(concept_texts)
    df['sensitivity_percentile'] = 1.0 - (df['sens_rank'] / n_concepts)
    df['omission_score'] = df['sensitivity_percentile'] * df['omission_rate']
    df = df.sort_values('omission_score', ascending=False).reset_index(drop=True)

    print(f"   Analyzing {len(df)} concepts")

    # Generate reports and visualizations
    print("7. Generating validation report and visualizations...")
    report_data = generate_validation_report(
        df=df,
        sim_matrix=sim_matrix,
        image_paths=image_paths,
        output_dir=output_dir,
        n_images=args.n_images,
        top_n_visualize=min(args.n_visualize, len(df))
    )

    # Generate markdown report
    print("8. Generating markdown report...")
    generate_markdown_report(df, output_dir / "omitted_features_report.md", top_n=min(20, len(df)))

    # Save JSON output
    print("9. Saving JSON output...")
    json_output = {
        'metadata': {
            'vcr_results_dir': str(results_dir),
            'presence_source': presence_source,
            'annotations_csv': args.annotations_csv,
            'presence_json': args.presence_json,
            'analysis_timestamp': datetime.now().isoformat(),
            'top_k_sensitivity': args.top_k,
            'n_concepts_analyzed': len(df),
            'seed': args.seed
        },
        'summary_statistics': {
            'avg_omission_rate': float(df['omission_rate'].mean()),
            'avg_presence_rate': float(df['presence_rate'].mean()),
            'concepts_omission_above_50pct': int((df['omission_rate'] > 0.5).sum()),
            'concepts_omission_above_70pct': int((df['omission_rate'] > 0.7).sum()),
            'max_omission_score': float(df['omission_score'].max()) if len(df) > 0 else 0,
        },
        **report_data
    }

    json_output_path = output_dir / "omitted_features_analysis.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"   Saved: {json_output_path}")

    # Print summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Presence source: {presence_source}")
    print(f"Concepts analyzed: {len(df)}")
    if presence_source not in ('pval_csvs_only', 'sensitivity_only'):
        print(f"Average omission rate: {df['omission_rate'].mean():.1%}")
        print(f"Concepts with >50% omission: {(df['omission_rate'] > 0.5).sum()}")
    else:
        print("(No presence data — omission rates are placeholders)")

    print()
    print("Top 10 concepts:")
    for idx, row in enumerate(df.head(10).itertuples(), 1):
        if presence_source not in ('pval_csvs_only', 'sensitivity_only'):
            print(f"  {idx:2d}. {row.concept:25s} sens_rank={row.sens_rank:3d}  omission={row.omission_rate:.1%}")
        else:
            print(f"  {idx:2d}. {row.concept:25s} sens_rank={row.sens_rank:3d}")

    print()
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
