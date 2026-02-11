#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import stats
from pathlib import Path
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse


def load_experiment_config(results_dir):
    """Load experiment configuration from experiment_config.json"""
    config_path = Path(results_dir) / "experiment_config.json"
    with open(config_path, "r") as f:
        full_config = json.load(f)
    
    config = full_config["config"]
    return config


def load_mean_directional_derivatives(results_dir):
    """Load and compute mean directional derivatives across all seeds"""
    import re
    list_of_mean_weighted_stabilities = []

    for subdir in sorted(os.listdir(results_dir)):
        subdir_path = Path(results_dir) / subdir
        # Must be a directory matching pattern seed_N (e.g., seed_0, seed_12)
        # Explicitly skip aggregated_*, analysis_outputs, and other non-seed folders
        if not subdir_path.is_dir():
            continue
        if not re.match(r'^seed_\d+$', subdir):
            continue
        weighted_sens_path = subdir_path / "weighted_sens.npy"
        if not weighted_sens_path.exists():
            print(f"WARNING: Skipping {subdir} (missing weighted_sens.npy)")
            continue
        print(f"Loading: {subdir_path}")
        weighted_sens = np.load(weighted_sens_path)
        list_of_mean_weighted_stabilities.append(weighted_sens.mean(axis=0))

    if not list_of_mean_weighted_stabilities:
        raise ValueError(f"No valid seed directories found in {results_dir}")

    return np.vstack(list_of_mean_weighted_stabilities)


def compute_statistics(mean_directional_derivatives):
    """Compute t-statistics and p-values"""
    t_stats, p_values = stats.ttest_1samp(
        mean_directional_derivatives,
        0,  # null hypothesis: mean = 0
        axis=0,
        alternative='two-sided'
    )
    return t_stats, p_values


def get_significant_concepts(mean_per_concept, p_values, concept_texts, n_concepts=20, alpha=0.05, require_significance=False):
    """Get top N concepts in positive and negative directions.

    Args:
        mean_per_concept: Mean directional derivative per concept
        p_values: P-values from t-test
        concept_texts: List of concept names
        n_concepts: Number of top concepts to return
        alpha: Significance level for Bonferroni correction
        require_significance: If True, only return significant concepts.
                              If False, return top concepts regardless of significance.
    """
    n_total_concepts = len(concept_texts)
    bonferroni_threshold = alpha / n_total_concepts

    # Get indices of significant concepts
    significant_mask = p_values < bonferroni_threshold
    n_significant = significant_mask.sum()

    print(f"   Bonferroni threshold: {bonferroni_threshold:.2e}")
    print(f"   Significant concepts: {n_significant} / {n_total_concepts}")

    # Sort by mean directional derivative
    sorted_indices = np.argsort(mean_per_concept)

    # Get top positive (highest mean)
    positive_concepts = []
    for idx in sorted_indices[::-1]:
        if require_significance and not significant_mask[idx]:
            continue
        positive_concepts.append({
            'rank': len(positive_concepts) + 1,
            'concept': concept_texts[idx],
            'mean_dd': mean_per_concept[idx],
            'p_value': p_values[idx],
            'significant': bool(significant_mask[idx]),
            'concept_idx': idx
        })
        if len(positive_concepts) >= n_concepts:
            break

    # Get top negative (lowest mean)
    negative_concepts = []
    for idx in sorted_indices:
        if require_significance and not significant_mask[idx]:
            continue
        negative_concepts.append({
            'rank': len(negative_concepts) + 1,
            'concept': concept_texts[idx],
            'mean_dd': mean_per_concept[idx],
            'p_value': p_values[idx],
            'significant': bool(significant_mask[idx]),
            'concept_idx': idx
        })
        if len(negative_concepts) >= n_concepts:
            break

    return positive_concepts, negative_concepts


def save_concept_tables(positive_concepts, negative_concepts, results_dir):
    """Save concept tables to CSV files"""
    output_dir = Path(results_dir) / "analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle empty lists
    if not positive_concepts and not negative_concepts:
        print("WARNING: No concepts to save")
        return output_dir

    # Save positive concepts
    if positive_concepts:
        df_pos = pd.DataFrame(positive_concepts)
        cols = ['rank', 'concept', 'mean_dd', 'p_value']
        if 'significant' in df_pos.columns:
            cols.append('significant')
        df_pos = df_pos[cols]
        pos_path = output_dir / f"top_{len(positive_concepts)}_positive_concepts.csv"
        df_pos.to_csv(pos_path, index=False)
        print(f"Saved: {pos_path}")

    # Save negative concepts
    if negative_concepts:
        df_neg = pd.DataFrame(negative_concepts)
        cols = ['rank', 'concept', 'mean_dd', 'p_value']
        if 'significant' in df_neg.columns:
            cols.append('significant')
        df_neg = df_neg[cols]
        neg_path = output_dir / f"top_{len(negative_concepts)}_negative_concepts.csv"
        df_neg.to_csv(neg_path, index=False)
        print(f"Saved: {neg_path}")

    return output_dir


def center_crop_to_square(img):
    """Center crop image to a square using the smaller dimension"""
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def create_concept_visualization(concept_idx, concept_name, top_indices, bottom_indices,
                               concept_activations, probe_paths, n_images, save_path):
    """Create and save visualization for a single concept"""
    fig, axes = plt.subplots(2, n_images, figsize=(16, 10))
    fig.suptitle(f'Concept: {concept_name}', fontsize=16, fontweight='bold', y=0.75)
    
    # Ensure axes is 2D
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    # Top activated images
    for i, img_idx in enumerate(top_indices):
        try:
            img_path = probe_paths[img_idx]
            img = Image.open(img_path).convert('RGB')
            img_cropped = center_crop_to_square(img)
            axes[0, i].imshow(img_cropped)
            axes[0, i].axis('off')
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f'Error loading\n{Path(probe_paths[img_idx]).name}', 
                          ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].axis('off')
    
    # Bottom activated images  
    for i, img_idx in enumerate(bottom_indices):
        try:
            img_path = probe_paths[img_idx]
            img = Image.open(img_path).convert('RGB')
            img_cropped = center_crop_to_square(img)
            axes[1, i].imshow(img_cropped)
            axes[1, i].axis('off')
        except Exception as e:
            axes[1, i].text(0.5, 0.5, f'Error loading\n{Path(probe_paths[img_idx]).name}', 
                          ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
    
    # Add row labels
    fig.text(0.06, 0.60, 'Most Activating\nImages', ha='center', va='center', 
             fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.06, 0.35, 'Least Activating\nImages', ha='center', va='center', 
             fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, top=0.9, bottom=0.05, hspace=-0.6)
    
    # Save with 300 DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.close()


def load_saved_image_paths(results_dir, seed=0):
    """
    Load image paths from saved JSON file.

    This is the preferred method as it guarantees correct index alignment
    with the saved similarity matrix.

    Args:
        results_dir: Path to results directory
        seed: Seed folder to load from

    Returns:
        List of image paths, or None if file doesn't exist
    """
    image_paths_file = Path(results_dir) / f"seed_{seed}" / "image_paths.json"
    if image_paths_file.exists():
        with open(image_paths_file, 'r') as f:
            paths = json.load(f)
        return [Path(p) for p in paths]
    return None


def prepare_image_paths(config, seed=0):
    """
    Prepare train and test image paths based on config.

    DEPRECATED: This regenerates paths using train_test_split which may not
    match the original experiment's image ordering. Use load_saved_image_paths()
    instead when image_paths.json is available.

    Args:
        config: Experiment configuration
        seed: Random seed to use (default 0 for seed_0 folder similarity matrix)
    """
    metadata_path = config['metadata_path']
    base_dir = config['ddi_base_dir']
    test_size = config['test_size']
    demo_size = config['demo_size']
    use_demos = config['prompt'].get('use_demos', False)

    # Load metadata
    df = pd.read_csv(metadata_path, index_col=0)

    # This matches DDIDataLoader.__init__ logic
    # First split: train/test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed  # Use the seed parameter
    )

    # Second split: demos (if needed)
    if use_demos:
        train_df, demo_df = train_test_split(
            train_df,
            test_size=demo_size,
            random_state=seed  # Use the same seed
        )
        print(f"Using ICL with {len(demo_df)} demo images")

    # Create probe paths (train set after demo split)
    probe_paths = [Path(base_dir) / file for file in train_df.DDI_file]

    return probe_paths


def generate_visualizations_for_concepts(concepts, sim_matrix, probe_paths, 
                                        concept_texts, output_dir, n_images=7):
    """Generate visualizations for a list of concepts"""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    for concept_data in concepts:
        concept_idx = concept_data['concept_idx']
        concept_name = concept_data['concept']
        
        # Get activation scores for this concept
        concept_activations = sim_matrix[concept_idx, :]
        
        # Get indices sorted by activation
        sorted_indices = np.argsort(-concept_activations)
        top_indices = sorted_indices[:n_images]
        bottom_indices = sorted_indices[-n_images:]
        
        # Create safe filename
        safe_name = "".join(c for c in concept_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name[:50]  # Limit length
        save_path = viz_dir / f"concept_{concept_data['rank']:02d}_{safe_name}.png"
        
        create_concept_visualization(
            concept_idx, concept_name, top_indices, bottom_indices,
            concept_activations, probe_paths, n_images, save_path
        )


def main(results_dir, n_concepts=20, n_images=7):
    """Main analysis pipeline with comprehensive error handling"""
    results_dir = Path(results_dir)

    print("="*60)
    print(f"Starting analysis for: {results_dir}")
    print("="*60)

    try:
        # Load experiment config
        print("\n1. Loading experiment configuration...")
        try:
            exp_config = load_experiment_config(results_dir)
        except FileNotFoundError:
            print(f"ERROR: experiment_config.json not found in {results_dir}")
            print("Skipping this directory...")
            return False
        except json.JSONDecodeError:
            print(f"ERROR: Invalid JSON in experiment_config.json")
            print("Skipping this directory...")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load experiment config: {e}")
            print("Skipping this directory...")
            return False

        # Load directional derivatives
        print("\n2. Loading mean directional derivatives...")
        try:
            mean_directional_derivatives = load_mean_directional_derivatives(results_dir)
            print(f"   Shape: {mean_directional_derivatives.shape}")
        except Exception as e:
            print(f"ERROR: Failed to load directional derivatives: {e}")
            print("Skipping this directory...")
            return False

        # Compute statistics
        print("\n3. Computing statistics...")
        try:
            t_stats, p_values = compute_statistics(mean_directional_derivatives)
            mean_per_concept = np.mean(mean_directional_derivatives, axis=0)

            n_total = len(p_values)
            n_significant = ((p_values * n_total) < 0.05).sum()
            print(f"   Total concepts: {n_total}")
            print(f"   Significant concepts (Bonferroni corrected): {n_significant}")
        except Exception as e:
            print(f"ERROR: Failed to compute statistics: {e}")
            print("Skipping this directory...")
            return False

        # Load concept texts
        print("\n4. Loading concept texts...")
        try:
            concept_texts_path = results_dir / "seed_0" / "concept_texts.json"
            with open(concept_texts_path, "r") as f:
                concept_texts = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: concept_texts.json not found at {concept_texts_path}")
            print("Skipping this directory...")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load concept texts: {e}")
            print("Skipping this directory...")
            return False

        # Get significant concepts
        print(f"\n5. Extracting top {n_concepts} significant concepts...")
        try:
            positive_concepts, negative_concepts = get_significant_concepts(
                mean_per_concept, p_values, concept_texts, n_concepts=n_concepts
            )
            print(f"   Found {len(positive_concepts)} positive and {len(negative_concepts)} negative concepts")
        except Exception as e:
            print(f"ERROR: Failed to extract significant concepts: {e}")
            print("Skipping this directory...")
            return False

        # Save tables
        print("\n6. Saving concept tables to CSV...")
        try:
            output_dir = save_concept_tables(positive_concepts, negative_concepts, results_dir)
        except Exception as e:
            print(f"ERROR: Failed to save concept tables: {e}")
            print("Skipping this directory...")
            return False

        # Prepare image paths
        print("\n7. Preparing image paths...")
        try:
            # First try to load saved image paths (preferred - guarantees correct alignment)
            probe_paths = load_saved_image_paths(results_dir, seed=0)
            if probe_paths is not None:
                print(f"   Loaded {len(probe_paths)} probe images from saved image_paths.json")
            else:
                # Fall back to regenerating paths (may have index mismatch issues)
                print("   WARNING: image_paths.json not found, regenerating paths (may cause index mismatch)")
                probe_paths = prepare_image_paths(exp_config, seed=0)
                print(f"   Regenerated {len(probe_paths)} probe images")
        except Exception as e:
            print(f"ERROR: Failed to prepare image paths: {e}")
            print("Skipping this directory...")
            return False

        # Load similarity matrix
        print("\n8. Loading similarity matrix...")
        try:
            sim_matrix_path = results_dir / "seed_0" / "similarity_matrix.npy"
            sim_matrix = np.load(sim_matrix_path)
            print(f"   Shape: {sim_matrix.shape}")
        except FileNotFoundError:
            print(f"ERROR: similarity_matrix.npy not found at {sim_matrix_path}")
            print("Skipping this directory...")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load similarity matrix: {e}")
            print("Skipping this directory...")
            return False

        # Generate visualizations
        print(f"\n9. Generating visualizations ({n_images} images per concept)...")
        try:
            print("   Positive concepts:")
            generate_visualizations_for_concepts(
                positive_concepts, sim_matrix, probe_paths, concept_texts, output_dir, n_images
            )
            print("   Negative concepts:")
            generate_visualizations_for_concepts(
                negative_concepts, sim_matrix, probe_paths, concept_texts, output_dir, n_images
            )
        except Exception as e:
            print(f"WARNING: Failed to generate some visualizations: {e}")
            print("Continuing anyway...")

        print("\n" + "="*60)
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}")
        print("="*60)
        return True

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        print("Skipping this directory...")
        import traceback
        traceback.print_exc()
        return False


def should_include_folder(folder_name):
    """Check if folder should be included based on suffix"""
    folder_lower = folder_name.lower()
    return 'contrastive' in folder_lower or 'malignant_prob' in folder_lower or 'malig_prob' in folder_lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze concept significance and generate visualizations from experimental results.'
    )
    parser.add_argument(
        'results_dirs',
        nargs='*',
        default=["/home/groups/roxanad/sonnet/vcr/scripts/contrastive_10k",
                 "/home/groups/roxanad/sonnet/vcr/scripts/malig_prob10k"],
        help='Directories containing experiment results. Can specify multiple directories.'
    )
    parser.add_argument(
        '--n-concepts',
        type=int,
        default=50,
        help='Number of top concepts to extract (default: 50)'
    )
    parser.add_argument(
        '--n-images',
        type=int,
        default=7,
        help='Number of images per concept visualization (default: 7)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip directories that already have analysis_outputs folder'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Direct mode: run on a single results directory (bypasses parent-dir walking)'
    )

    args = parser.parse_args()

    # Direct single-dir mode
    if args.results_dir:
        success = main(args.results_dir, n_concepts=args.n_concepts, n_images=args.n_images)
        sys.exit(0 if success else 1)

    # Track statistics
    total_processed = 0
    total_success = 0
    total_skipped = 0
    total_failed = 0

    # Process each results directory
    for results_dirs in args.results_dirs:
        results_dirs = Path(results_dirs)

        if not results_dirs.exists():
            print(f"WARNING: Directory does not exist: {results_dirs}")
            continue

        if not results_dirs.is_dir():
            print(f"WARNING: Not a directory: {results_dirs}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing parent directory: {results_dirs}")
        print(f"{'='*80}\n")

        # Get all subdirectories
        try:
            subdirs = [d for d in os.listdir(results_dirs) if os.path.isdir(os.path.join(results_dirs, d))]
        except Exception as e:
            print(f"ERROR: Failed to list directory {results_dirs}: {e}")
            continue

        # Filter subdirectories
        filtered_subdirs = [d for d in subdirs if should_include_folder(d)]

        if not filtered_subdirs:
            print(f"No matching subdirectories found in {results_dirs}")
            print("Looking for folders containing 'contrastive' or 'malignant_prob' or 'malig_prob'")
            continue

        print(f"Found {len(filtered_subdirs)} matching subdirectories (out of {len(subdirs)} total)")
        print(f"Filtered subdirectories: {filtered_subdirs}\n")

        # Process each filtered subdirectory
        for results_dir in filtered_subdirs:
            full_path = os.path.join(results_dirs, results_dir)
            analysis_path = os.path.join(full_path, 'analysis_outputs')

            # Check if we should skip existing
            if args.skip_existing and os.path.exists(analysis_path):
                print(f"Skipping {results_dir} (analysis_outputs already exists)")
                total_skipped += 1
                continue

            total_processed += 1
            print(f"\n[{total_processed}] Processing: {results_dir}")

            # Run analysis with error handling
            success = main(full_path, n_concepts=args.n_concepts, n_images=args.n_images)

            if success:
                total_success += 1
            else:
                total_failed += 1

            print()  # Add spacing between runs

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total directories processed: {total_processed}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Skipped (already processed): {total_skipped}")
    print(f"{'='*80}\n")