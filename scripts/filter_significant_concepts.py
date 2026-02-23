#!/usr/bin/env python
"""
Filter concept results to only significant concepts.

Creates:
1. Filtered CSVs with only significant concepts
2. A folder with only the significant concept visualizations
3. An annotation template CSV for manual review

Usage:
    python scripts/filter_significant_concepts.py --results_dir R1-Onevision-7B_DDI_R1_malignant_prob_detailed_medical
"""

import argparse
import shutil
import pandas as pd
from pathlib import Path


def filter_significant_concepts(results_dir):
    """Filter to only significant concepts and create annotation-ready outputs."""
    results_dir = Path(results_dir)
    analysis_dir = results_dir / "analysis_outputs"

    if not analysis_dir.exists():
        print(f"ERROR: {analysis_dir} does not exist")
        return False

    # Create output directory for significant concepts only
    sig_dir = analysis_dir / "significant_only"
    sig_dir.mkdir(parents=True, exist_ok=True)

    sig_viz_dir = sig_dir / "visualizations"
    sig_viz_dir.mkdir(parents=True, exist_ok=True)

    all_significant = []

    for direction in ["positive", "negative"]:
        # Find the concepts CSV - use most recent if multiple exist
        csv_files = list(analysis_dir.glob(f"top_*_{direction}_concepts.csv"))
        if not csv_files:
            print(f"WARNING: No {direction} concepts CSV found")
            continue

        # Sort by modification time, most recent first
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        csv_path = csv_files[0]

        if len(csv_files) > 1:
            print(f"\nFound {len(csv_files)} {direction} CSVs, using most recent: {csv_path.name}")
        else:
            print(f"\nProcessing: {csv_path.name}")

        df = pd.read_csv(csv_path)
        print(f"  Total concepts: {len(df)}")

        # Filter to significant only
        if 'significant' not in df.columns:
            print(f"  WARNING: No 'significant' column found, skipping filtering")
            df_sig = df
        else:
            df_sig = df[df['significant'] == True].copy()

        print(f"  Significant concepts: {len(df_sig)}")

        if len(df_sig) == 0:
            print(f"  No significant {direction} concepts found")
            continue

        # Re-rank
        df_sig['rank'] = range(1, len(df_sig) + 1)

        # Save filtered CSV
        sig_csv_path = sig_dir / f"significant_{direction}_concepts.csv"
        df_sig.to_csv(sig_csv_path, index=False)
        print(f"  Saved: {sig_csv_path}")

        # Add direction column for combined CSV
        df_sig['direction'] = direction
        all_significant.append(df_sig)

        # Copy corresponding visualizations
        viz_dir = analysis_dir / "visualizations"
        if viz_dir.exists():
            copied = 0
            for _, row in df_sig.iterrows():
                concept_name = row['concept']
                # Find matching visualization file
                # Filename pattern: concept_NN_conceptname.png
                safe_name = "".join(c for c in concept_name if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]

                # Normalize for comparison: lowercase, replace spaces/underscores, keep only alphanumeric
                def normalize(s):
                    return ''.join(c.lower() for c in s if c.isalnum())

                target_normalized = normalize(safe_name)

                # Try to find the file with exact match (not substring)
                found = False
                for viz_file in viz_dir.glob("concept_*.png"):
                    # Extract concept name from filename: concept_NN_conceptname.png -> conceptname
                    fname = viz_file.stem  # e.g., "concept_01_blue"
                    parts = fname.split('_', 2)  # ['concept', '01', 'blue']
                    if len(parts) >= 3:
                        file_concept = parts[2]
                        file_normalized = normalize(file_concept)
                        # Exact match after normalization
                        if file_normalized == target_normalized:
                            new_name = f"{direction}_{row['rank']:02d}_{safe_name}.png"
                            shutil.copy(viz_file, sig_viz_dir / new_name)
                            copied += 1
                            found = True
                            break

                if not found:
                    print(f"    WARNING: No visualization found for '{concept_name}'")

            print(f"  Copied {copied} visualizations to {sig_viz_dir}")

    # Create combined annotation template
    if all_significant:
        combined = pd.concat(all_significant, ignore_index=True)
        combined = combined.sort_values(['direction', 'rank'])

        # Create annotation template with columns for manual review
        annotation_df = combined[['direction', 'rank', 'concept', 'mean_dd', 'p_value']].copy()
        annotation_df['keep'] = ''  # For marking whether to keep
        annotation_df['category'] = ''  # For categorizing (e.g., "color", "texture", "shape")
        annotation_df['notes'] = ''  # For any notes

        annotation_path = sig_dir / "annotation_template.csv"
        annotation_df.to_csv(annotation_path, index=False)
        print(f"\nCreated annotation template: {annotation_path}")
        print(f"  Total significant concepts: {len(annotation_df)}")
        print(f"  Columns: direction, rank, concept, mean_dd, p_value, keep, category, notes")

    print(f"\n{'='*60}")
    print(f"Output directory: {sig_dir}")
    print(f"{'='*60}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Filter to significant concepts only')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Results directory with analysis_outputs folder')
    args = parser.parse_args()

    success = filter_significant_concepts(args.results_dir)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
