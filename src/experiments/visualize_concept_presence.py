"""
Visualization script for concept presence analysis results.

Creates multiple visualizations to compare concept presence in reasoning vs answers:
1. Heatmap: Shows presence/absence across all traces
2. Bar chart: Compares presence rates between reasoning and answers
3. Venn diagram: Shows overlap statistics
4. Per-trace analysis: Detailed breakdown

Usage:
    python src/experiments/visualize_concept_presence.py \
        --input_json "results/concept_presence_analysis/output.json" \
        --output_dir "results/visualizations"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_results(path: str) -> Dict[str, Any]:
    """Load concept presence analysis results."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_heatmap(results: Dict[str, Any], output_dir: Path):
    """Create heatmap showing concept presence across traces."""
    trace_analyses = results['trace_analyses']
    concepts = [c['concept'] for c in results['concepts']]

    n_traces = len(trace_analyses)
    n_concepts = len(concepts)

    # Create binary matrices
    reasoning_matrix = np.zeros((n_traces, n_concepts))
    answer_matrix = np.zeros((n_traces, n_concepts))

    for i, trace in enumerate(trace_analyses):
        for j, concept in enumerate(concepts):
            if concept in trace['reasoning_analysis']['concepts_present']:
                reasoning_matrix[i, j] = 1
            if concept in trace['answer_analysis']['concepts_present']:
                answer_matrix[i, j] = 1

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Reasoning heatmap
    sns.heatmap(reasoning_matrix, ax=ax1, cmap='Blues', cbar_kws={'label': 'Present'},
                xticklabels=concepts, yticklabels=[f"Trace {t['trace_index']}" for t in trace_analyses],
                linewidths=0.5, linecolor='gray')
    ax1.set_title('Concept Presence in Reasoning Traces', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Concepts', fontsize=12)
    ax1.set_ylabel('Traces', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Answer heatmap
    sns.heatmap(answer_matrix, ax=ax2, cmap='Oranges', cbar_kws={'label': 'Present'},
                xticklabels=concepts, yticklabels=[f"Trace {t['trace_index']}" for t in trace_analyses],
                linewidths=0.5, linecolor='gray')
    ax2.set_title('Concept Presence in Answers', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Concepts', fontsize=12)
    ax2.set_ylabel('Traces', fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'concept_presence_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


def create_bar_chart(results: Dict[str, Any], output_dir: Path, top_k: int = 20):
    """Create bar chart comparing presence rates."""
    stats = results['summary_statistics']['per_concept']

    # Sort concepts by presence in reasoning
    concept_stats = [(concept, data['presence_rate_reasoning'], data['presence_rate_answer'])
                     for concept, data in stats.items()]
    concept_stats.sort(key=lambda x: x[1], reverse=True)

    # Take top K
    concept_stats = concept_stats[:top_k]
    concepts = [c[0] for c in concept_stats]
    reasoning_rates = [c[1] for c in concept_stats]
    answer_rates = [c[2] for c in concept_stats]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(concepts))
    width = 0.35

    bars1 = ax.bar(x - width/2, reasoning_rates, width, label='Reasoning', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, answer_rates, width, label='Answer', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Concepts', fontsize=12, fontweight='bold')
    ax.set_ylabel('Presence Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Concepts: Presence Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only show label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}',
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / 'concept_presence_comparison.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved bar chart to {output_path}")
    plt.close()


def create_venn_diagram(results: Dict[str, Any], output_dir: Path):
    """Create Venn diagram showing overlap between reasoning and answer concepts."""
    trace_analyses = results['trace_analyses']

    # Aggregate statistics across all traces
    only_reasoning = 0
    only_answer = 0
    both = 0

    for trace in trace_analyses:
        reasoning_concepts = set(trace['reasoning_analysis']['concepts_present'])
        answer_concepts = set(trace['answer_analysis']['concepts_present'])

        only_reasoning += len(reasoning_concepts - answer_concepts)
        only_answer += len(answer_concepts - reasoning_concepts)
        both += len(reasoning_concepts & answer_concepts)

    # Create Venn diagram
    fig, ax = plt.subplots(figsize=(10, 8))

    venn = venn2(subsets=(only_reasoning, only_answer, both),
                 set_labels=('Reasoning Only', 'Answer Only'),
                 set_colors=('#3498db', '#e74c3c'),
                 alpha=0.6,
                 ax=ax)

    # Add "Both" label
    if venn.get_label_by_id('11'):
        venn.get_label_by_id('11').set_text(f'Both\n{both}')

    ax.set_title('Concept Overlap: Reasoning vs Answer\n(Total mentions across all traces)',
                 fontsize=14, fontweight='bold')

    # Add statistics text
    total = only_reasoning + only_answer + both
    stats_text = f"""
Total concept mentions: {total}
Only in reasoning: {only_reasoning} ({only_reasoning/total:.1%})
Only in answer: {only_answer} ({only_answer/total:.1%})
In both: {both} ({both/total:.1%})
"""
    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = output_dir / 'concept_overlap_venn.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved Venn diagram to {output_path}")
    plt.close()


def create_per_trace_analysis(results: Dict[str, Any], output_dir: Path):
    """Create per-trace analysis showing concept counts."""
    trace_analyses = results['trace_analyses']

    # Extract data
    trace_indices = [t['trace_index'] for t in trace_analyses]
    reasoning_counts = [len(t['reasoning_analysis']['concepts_present']) for t in trace_analyses]
    answer_counts = [len(t['answer_analysis']['concepts_present']) for t in trace_analyses]
    labels = [t['label'] for t in trace_analyses]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(trace_indices))
    width = 0.35

    # Color by label
    reasoning_colors = ['#2ecc71' if label == 'Benign' else '#e67e22' for label in labels]
    answer_colors = ['#27ae60' if label == 'Benign' else '#d35400' for label in labels]

    bars1 = ax.bar(x - width/2, reasoning_counts, width, label='Reasoning',
                   color=reasoning_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, answer_counts, width, label='Answer',
                   color=answer_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Trace Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Concepts Present', fontsize=12, fontweight='bold')
    ax.set_title('Concept Count per Trace (Green=Benign, Orange=Malignant)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(trace_indices, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'per_trace_concept_counts.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved per-trace analysis to {output_path}")
    plt.close()


def create_concept_correlation_matrix(results: Dict[str, Any], output_dir: Path):
    """Create correlation matrix showing which concepts co-occur."""
    trace_analyses = results['trace_analyses']
    concepts = [c['concept'] for c in results['concepts']]
    n_concepts = len(concepts)

    # Create co-occurrence matrix (reasoning + answer combined)
    cooccurrence = np.zeros((n_concepts, n_concepts))

    for trace in trace_analyses:
        present_concepts = set(
            trace['reasoning_analysis']['concepts_present'] +
            trace['answer_analysis']['concepts_present']
        )

        for i, concept1 in enumerate(concepts):
            if concept1 in present_concepts:
                for j, concept2 in enumerate(concepts):
                    if concept2 in present_concepts:
                        cooccurrence[i, j] += 1

    # Normalize to get correlation-like values
    # (number of times they appear together / geometric mean of individual appearances)
    diag = np.sqrt(np.diag(cooccurrence))
    correlation = cooccurrence / np.outer(diag, diag)
    np.fill_diagonal(correlation, 1.0)  # Set diagonal to 1

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(correlation, ax=ax, cmap='coolwarm', center=0,
                xticklabels=concepts, yticklabels=concepts,
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Co-occurrence Rate'})
    ax.set_title('Concept Co-occurrence Matrix\n(How often concepts appear together)',
                 fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    output_path = output_dir / 'concept_cooccurrence_matrix.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved co-occurrence matrix to {output_path}")
    plt.close()


def generate_summary_report(results: Dict[str, Any], output_dir: Path):
    """Generate text summary report."""
    metadata = results['metadata']
    stats = results['summary_statistics']

    report = f"""
# Concept Presence Analysis Report

## Metadata
- Input File: {metadata['input_file']}
- Judge Model: {metadata['judge_model']}
- Analysis Date: {metadata['analysis_timestamp']}
- Traces Analyzed: {metadata['num_traces_analyzed']}
- Concepts: {metadata['num_concepts']}
- Temperature: {metadata.get('temperature', 'N/A')}

## Overall Statistics
- Average concepts per reasoning: {stats['overall']['avg_concepts_per_reasoning']:.2f}
- Average concepts per answer: {stats['overall']['avg_concepts_per_answer']:.2f}
- Traces with concept matches: {stats['overall']['traces_with_concept_matches']} / {metadata['num_traces_analyzed']}

## Top 10 Concepts by Presence in Reasoning
"""

    # Sort concepts by presence rate
    concept_stats = [(name, data) for name, data in stats['per_concept'].items()]
    concept_stats.sort(key=lambda x: x[1]['presence_rate_reasoning'], reverse=True)

    report += "\n| Rank | Concept | Reasoning Rate | Answer Rate | Difference |\n"
    report += "|------|---------|----------------|-------------|------------|\n"

    for i, (concept, data) in enumerate(concept_stats[:10], 1):
        r_rate = data['presence_rate_reasoning']
        a_rate = data['presence_rate_answer']
        diff = r_rate - a_rate
        report += f"| {i} | {concept:15s} | {r_rate:6.1%} | {a_rate:6.1%} | {diff:+6.1%} |\n"

    # Add parsing success rate
    parse_failures = sum(1 for t in results['trace_analyses']
                        if not t['reasoning_analysis']['parse_successful']
                        or not t['answer_analysis']['parse_successful'])

    report += f"\n## Parsing Statistics\n"
    report += f"- Successful parses: {metadata['num_traces_analyzed'] * 2 - parse_failures} / {metadata['num_traces_analyzed'] * 2}\n"
    report += f"- Parse failures: {parse_failures}\n"

    # Save report
    output_path = output_dir / 'summary_report.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Saved summary report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize concept presence analysis results'
    )
    parser.add_argument(
        '--input_json',
        type=str,
        default=None,
        help='Path to concept presence analysis JSON output (defaults to most recent in results/concept_presence_analysis/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/concept_presence_visualizations',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help='Number of top concepts to show in bar chart'
    )

    args = parser.parse_args()

    # Auto-detect most recent JSON file if not specified
    if args.input_json is None:
        results_dir = Path('results/concept_presence_analysis')
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        json_files = list(results_dir.glob('*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {results_dir}")

        # Sort by modification time, most recent first
        args.input_json = str(max(json_files, key=lambda p: p.stat().st_mtime))
        print(f"Auto-detected most recent file: {args.input_json}")
        print()

    print("=" * 80)
    print("Concept Presence Visualization")
    print("=" * 80)
    print(f"Input: {args.input_json}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load results
    print("Loading results...")
    results = load_results(args.input_json)
    print(f"✓ Loaded {results['metadata']['num_traces_analyzed']} traces")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("Generating visualizations...")
    create_heatmap(results, output_dir)
    create_bar_chart(results, output_dir, top_k=args.top_k)
    create_venn_diagram(results, output_dir)
    create_per_trace_analysis(results, output_dir)
    create_concept_correlation_matrix(results, output_dir)
    generate_summary_report(results, output_dir)

    print()
    print("=" * 80)
    print("✓ Visualization complete!")
    print(f"✓ All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
