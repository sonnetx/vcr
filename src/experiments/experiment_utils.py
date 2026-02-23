"""
Shared utilities for VCR experiments.

This module contains common dataclasses, configuration objects, and utility functions
used across multiple experiment scripts to avoid duplication.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import Counter


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class PromptConfig:
    """Configuration for prompt templates - works for both Flamingo and R1 models."""
    # For Flamingo models
    base_prompt: str = ""
    demo_template: str = ""
    query_template: str = ""
    completion: str = ""
    use_demos: bool = False

    # For R1 models (overrides base_prompt, demo_template if set)
    system_prompt: str = ""

    def __post_init__(self):
        """Set defaults for R1 models if system_prompt is provided."""
        if self.system_prompt and not self.base_prompt:
            # R1 model defaults
            if not self.query_template:
                self.query_template = "Analyze this skin lesion image and determine if it is benign or malignant."


@dataclass
class ExperimentConfig:
    """Dataclass for experiment parameters - unified across Flamingo and R1 experiments."""
    results_dir: str
    model_name: str
    layer_name: str
    metadata_path: str
    ddi_base_dir: str
    prompt: PromptConfig = field(default_factory=PromptConfig)
    concept_files: List[str] = field(default_factory=list)
    test_size: float = 0.5
    demo_size: float = 0.02
    random_state: int = 42
    filter_skin_tone: Optional[int] = None
    task_definition: str = 'malignant_prob'
    top_k_concepts: int = 20  # Number of top concepts to analyze


# ============================================================================
# Reasoning Trace Analysis
# ============================================================================

def extract_concepts_from_reasoning(reasoning_text: str, concept_texts: List[str]) -> Dict[str, int]:
    """
    Extract and count concept mentions in reasoning trace.

    Args:
        reasoning_text: The reasoning trace text
        concept_texts: List of all possible concept texts

    Returns:
        Dictionary mapping concept -> count of mentions
    """
    if not reasoning_text:
        return {}

    # Convert reasoning to lowercase for case-insensitive matching
    reasoning_lower = reasoning_text.lower()

    # Count mentions of each concept using word boundaries
    concept_counts = {}
    for concept in concept_texts:
        # Skip single-character concepts (too noisy, e.g., 'a', 'i')
        if len(concept) <= 1:
            continue

        concept_lower = concept.lower()

        # Use regex word boundaries to match whole words only
        pattern = r'\b' + re.escape(concept_lower) + r'\b'
        matches = re.findall(pattern, reasoning_lower)
        count = len(matches)

        if count > 0:
            concept_counts[concept] = count

    return concept_counts


def analyze_reasoning_concept_overlap(
    reasoning_traces: List[str],
    top_concepts: List[str],
    all_concepts: List[str]
) -> Dict[str, Any]:
    """
    Analyze overlap between concepts mentioned in reasoning traces and top identified concepts.

    Args:
        reasoning_traces: List of reasoning traces from model
        top_concepts: Top N concepts identified by sensitivity analysis
        all_concepts: All possible concepts

    Returns:
        Dictionary with analysis results
    """
    # Extract concepts from all reasoning traces
    all_reasoning_concepts = Counter()
    for trace in reasoning_traces:
        concepts = extract_concepts_from_reasoning(trace, all_concepts)
        all_reasoning_concepts.update(concepts)

    # Rank only the top VCR concepts by their mention frequency in reasoning
    top_concept_reasoning_counts = {
        c: all_reasoning_concepts.get(c, 0) for c in top_concepts
    }
    most_common_in_reasoning = sorted(
        [c for c, count in top_concept_reasoning_counts.items() if count > 0],
        key=lambda c: top_concept_reasoning_counts[c],
        reverse=True
    )

    # Calculate overlap: which top VCR concepts appear at all in reasoning
    overlap = [c for c in top_concepts if all_reasoning_concepts.get(c, 0) > 0]

    # Calculate statistics
    results = {
        'top_concepts': top_concepts,
        'most_common_in_reasoning': most_common_in_reasoning,
        'top_concept_reasoning_counts': top_concept_reasoning_counts,
        'overlap': overlap,
        'overlap_count': len(overlap),
        'overlap_percentage': len(overlap) / len(top_concepts) * 100 if top_concepts else 0,
        'reasoning_concept_counts': dict(all_reasoning_concepts.most_common(100)),
        'total_reasoning_concepts': len(all_reasoning_concepts),
    }

    return results


def compute_cross_seed_consistency(
    per_seed_top_concepts: List[List[str]],
    top_k: int
) -> Dict[str, Any]:
    """
    Compute consistency of top-K VCR sensitivity concepts across random seeds.

    Args:
        per_seed_top_concepts: List of top-K concept lists, one per seed
        top_k: Number of top concepts used

    Returns:
        Dictionary with Jaccard similarities and stable concept lists
    """
    import numpy as np
    from itertools import combinations

    num_seeds = len(per_seed_top_concepts)
    top_sets = [set(concepts[:top_k]) for concepts in per_seed_top_concepts]

    # Pairwise Jaccard similarity
    jaccard_pairwise = []
    for i, j in combinations(range(num_seeds), 2):
        intersection = len(top_sets[i] & top_sets[j])
        union = len(top_sets[i] | top_sets[j])
        jaccard_pairwise.append(intersection / union if union > 0 else 0.0)

    # Count how many seeds each concept appears in
    concept_seed_counts = Counter()
    for s in top_sets:
        concept_seed_counts.update(s)

    stable_100 = sorted([c for c, n in concept_seed_counts.items() if n == num_seeds])
    stable_75 = sorted([c for c, n in concept_seed_counts.items() if n >= max(1, int(num_seeds * 0.75))])
    stable_50 = sorted([c for c, n in concept_seed_counts.items() if n >= max(1, int(num_seeds * 0.5))])

    return {
        'jaccard_mean': float(np.mean(jaccard_pairwise)) if jaccard_pairwise else 1.0,
        'jaccard_std': float(np.std(jaccard_pairwise)) if jaccard_pairwise else 0.0,
        'jaccard_pairwise': jaccard_pairwise,
        'stable_concepts_100pct': stable_100,
        'stable_concepts_75pct': stable_75,
        'stable_concepts_50pct': stable_50,
        'num_stable_100pct': len(stable_100),
        'num_stable_75pct': len(stable_75),
        'num_stable_50pct': len(stable_50),
    }


# ============================================================================
# Concept Deduplication
# ============================================================================

def deduplicate_concepts(concepts, presence_matrix=None):
    """Deduplicate concepts that share identical definition strings.

    Multiple VCR codewords can map to the same semantic definition. This
    collapses them into a single entry, preserving the mapping back to
    all original codeword names.

    Args:
        concepts: list of concept dicts, each with at least:
            - 'concept': str (definition string, used as dedup key)
            - 'original_name': str (VCR codeword)
        presence_matrix: optional numpy array (n_concepts x n_traces).
            If provided, rows for merged concepts are combined via OR.

    Returns:
        If presence_matrix is None: deduped_concepts list
        If presence_matrix is provided: (deduped_concepts, merged_matrix)
    """
    from collections import OrderedDict

    seen = OrderedDict()  # definition_key -> (new_index, concept_dict)
    old_to_new = {}       # old_index -> new_index

    for i, concept in enumerate(concepts):
        key = concept['concept'].strip()
        original = concept.get('original_name', concept['concept'])

        if key in seen:
            new_idx, existing = seen[key]
            existing['all_original_names'].append(original)
            existing['duplicate_count'] += 1
            old_to_new[i] = new_idx
        else:
            new_idx = len(seen)
            deduped = dict(concept)
            deduped['all_original_names'] = [original]
            deduped['duplicate_count'] = 1
            seen[key] = (new_idx, deduped)
            old_to_new[i] = new_idx

    deduped_concepts = []
    for rank, (_, (_, concept_dict)) in enumerate(seen.items(), 1):
        concept_dict['rank'] = rank
        deduped_concepts.append(concept_dict)

    n_removed = len(concepts) - len(deduped_concepts)
    if n_removed > 0:
        print(f"  Deduped {len(concepts)} -> {len(deduped_concepts)} concepts ({n_removed} duplicates merged)")
        for c in deduped_concepts:
            if c['duplicate_count'] > 1:
                names = ', '.join(c['all_original_names'])
                print(f"    {c['concept'][:50]}: [{names}]")

    if presence_matrix is not None:
        import numpy as np
        n_new = len(deduped_concepts)
        n_traces = presence_matrix.shape[1]
        merged = np.zeros((n_new, n_traces), dtype=presence_matrix.dtype)

        for old_idx, new_idx in old_to_new.items():
            merged[new_idx] = np.maximum(merged[new_idx], presence_matrix[old_idx])

        return deduped_concepts, merged

    return deduped_concepts


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_experiment_data(results_dir):
    """
    Load data from a previous experiment run.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with loaded data (similarity_matrix, choice_differences, concept_texts)
    """
    import json
    import numpy as np
    from pathlib import Path

    results_dir = Path(results_dir)
    data = {}

    # Check if we have saved similarity matrix and choice differences
    if (results_dir / 'similarity_matrix.npy').exists():
        data['similarity_matrix'] = np.load(results_dir / 'similarity_matrix.npy')

    if (results_dir / 'choice_differences.npy').exists():
        data['choice_differences'] = np.load(results_dir / 'choice_differences.npy')

    if (results_dir / 'concept_texts.json').exists():
        with open(results_dir / 'concept_texts.json', 'r') as f:
            concept_texts = json.load(f)
            data['concept_texts'] = concept_texts

    return data
