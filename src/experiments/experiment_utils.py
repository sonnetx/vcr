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

    # Get most common concepts in reasoning
    most_common_reasoning = [concept for concept, count in all_reasoning_concepts.most_common(50)]

    # Calculate overlap with top concepts
    overlap = set(top_concepts) & set(most_common_reasoning)

    # Calculate statistics
    results = {
        'top_concepts': top_concepts,
        'most_common_in_reasoning': most_common_reasoning[:len(top_concepts)],
        'overlap': list(overlap),
        'overlap_count': len(overlap),
        'overlap_percentage': len(overlap) / len(top_concepts) * 100 if top_concepts else 0,
        'reasoning_concept_counts': dict(all_reasoning_concepts.most_common(100)),
        'total_reasoning_concepts': len(all_reasoning_concepts),
    }

    return results


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
