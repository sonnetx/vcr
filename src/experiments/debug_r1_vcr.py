"""
Debugging utilities for R1-Onevision VCR implementation.

This script provides diagnostic functions to help debug issues with:
- Layer hooks and activation capture
- Gradient flow through the model
- Directional derivative computation
- Model outputs and predictions

"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from PIL import Image

from models.r1_onevision import R1OnevisionAPI
from experiments.r1_concept_analyzer import R1ConceptAnalyzer, create_r1_analyzer
from interpretability.utils import ImageDataset


def check_layer_hook_working(analyzer: R1ConceptAnalyzer, test_image_path: str):
    """
    Verify that layer hook is capturing activations correctly.

    Args:
        analyzer: R1ConceptAnalyzer with hook already set up
        test_image_path: Path to a test image
    """
    print("=" * 70)
    print("CHECKING LAYER HOOK")
    print("=" * 70)

    # Check initial state
    print(f"\nActivations before forward pass: {len(analyzer._activations)}")

    # Test forward pass
    system_prompt = "You are a medical image analysis assistant."
    query_template = "Is this lesion benign or malignant?"

    # Create proper dataset with identity preprocessor (same as bootstrap script)
    identity_preprocess = lambda x: x
    test_dataset = ImageDataset([test_image_path], identity_preprocess)

    # Collect activations
    activations = analyzer.collect_activations(
        dataset=test_dataset,
        system_prompt=system_prompt,
        query_template=query_template,
        batch_size=1
    )

    # Check results
    print(f"Activations after forward pass: {len(analyzer._activations)}")

    if len(activations) > 0:
        print(f"✓ Hook is working!")
        print(f"  Activation shape: {activations[0].shape}")
        print(f"  Activation dtype: {activations[0].dtype}")
        print(f"  Activation mean: {activations[0].mean():.4f}")
        print(f"  Activation std: {activations[0].std():.4f}")
    else:
        print("✗ Hook is NOT capturing activations!")
        print("  Check that layer name is correct")
        print("  Run inspect_r1_layers.py to find valid layer names")


def verify_model_outputs(analyzer: R1ConceptAnalyzer, test_image_path: str):
    """
    Test that model can generate predictions and reasoning traces.

    Args:
        analyzer: R1ConceptAnalyzer instance
        test_image_path: Path to a test image
    """
    print("\n" + "=" * 70)
    print("VERIFYING MODEL OUTPUTS")
    print("=" * 70)

    try:
        result = analyzer.model(
            prompt="What is in this image?",
            image_paths=[test_image_path],
            max_new_tokens=50,
            return_reasoning=True
        )

        print("\n✓ Model forward pass successful!")
        print(f"  Reasoning length: {len(result.get('reasoning', ''))}")
        print(f"  Answer length: {len(result.get('answer', ''))}")
        print(f"\n  First 100 chars of reasoning: {result.get('reasoning', '')[:100]}")
        print(f"  Answer: {result.get('answer', '')}")

    except Exception as e:
        print(f"\n✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def check_gradient_flow(
    analyzer: R1ConceptAnalyzer,
    test_image_path: str,
    system_prompt: str,
    query_template: str,
    completion: str = " malignant"
):
    """
    Verify that gradients are flowing correctly through the model.

    Args:
        analyzer: R1ConceptAnalyzer with hook set up
        test_image_path: Path to test image
        system_prompt: System prompt
        query_template: Query template
        completion: Target completion to compute gradients for
    """
    print("\n" + "=" * 70)
    print("CHECKING GRADIENT FLOW")
    print("=" * 70)

    # Use temporary hook (like calculate_directional_derivatives does)
    layer_outputs = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            layer_outputs.append(output[0])
        else:
            layer_outputs.append(output)

    hook = analyzer.wrapped_layer.register_forward_hook(hook_fn)

    try:
        # Forward pass
        log_prob = analyzer._forward_with_completion(
            test_image_path, system_prompt, query_template, completion
        )

        # Check activation
        if len(layer_outputs) == 0:
            print("\n✗ No activation captured!")
            return

        activation = layer_outputs[-1]
        print(f"\n✓ Activation captured")
        print(f"  Activation requires_grad: {activation.requires_grad}")
        print(f"  Activation shape: {activation.shape}")
        print(f"  Log prob value: {log_prob.item():.6f}")
        print(f"  Log prob requires_grad: {log_prob.requires_grad}")

        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=log_prob,
            inputs=activation,
            create_graph=False,
            retain_graph=False
        )[0]

        print(f"\n✓ Gradient computed successfully")
        print(f"  Gradient shape: {gradient.shape}")
        print(f"  Gradient mean: {gradient.mean().item():.6f}")
        print(f"  Gradient std: {gradient.std().item():.6f}")
        print(f"  Gradient min: {gradient.min().item():.6f}")
        print(f"  Gradient max: {gradient.max().item():.6f}")

        # Check if gradients are zero
        if torch.all(gradient == 0):
            print("\n⚠ Warning: All gradients are zero!")
            print("  This may indicate a problem with gradient flow")

    except Exception as e:
        print(f"\n✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hook.remove()


def verify_sensitivities(
    raw_sens: np.ndarray,
    weighted_sens: np.ndarray,
    concept_texts: List[str],
    top_k: int = 20
):
    """
    Verify and visualize concept sensitivities.

    Args:
        raw_sens: Raw sensitivities array
        weighted_sens: Weighted sensitivities array
        concept_texts: List of concept text strings
        top_k: Number of top concepts to display
    """
    print("\n" + "=" * 70)
    print("VERIFYING SENSITIVITIES")
    print("=" * 70)

    print(f"\nRaw sensitivities:")
    print(f"  Shape: {raw_sens.shape}")
    print(f"  Range: [{raw_sens.min():.4f}, {raw_sens.max():.4f}]")
    print(f"  Mean: {raw_sens.mean():.4f}")
    print(f"  Std: {raw_sens.std():.4f}")

    print(f"\nWeighted sensitivities:")
    print(f"  Shape: {weighted_sens.shape}")
    print(f"  Range: [{weighted_sens.min():.4f}, {weighted_sens.max():.4f}]")
    print(f"  Mean: {weighted_sens.mean():.4f}")
    print(f"  Std: {weighted_sens.std():.4f}")

    # Check for all-zero sensitivities
    if np.all(raw_sens == 0):
        print("\n✗ WARNING: All raw sensitivities are zero!")
        print("  Possible causes:")
        print("  1. Hook not capturing activations")
        print("  2. Gradient not flowing through activation")
        print("  3. Log probability computation incorrect")
        return

    # Get top concepts
    top_indices = np.argsort(np.abs(weighted_sens))[-top_k:][::-1]

    print(f"\nTop {top_k} concepts by absolute weighted sensitivity:")
    for i, idx in enumerate(top_indices, 1):
        concept = concept_texts[idx] if idx < len(concept_texts) else f"concept_{idx}"
        print(f"  {i:2d}. {concept:30s} | raw: {raw_sens[idx]:8.4f} | weighted: {weighted_sens[idx]:8.4f}")


def plot_sensitivity_distribution(raw_sens: np.ndarray, weighted_sens: np.ndarray):
    """
    Plot distributions of raw and weighted sensitivities.

    Args:
        raw_sens: Raw sensitivities array
        weighted_sens: Weighted sensitivities array
    """
    print("\n" + "=" * 70)
    print("PLOTTING SENSITIVITY DISTRIBUTIONS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw sensitivities
    axes[0].hist(raw_sens, bins=50, edgecolor='black')
    axes[0].set_title('Raw Sensitivities Distribution')
    axes[0].set_xlabel('Sensitivity')
    axes[0].set_ylabel('Count')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Weighted sensitivities
    axes[1].hist(weighted_sens, bins=50, edgecolor='black')
    axes[1].set_title('Weighted Sensitivities Distribution')
    axes[1].set_xlabel('Sensitivity')
    axes[1].set_ylabel('Count')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('sensitivity_distributions.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot to: sensitivity_distributions.png")
    plt.close()


def compare_gradient_vs_correlation(
    analyzer: R1ConceptAnalyzer,
    gradient_sens: np.ndarray,
    similarity_matrix: np.ndarray,
    model_predictions: np.ndarray,
    concept_texts: List[str],
    top_k: int = 20
):
    """
    Compare gradient-based sensitivities with correlation-based approach.

    Args:
        analyzer: R1ConceptAnalyzer instance
        gradient_sens: Gradient-based sensitivities
        similarity_matrix: CLIP similarity matrix [n_concepts, n_images]
        model_predictions: Model's malignant probabilities [n_images]
        concept_texts: List of concept text strings
        top_k: Number of top concepts to compare
    """
    print("\n" + "=" * 70)
    print("COMPARING GRADIENT VS CORRELATION")
    print("=" * 70)

    # Compute correlation-based sensitivities
    correlation_sens = []
    for i in range(similarity_matrix.shape[0]):
        concept_sims = similarity_matrix[i, :]
        correlation = np.corrcoef(concept_sims, model_predictions)[0, 1]
        correlation_sens.append(correlation)
    correlation_sens = np.array(correlation_sens)

    # Get top concepts for each method
    grad_top_indices = np.argsort(np.abs(gradient_sens))[-top_k:][::-1]
    corr_top_indices = np.argsort(np.abs(correlation_sens))[-top_k:][::-1]

    print(f"\nTop {top_k} concepts by gradient-based sensitivity:")
    for i, idx in enumerate(grad_top_indices[:10], 1):
        concept = concept_texts[idx] if idx < len(concept_texts) else f"concept_{idx}"
        print(f"  {i:2d}. {concept:30s} | gradient: {gradient_sens[idx]:8.4f} | correlation: {correlation_sens[idx]:8.4f}")

    print(f"\nTop {top_k} concepts by correlation-based sensitivity:")
    for i, idx in enumerate(corr_top_indices[:10], 1):
        concept = concept_texts[idx] if idx < len(concept_texts) else f"concept_{idx}"
        print(f"  {i:2d}. {concept:30s} | gradient: {gradient_sens[idx]:8.4f} | correlation: {correlation_sens[idx]:8.4f}")

    # Compute overlap
    grad_top_set = set(grad_top_indices)
    corr_top_set = set(corr_top_indices)
    overlap = grad_top_set.intersection(corr_top_set)

    print(f"\nOverlap between top-{top_k} concepts:")
    print(f"  {len(overlap)} / {top_k} ({100 * len(overlap) / top_k:.1f}%)")


def diagnose_common_issues(analyzer: R1ConceptAnalyzer):
    """
    Check for common issues and provide recommendations.

    Args:
        analyzer: R1ConceptAnalyzer instance
    """
    print("\n" + "=" * 70)
    print("DIAGNOSING COMMON ISSUES")
    print("=" * 70)

    issues = []

    # Check 1: Hook installed
    if analyzer.wrapped_layer is None:
        issues.append("✗ No hook installed. Call setup_layer_hook() first.")
    else:
        print("✓ Hook is installed")

    # Check 2: Model on correct device
    print(f"✓ Model device: {analyzer.model.model.device}")

    # Check 3: Model in eval mode
    if analyzer.model.model.training:
        issues.append("⚠ Model is in training mode. Should be in eval mode for inference.")
    else:
        print("✓ Model in eval mode")

    # Print issues
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No common issues detected")


def full_diagnostic_suite(
    model_name: str = 'R1-Onevision-7B',
    layer_name: str = 'model.model.language_model.layers.27',
    test_image_path: Optional[str] = None
):
    """
    Run full diagnostic suite on R1-Onevision VCR implementation.

    Args:
        model_name: R1-Onevision model variant
        layer_name: Layer to hook for analysis
        test_image_path: Path to test image (if None, will skip image-based tests)
    """
    print("=" * 70)
    print("R1-ONEVISION VCR DIAGNOSTIC SUITE")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Layer: {layer_name}")
    print(f"Test image: {test_image_path or 'Not provided'}")

    # Create analyzer
    print("\nCreating analyzer...")
    try:
        analyzer = create_r1_analyzer(model_name=model_name)
        print("✓ Analyzer created successfully")
    except Exception as e:
        print(f"✗ Failed to create analyzer: {e}")
        return

    # Setup hook
    print(f"\nSetting up layer hook: {layer_name}")
    try:
        analyzer.setup_layer_hook(layer_name)
        print("✓ Layer hook set up successfully")
    except Exception as e:
        print(f"✗ Failed to set up layer hook: {e}")
        print("\nTip: Run inspect_r1_layers.py to find valid layer names")
        return

    # Diagnose common issues
    diagnose_common_issues(analyzer)

    # If test image provided, run image-based tests
    if test_image_path and os.path.exists(test_image_path):
        check_layer_hook_working(analyzer, test_image_path)
        verify_model_outputs(analyzer, test_image_path)

        # Test gradient flow
        system_prompt = "You are a medical image analysis assistant."
        query_template = "Is this lesion benign or malignant?"
        check_gradient_flow(analyzer, test_image_path, system_prompt, query_template)
    else:
        print("\n⚠ No test image provided - skipping image-based tests")
        print("  Provide test_image_path to run full diagnostics")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Debug R1-Onevision VCR implementation')
    parser.add_argument('--model', type=str, default='R1-Onevision-7B',
                       choices=['R1-Onevision-7B'],
                       help='R1-Onevision model variant')
    parser.add_argument('--layer', type=str, default='model.model.language_model.layers.27',
                       help='Layer name to hook (use inspect_r1_layers.py to find valid names)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to test image')

    args = parser.parse_args()

    full_diagnostic_suite(
        model_name=args.model,
        layer_name=args.layer,
        test_image_path=args.image
    )
