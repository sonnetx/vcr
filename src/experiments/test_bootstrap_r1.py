"""
Quick test script to verify bootstrap_resample_for_pvalues_r1.py structure
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all imports work"""
    print("="*70)
    print("Test 1: Import Verification")
    print("="*70)
    try:
        from experiments.bootstrap_resample_for_pvalues_r1 import (
            extract_concepts_from_reasoning,
            analyze_reasoning_concept_overlap,
            PromptConfig,
            ExperimentConfig
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_concept_extraction():
    """Test concept extraction from reasoning text"""
    print("\n" + "="*70)
    print("Test 2: Concept Extraction from Reasoning")
    print("="*70)

    from experiments.bootstrap_resample_for_pvalues_r1 import extract_concepts_from_reasoning

    # Test data
    reasoning_text = """
    Looking at the lesion, I observe several key features:
    1. The color is irregular with multiple shades
    2. The border appears asymmetric
    3. The texture shows variation
    4. There are signs of darkness in certain areas
    Therefore, based on these observations, this appears concerning.
    """

    concepts = ['color', 'border', 'texture', 'asymmetric', 'darkness', 'regular', 'smooth']

    # Extract concepts
    concept_counts = extract_concepts_from_reasoning(reasoning_text, concepts)

    print(f"Reasoning text (first 100 chars): {reasoning_text[:100]}...")
    print(f"Test concepts: {concepts}")
    print(f"Found concepts: {concept_counts}")

    # Verify expected concepts were found
    assert 'color' in concept_counts, "Should find 'color'"
    assert 'border' in concept_counts, "Should find 'border'"
    assert 'texture' in concept_counts, "Should find 'texture'"
    assert 'asymmetric' in concept_counts, "Should find 'asymmetric'"

    print("✓ Concept extraction working correctly")

def test_reasoning_overlap_analysis():
    """Test reasoning-concept overlap analysis"""
    print("\n" + "="*70)
    print("Test 3: Reasoning-Concept Overlap Analysis")
    print("="*70)

    from experiments.bootstrap_resample_for_pvalues_r1 import analyze_reasoning_concept_overlap

    # Test data
    reasoning_traces = [
        "The lesion shows irregular borders and asymmetric shape with varied colors.",
        "I observe darkness and rough texture with unclear borders.",
        "The color pattern is irregular and the shape is asymmetric."
    ]

    top_concepts = ['borders', 'asymmetric', 'colors', 'darkness', 'texture']
    all_concepts = ['borders', 'asymmetric', 'colors', 'darkness', 'texture',
                   'smooth', 'regular', 'light', 'clear', 'uniform']

    # Run analysis
    results = analyze_reasoning_concept_overlap(
        reasoning_traces=reasoning_traces,
        top_concepts=top_concepts,
        all_concepts=all_concepts
    )

    print(f"Top concepts: {results['top_concepts']}")
    print(f"Most common in reasoning: {results['most_common_in_reasoning']}")
    print(f"Overlap: {results['overlap']}")
    print(f"Overlap count: {results['overlap_count']}")
    print(f"Overlap percentage: {results['overlap_percentage']:.1f}%")

    # Verify structure
    assert 'overlap_count' in results
    assert 'overlap_percentage' in results
    assert 'reasoning_concept_counts' in results
    assert results['overlap_count'] > 0, "Should have some overlap"

    print("✓ Overlap analysis working correctly")

def test_config_classes():
    """Test configuration dataclasses"""
    print("\n" + "="*70)
    print("Test 4: Configuration Dataclasses")
    print("="*70)

    from experiments.bootstrap_resample_for_pvalues_r1 import PromptConfig, ExperimentConfig

    # Test PromptConfig
    prompt_config = PromptConfig()
    print(f"PromptConfig created with system_prompt: {prompt_config.system_prompt[:50]}...")

    # Test ExperimentConfig
    exp_config = ExperimentConfig(
        results_dir='test_results',
        model_name='R1-Onevision-7B',
        layer_name='reasoning_model',
        metadata_path='/path/to/metadata.csv',
        ddi_base_dir='/path/to/ddi',
    )
    print(f"ExperimentConfig created with model: {exp_config.model_name}")

    print("✓ Configuration classes working correctly")

def test_local_clip():
    """Test that local CLIPEmbedder definition works"""
    print("\n" + "="*70)
    print("Test 5: Local CLIP Definition")
    print("="*70)

    try:
        # The script defines CLIPEmbedder locally, check it's accessible
        from experiments.bootstrap_resample_for_pvalues_r1 import CLIPEmbedder, PathDataset
        print("✓ Successfully accessed CLIPEmbedder and PathDataset from script")
        print("  (Note: These use open_clip directly, defined locally in the script)")
        return True
    except Exception as e:
        print(f"✗ Failed to access local CLIP definitions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Bootstrap R1 Test Suite")
    print("="*70)

    success = True

    # Run tests
    if not test_imports():
        print("\n✗ Import test failed, cannot continue")
        sys.exit(1)

    try:
        test_concept_extraction()
        test_reasoning_overlap_analysis()
        test_config_classes()
        if not test_local_clip():
            success = False
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "="*70)
    if success:
        print("✓ All Tests Passed!")
    else:
        print("✗ Some tests failed")
    print("="*70)
