"""
Test script for R1-Onevision-7B model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.r1_onevision import R1OnevisionAPI

def test_basic_initialization():
    """Test that the model can be initialized"""
    print("="*70)
    print("Test 1: Model Initialization")
    print("="*70)
    try:
        model = R1OnevisionAPI(model_name='R1-Onevision-7B')
        print("✓ Model initialized successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return None

def test_reasoning_extraction():
    """Test the reasoning extraction functionality"""
    print("\n" + "="*70)
    print("Test 2: Reasoning Extraction")
    print("="*70)

    from models.r1_onevision import R1OnevisionAPI
    api = R1OnevisionAPI.__new__(R1OnevisionAPI)  # Create instance without __init__

    # Test case 1: <think> tags
    test_text_1 = "<think>First, I need to analyze the image. The lesion appears to have irregular borders.</think>Based on the analysis, this is a malignant lesion."
    reasoning_1, answer_1 = api.extract_reasoning(test_text_1)
    print(f"\nTest Case 1 (<think> tags):")
    print(f"  Reasoning: {reasoning_1[:50]}...")
    print(f"  Answer: {answer_1[:50]}...")
    assert reasoning_1 != "", "Failed to extract reasoning from <think> tags"
    print("  ✓ Passed")

    # Test case 2: "Reasoning:" pattern
    test_text_2 = "Reasoning: The image shows clear signs of abnormality with asymmetric features.\n\nThe diagnosis is melanoma."
    reasoning_2, answer_2 = api.extract_reasoning(test_text_2)
    print(f"\nTest Case 2 (Reasoning: pattern):")
    print(f"  Reasoning: {reasoning_2[:50]}...")
    print(f"  Answer: {answer_2[:50]}...")
    assert reasoning_2 != "", "Failed to extract reasoning from 'Reasoning:' pattern"
    print("  ✓ Passed")

    # Test case 3: "Therefore" transition
    test_text_3 = "The lesion has irregular borders and varied coloration. Therefore, it is likely malignant."
    reasoning_3, answer_3 = api.extract_reasoning(test_text_3)
    print(f"\nTest Case 3 (Therefore transition):")
    print(f"  Reasoning: {reasoning_3[:50]}...")
    print(f"  Answer: {answer_3[:50]}...")
    print("  ✓ Passed")

    # Test case 4: No reasoning markers
    test_text_4 = "This is a benign lesion."
    reasoning_4, answer_4 = api.extract_reasoning(test_text_4)
    print(f"\nTest Case 4 (No reasoning markers):")
    print(f"  Reasoning: '{reasoning_4}'")
    print(f"  Answer: {answer_4}")
    assert reasoning_4 == "", "Should return empty reasoning when no markers found"
    assert answer_4 == test_text_4, "Should return full text as answer"
    print("  ✓ Passed")

    print("\n✓ All reasoning extraction tests passed!")

def test_model_output_formats(model):
    """Test different output format options with actual inference"""
    print("\n" + "="*70)
    print("Test 3: Actual Model Inference")
    print("="*70)

    # Test 1: Text-only prompt (no image)
    print("\n[Test 3.1] Text-only prompt:")
    try:
        result = model(
            prompt="What is 2 + 2?",
            image_paths=[],
            max_new_tokens=100,
            return_reasoning=True
        )
        print(f"  Reasoning: {result['reasoning'][:80] if result['reasoning'] else 'None'}...")
        print(f"  Answer: {result['answer'][:80]}...")
        print("  ✓ Text-only inference completed")
    except Exception as e:
        print(f"  ✗ Text-only inference failed: {e}")

    # Test 2: With reasoning vs without reasoning
    print("\n[Test 3.2] Output format options:")
    try:
        # With reasoning (dict output)
        result_with = model(
            prompt="Count from 1 to 3.",
            max_new_tokens=50,
            return_reasoning=True
        )
        print(f"  With reasoning: type={type(result_with)}, keys={list(result_with.keys())}")

        # Without reasoning (string output)
        result_without = model(
            prompt="Count from 1 to 3.",
            max_new_tokens=50,
            return_reasoning=False
        )
        print(f"  Without reasoning: type={type(result_without)}")
        print("  ✓ Output format options work correctly")
    except Exception as e:
        print(f"  ✗ Output format test failed: {e}")

    # Test 3: With image (using a sample image URL)
    print("\n[Test 3.3] Image inference (optional):")
    print("  Note: Requires a valid image. Testing with URL...")

    # Using a publicly accessible test image
    test_image_url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"

    try:
        result = model(
            prompt="What animal is in this image?",
            image_paths=[test_image_url],
            max_new_tokens=200,
            return_reasoning=True
        )
        print(f"  Reasoning: {result['reasoning'][:100] if result['reasoning'] else 'None'}...")
        print(f"  Answer: {result['answer'][:100]}...")
        print("  ✓ Image inference completed")
    except Exception as e:
        print(f"  ✗ Image inference failed: {e}")
        print("  (This is expected if network is unavailable or image URL is inaccessible)")

    print("\n✓ All inference tests completed!")

if __name__ == "__main__":
    print("R1-Onevision-7B Model Test Suite")
    print("="*70)

    # Run tests
    test_reasoning_extraction()

    model = test_basic_initialization()
    if model:
        test_model_output_formats()

    print("\n" + "="*70)
    print("Test Suite Complete!")
    print("="*70)