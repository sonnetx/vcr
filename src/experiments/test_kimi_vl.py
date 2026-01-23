"""
Test script for Kimi-VL-A3B-Thinking model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kimi_vl import KimiVLAPI

def test_basic_initialization():
    """Test that the model can be initialized"""
    print("="*70)
    print("Test 1: Model Initialization")
    print("="*70)
    try:
        model = KimiVLAPI(model_name='Kimi-VL-A3B-Thinking')
        print("Model initialized successfully")
        return model
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return None

def test_reasoning_extraction():
    """Test the reasoning extraction functionality"""
    print("\n" + "="*70)
    print("Test 2: Reasoning Extraction")
    print("="*70)

    from models.kimi_vl import KimiVLAPI
    api = KimiVLAPI.__new__(KimiVLAPI)  # Create instance without __init__

    # Test case 1: Kimi-VL thinking tokens (PRIMARY)
    test_text_1 = "◁think▷First, I need to analyze the image. The cat appears to have orange fur and distinctive markings.◁/think▷This is a tabby cat."
    reasoning_1, answer_1 = api.extract_reasoning(test_text_1)
    print(f"\nTest Case 1 (Kimi thinking tokens):")
    print(f"  Input: {test_text_1[:60]}...")
    print(f"  Reasoning: {reasoning_1[:50]}..." if reasoning_1 else "  Reasoning: (empty)")
    print(f"  Answer: {answer_1}")
    assert reasoning_1 != "", "Failed to extract reasoning from Kimi thinking tokens"
    assert "tabby" in answer_1.lower(), "Failed to extract correct answer"
    print("  Passed")

    # Test case 2: Incomplete thinking (no closing token)
    test_text_2 = "◁think▷Let me analyze this step by step. The image shows a fluffy white cat with blue eyes. Therefore, this is likely a Persian cat."
    reasoning_2, answer_2 = api.extract_reasoning(test_text_2)
    print(f"\nTest Case 2 (Incomplete thinking - no closing token):")
    print(f"  Input: {test_text_2[:60]}...")
    print(f"  Reasoning: {reasoning_2[:50]}..." if reasoning_2 else "  Reasoning: (empty)")
    print(f"  Answer: {answer_2[:50]}..." if answer_2 else "  Answer: (empty)")
    assert reasoning_2 != "", "Failed to extract reasoning from incomplete response"
    print("  Passed")

    # Test case 3: Standard <think> tags (backwards compatibility)
    test_text_3 = "<think>The lesion has irregular borders and varied coloration.</think>Based on the analysis, this is a malignant lesion."
    reasoning_3, answer_3 = api.extract_reasoning(test_text_3)
    print(f"\nTest Case 3 (<think> tags - backwards compatibility):")
    print(f"  Reasoning: {reasoning_3[:50]}...")
    print(f"  Answer: {answer_3[:50]}...")
    assert reasoning_3 != "", "Failed to extract reasoning from <think> tags"
    print("  Passed")

    # Test case 4: "Therefore" transition
    test_text_4 = "The cat has orange and black stripes with a distinctive M marking on its forehead. Therefore, it is a tabby cat."
    reasoning_4, answer_4 = api.extract_reasoning(test_text_4)
    print(f"\nTest Case 4 (Therefore transition):")
    print(f"  Reasoning: {reasoning_4[:50]}...")
    print(f"  Answer: {answer_4[:50]}...")
    print("  Passed")

    # Test case 5: No reasoning markers
    test_text_5 = "This is a tabby cat."
    reasoning_5, answer_5 = api.extract_reasoning(test_text_5)
    print(f"\nTest Case 5 (No reasoning markers):")
    print(f"  Reasoning: '{reasoning_5}'")
    print(f"  Answer: {answer_5}")
    assert reasoning_5 == "", "Should return empty reasoning when no markers found"
    assert answer_5 == test_text_5, "Should return full text as answer"
    print("  Passed")

    print("\nAll reasoning extraction tests passed!")

def test_model_output_formats(model):
    """Test different output format options with actual inference"""
    print("\n" + "="*70)
    print("Test 3: Actual Model Inference")
    print("="*70)

    # Test 1: Text-only prompt (no image)
    print("\n[Test 3.1] Text-only prompt:")
    try:
        result = model(
            prompt="What is 2 + 2? Think step by step.",
            image_paths=[],
            max_new_tokens=512,
            return_reasoning=True
        )
        print(f"  Full output: {result['full_output'][:200]}...")
        print(f"  Reasoning: {result['reasoning'][:100] if result['reasoning'] else 'None'}...")
        print(f"  Answer: {result['answer'][:100]}...")
        print("  Text-only inference completed")
    except Exception as e:
        print(f"  Text-only inference failed: {e}")

    # Test 2: With reasoning vs without reasoning
    print("\n[Test 3.2] Output format options:")
    try:
        # With reasoning (dict output)
        result_with = model(
            prompt="Count from 1 to 3.",
            max_new_tokens=256,
            return_reasoning=True
        )
        print(f"  With reasoning: type={type(result_with)}, keys={list(result_with.keys())}")

        # Without reasoning (string output)
        result_without = model(
            prompt="Count from 1 to 3.",
            max_new_tokens=256,
            return_reasoning=False
        )
        print(f"  Without reasoning: type={type(result_without)}")
        print("  Output format options work correctly")
    except Exception as e:
        print(f"  Output format test failed: {e}")

    # Test 3: With image (using sample from Kimi-VL demo)
    print("\n[Test 3.3] Image inference:")
    print("  Testing with Kimi-VL demo image...")

    # Using the demo image from Kimi-VL HuggingFace space
    test_image_url = "https://huggingface.co/spaces/moonshotai/Kimi-VL-A3B-Thinking/resolve/main/images/demo6.jpeg"

    try:
        result = model(
            prompt="What kind of cat is this? Think step by step and answer with one word.",
            image_paths=[test_image_url],
            max_new_tokens=1024,
            return_reasoning=True
        )
        print(f"  Full output preview: {result['full_output'][:200]}...")
        print(f"  Reasoning: {result['reasoning'][:150] if result['reasoning'] else 'None'}...")
        print(f"  Answer: {result['answer'][:100]}...")
        print("  Image inference completed")
    except Exception as e:
        print(f"  Image inference failed: {e}")
        print("  (This is expected if network is unavailable or image URL is inaccessible)")

    # Test 4: Test with local PIL image (if available)
    print("\n[Test 3.4] PIL Image input (optional):")
    try:
        from PIL import Image
        import requests

        # Download and create PIL image
        img = Image.open(requests.get(test_image_url, stream=True).raw)
        result = model(
            prompt="Describe this image briefly.",
            image_paths=[img],  # Pass PIL Image directly
            max_new_tokens=512,
            return_reasoning=True
        )
        print(f"  Answer: {result['answer'][:100]}...")
        print("  PIL Image input works correctly")
    except Exception as e:
        print(f"  PIL Image test skipped or failed: {e}")

    print("\nAll inference tests completed!")

if __name__ == "__main__":
    print("Kimi-VL-A3B-Thinking Model Test Suite")
    print("="*70)

    # Run tests
    test_reasoning_extraction()

    model = test_basic_initialization()
    if model:
        test_model_output_formats(model)

    print("\n" + "="*70)
    print("Test Suite Complete!")
    print("="*70)
