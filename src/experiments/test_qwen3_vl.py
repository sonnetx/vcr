"""
Test script for Qwen3-VL-8B-Thinking model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen3_vl import Qwen3VLAPI

def test_basic_initialization():
    """Test that the model can be initialized"""
    print("="*70)
    print("Test 1: Model Initialization")
    print("="*70)
    try:
        model = Qwen3VLAPI(model_name='Qwen3-VL-8B-Thinking')
        print("✓ Model initialized successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return None

def test_model_inference(model):
    """Test model inference with actual generation"""
    print("\n" + "="*70)
    print("Test 2: Model Inference")
    print("="*70)

    # Test 1: Text-only prompt (no image)
    print("\n[Test 2.1] Text-only prompt with reasoning:")
    try:
        result = model(
            prompt="What is 2 + 2? Think through it step by step.",
            image_paths=[],
            max_new_tokens=500,
            return_reasoning=True
        )
        print(f"  Full output: {result['full_output'][:200]}...")
        print(f"  Reasoning: {result['reasoning'][:100] if result['reasoning'] else 'None'}...")
        print(f"  Answer: {result['answer'][:100]}...")
        print("  ✓ Text-only inference completed")
    except Exception as e:
        print(f"  ✗ Text-only inference failed: {e}")
        raise

    # Test 2: return_reasoning=False returns string
    print("\n[Test 2.2] Output format (return_reasoning=False):")
    try:
        result_str = model(
            prompt="What is 1 + 1?",
            max_new_tokens=200,
            return_reasoning=False
        )
        assert isinstance(result_str, str), f"Expected str, got {type(result_str)}"
        print(f"  Result (string): {result_str[:100]}...")
        print("  ✓ String output format works")
    except Exception as e:
        print(f"  ✗ String output test failed: {e}")
        raise

    print("\n✓ Text inference tests completed!")

def test_image_inference(model):
    """Test model inference with images"""
    print("\n" + "="*70)
    print("Test 3: Image Inference")
    print("="*70)

    # Using a publicly accessible test image
    test_image_url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"

    print(f"\n[Test 3.1] Image URL inference:")
    print(f"  Image: {test_image_url}")
    try:
        result = model(
            prompt="What animal is in this image? Describe what you see.",
            image_paths=[test_image_url],
            max_new_tokens=500,
            return_reasoning=True
        )
        print(f"  Full output: {result['full_output'][:200]}...")
        print(f"  Reasoning: {result['reasoning'][:100] if result['reasoning'] else 'None'}...")
        print(f"  Answer: {result['answer'][:100]}...")
        print("  ✓ Image inference completed")
    except Exception as e:
        print(f"  ✗ Image inference failed: {e}")
        raise

    print("\n✓ Image inference tests completed!")

if __name__ == "__main__":
    print("Qwen3-VL-8B-Thinking Model Test Suite")
    print("="*70)

    # Initialize model
    model = test_basic_initialization()
    if model is None:
        print("\n✗ Model initialization failed, cannot proceed with tests")
        sys.exit(1)

    # Run inference tests
    test_model_inference(model)
    test_image_inference(model)

    print("\n" + "="*70)
    print("Test Suite Complete!")
    print("="*70)
