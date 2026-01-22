"""
Test script for GLM-4.1V-9B-Thinking model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.glm4_vision import GLM4VisionAPI

def test_basic_initialization():
    """Test that the model can be initialized"""
    print("="*70)
    print("Test 1: Model Initialization")
    print("="*70)
    try:
        model = GLM4VisionAPI(model_name='GLM-4.1V-9B-Thinking')
        print("Model initialized successfully")
        return model
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return None

def test_reasoning_extraction():
    """Test the reasoning extraction functionality without loading model"""
    print("="*70)
    print("Test: Reasoning Extraction Patterns")
    print("="*70)

    # Create a dummy instance just for testing extract_reasoning
    # We'll test the method directly without loading the model
    class DummyAPI:
        pass

    import re
    from typing import Tuple

    def extract_reasoning(text: str) -> Tuple[str, str]:
        """Copy of the extract_reasoning method for testing"""
        # Pattern 1: <think>...</think> followed by <answer>...</answer> (PRIMARY for GLM)
        think_answer_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)(?:</answer>|$)'
        think_answer_match = re.search(think_answer_pattern, text, re.DOTALL)

        if think_answer_match:
            reasoning = think_answer_match.group(1).strip()
            final_answer = think_answer_match.group(2).strip()
            return reasoning, final_answer

        # Pattern 2: R1-style <think>...<answer> without </think>
        think_answer_no_close_pattern = r'<think>(.*?)<answer>(.*?)(?:</answer>|$)'
        think_answer_no_close_match = re.search(think_answer_no_close_pattern, text, re.DOTALL)

        if think_answer_no_close_match:
            reasoning = think_answer_no_close_match.group(1).strip()
            final_answer = think_answer_no_close_match.group(2).strip()
            return reasoning, final_answer

        # Pattern 3: <think>...</think> without <answer> tags
        think_pattern = r'<think>(.*?)</think>\s*(.*?)$'
        think_match = re.search(think_pattern, text, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            final_answer = think_match.group(2).strip()
            final_answer = re.sub(r'^<answer>\s*', '', final_answer)
            final_answer = re.sub(r'\s*</answer>$', '', final_answer)
            return reasoning, final_answer

        # Pattern 4: <think> tag without closing
        think_only_pattern = r'<think>(.*?)$'
        think_only_match = re.search(think_only_pattern, text, re.DOTALL)

        if think_only_match:
            reasoning = think_only_match.group(1).strip()
            answer_in_think = re.search(r'<answer>(.*?)(?:</answer>|$)', reasoning, re.DOTALL)
            if answer_in_think:
                reasoning = reasoning[:reasoning.index('<answer>')].strip()
                final_answer = answer_in_think.group(1).strip()
                return reasoning, final_answer
            conclusion_pattern = r'(.*?)(?:Therefore|Thus|In conclusion|To summarize|Final answer)[,:]?\s*(.*?)$'
            conclusion_match = re.search(conclusion_pattern, reasoning, re.DOTALL | re.IGNORECASE)
            if conclusion_match and conclusion_match.group(2).strip():
                return conclusion_match.group(1).strip(), conclusion_match.group(2).strip()
            final_answer = reasoning.split('.')[-1].strip() if '.' in reasoning else reasoning
            return reasoning, final_answer

        # Default
        return "", text.strip()

    # Test cases
    test_cases = [
        # Test 1: GLM format - <think>...</think><answer>...</answer>
        (
            "<think>Let me analyze this step by step. First, I see a dog.</think><answer>This is a golden retriever dog.</answer>",
            ("Let me analyze this step by step. First, I see a dog.", "This is a golden retriever dog.")
        ),
        # Test 2: GLM format without closing </answer> tag
        (
            "<think>Analyzing the image carefully.</think><answer>The animal is a corgi.",
            ("Analyzing the image carefully.", "The animal is a corgi.")
        ),
        # Test 3: Standard <think>...</think> format without answer tags
        (
            "<think>Let me think about this.</think>The answer is 4.",
            ("Let me think about this.", "The answer is 4.")
        ),
        # Test 4: Incomplete think tag with answer inside
        (
            "<think>I'm analyzing. The animal has fur.<answer>It is a dog.</answer>",
            ("I'm analyzing. The animal has fur.", "It is a dog.")
        ),
        # Test 5: No think tags at all
        (
            "This is a simple response without any reasoning tags.",
            ("", "This is a simple response without any reasoning tags.")
        ),
    ]

    all_passed = True
    for i, (input_text, expected) in enumerate(test_cases, 1):
        reasoning, answer = extract_reasoning(input_text)
        expected_reasoning, expected_answer = expected

        if reasoning == expected_reasoning and answer == expected_answer:
            print(f"  Test case {i}: PASSED")
        else:
            print(f"  Test case {i}: FAILED")
            print(f"    Expected reasoning: {expected_reasoning[:50]}...")
            print(f"    Got reasoning: {reasoning[:50]}...")
            print(f"    Expected answer: {expected_answer[:50]}...")
            print(f"    Got answer: {answer[:50]}...")
            all_passed = False

    if all_passed:
        print("\nAll reasoning extraction tests passed!")
    else:
        print("\nSome reasoning extraction tests failed!")

    return all_passed

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
        print("  Text-only inference completed")
    except Exception as e:
        print(f"  Text-only inference failed: {e}")
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
        print("  String output format works")
    except Exception as e:
        print(f"  String output test failed: {e}")
        raise

    print("\nText inference tests completed!")

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
        print("  Image inference completed")
    except Exception as e:
        print(f"  Image inference failed: {e}")
        raise

    print("\nImage inference tests completed!")

if __name__ == "__main__":
    print("GLM-4.1V-9B-Thinking Model Test Suite")
    print("="*70)

    # Test reasoning extraction (doesn't need model)
    test_reasoning_extraction()

    print("\n")

    # Initialize model
    model = test_basic_initialization()
    if model is None:
        print("\nModel initialization failed, cannot proceed with inference tests")
        sys.exit(1)

    # Run inference tests
    test_model_inference(model)
    test_image_inference(model)

    print("\n" + "="*70)
    print("Test Suite Complete!")
    print("="*70)
