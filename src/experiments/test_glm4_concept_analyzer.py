"""
Test script for GLM-4.1V-9B-Thinking concept analyzer.

This script helps:
1. Discover available layer names in the GLM model
2. Test layer hooking and activation collection
3. Debug the VCR pipeline components
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch

# Compatibility patch for torch.compiler.is_compiling
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.glm4_vision import GLM4VisionAPI


def test_model_initialization():
    """Test that the model can be initialized and inspect its structure."""
    print("="*70)
    print("Test 1: Model Initialization and Structure Inspection")
    print("="*70)

    try:
        model = GLM4VisionAPI(model_name='GLM-4.1V-9B-Thinking')
        print("Model initialized successfully")
        return model
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_model_structure(model):
    """Inspect the model structure to find available layers."""
    print("\n" + "="*70)
    print("Test 2: Model Structure Inspection")
    print("="*70)

    base_model = model.model
    if hasattr(base_model, 'module'):
        base_model = base_model.module

    print(f"\nBase model type: {type(base_model)}")
    print(f"\nTop-level attributes:")
    for attr in dir(base_model):
        if not attr.startswith('_'):
            obj = getattr(base_model, attr, None)
            if isinstance(obj, torch.nn.Module):
                print(f"  {attr}: {type(obj).__name__}")

    # Try to find language model layers
    print("\n" + "-"*50)
    print("Searching for language model layers...")
    print("-"*50)

    layer_paths = []

    def find_layers(module, prefix=""):
        """Recursively find all layer modules."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this is a layers container
            if name == 'layers' and hasattr(child, '__len__'):
                num_layers = len(child)
                print(f"  Found layers at '{full_name}' with {num_layers} layers")
                for i in range(min(3, num_layers)):  # Show first 3
                    layer_paths.append(f"{full_name}.{i}")
                if num_layers > 3:
                    layer_paths.append(f"{full_name}.{num_layers-1}")  # Also add last
                    print(f"    Example paths: {full_name}.0, {full_name}.1, ..., {full_name}.{num_layers-1}")
            elif name == 'blocks' and hasattr(child, '__len__'):
                num_blocks = len(child)
                print(f"  Found blocks at '{full_name}' with {num_blocks} blocks")
                layer_paths.append(f"{full_name}.0")
                if num_blocks > 1:
                    layer_paths.append(f"{full_name}.{num_blocks-1}")
            else:
                # Recurse into child modules
                find_layers(child, full_name)

    find_layers(base_model)

    print(f"\nDiscovered {len(layer_paths)} example layer paths:")
    for path in layer_paths[:10]:
        print(f"  {path}")
    if len(layer_paths) > 10:
        print(f"  ... and {len(layer_paths) - 10} more")

    return layer_paths


def test_layer_hooking(model, layer_name):
    """Test hooking a specific layer and collecting activations."""
    print("\n" + "="*70)
    print(f"Test 3: Layer Hooking - {layer_name}")
    print("="*70)

    base_model = model.model
    if hasattr(base_model, 'module'):
        base_model = base_model.module

    # Try to access the layer
    try:
        parts = layer_name.split('.')
        target = base_model
        for part in parts:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)
        print(f"Successfully accessed layer: {layer_name}")
        print(f"Layer type: {type(target).__name__}")

        # Check layer properties
        if hasattr(target, 'weight'):
            print(f"Layer has weights with shape: {target.weight.shape}")

        # Count parameters
        num_params = sum(p.numel() for p in target.parameters())
        print(f"Layer parameters: {num_params:,}")

        return True
    except Exception as e:
        print(f"Failed to access layer '{layer_name}': {e}")
        return False


def test_activation_collection(model, layer_name):
    """Test collecting activations from a hooked layer with a sample forward pass."""
    print("\n" + "="*70)
    print(f"Test 4: Activation Collection - {layer_name}")
    print("="*70)

    base_model = model.model
    if hasattr(base_model, 'module'):
        base_model = base_model.module

    # Access the layer
    try:
        parts = layer_name.split('.')
        target = base_model
        parent = None
        last_part = None

        for i, part in enumerate(parts):
            parent = target
            last_part = part
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)

        print(f"Layer accessed: {type(target).__name__}")
    except Exception as e:
        print(f"Failed to access layer: {e}")
        return None

    # Set up hook
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        activations.append(act.detach().cpu())
        print(f"  Hook captured activation with shape: {act.shape}")

    hook = target.register_forward_hook(hook_fn)

    # Run a forward pass with a simple text prompt
    print("\nRunning forward pass with text-only prompt...")
    try:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello, what is 2+2?"}]}
        ]

        inputs = model.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(base_model.device)

        print(f"Input shape: {inputs['input_ids'].shape}")

        with torch.no_grad():
            outputs = base_model(**inputs)

        print(f"Forward pass completed successfully")
        print(f"Number of activations captured: {len(activations)}")

        if activations:
            act = activations[0]
            print(f"\nActivation details:")
            print(f"  Shape: {act.shape}")
            print(f"  Dtype: {act.dtype}")
            print(f"  Min: {act.min().item():.6f}")
            print(f"  Max: {act.max().item():.6f}")
            print(f"  Mean: {act.mean().item():.6f}")
            print(f"  Std: {act.std().item():.6f}")

            # Check for any NaN or Inf
            if torch.isnan(act).any():
                print("  WARNING: Contains NaN values!")
            if torch.isinf(act).any():
                print("  WARNING: Contains Inf values!")

    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hook.remove()

    return activations


def test_with_image(model, layer_name):
    """Test activation collection with an image input."""
    print("\n" + "="*70)
    print(f"Test 5: Activation Collection with Image - {layer_name}")
    print("="*70)

    base_model = model.model
    if hasattr(base_model, 'module'):
        base_model = base_model.module

    # Access the layer
    try:
        parts = layer_name.split('.')
        target = base_model
        for part in parts:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)
    except Exception as e:
        print(f"Failed to access layer: {e}")
        return None

    # Set up hook
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output
        activations.append(act.detach().cpu())

    hook = target.register_forward_hook(hook_fn)

    # Test with image URL
    test_image_url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"

    print(f"Testing with image: {test_image_url}")
    try:
        # Load image
        img = model.load_and_resize_image(test_image_url)
        print(f"Image loaded: {img.size}")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What animal is in this image?"}
            ]}
        ]

        inputs = model.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(base_model.device)

        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        if 'pixel_values' in inputs:
            print(f"Pixel values shape: {inputs['pixel_values'].shape}")

        with torch.no_grad():
            outputs = base_model(**inputs)

        print(f"Forward pass with image completed successfully")
        print(f"Number of activations captured: {len(activations)}")

        if activations:
            act = activations[0]
            print(f"\nActivation details:")
            print(f"  Shape: {act.shape}")
            print(f"  Dtype: {act.dtype}")
            print(f"  Last token activation shape: {act[:, -1, :].shape}")

    except Exception as e:
        print(f"Forward pass with image failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        hook.remove()

    return activations


def test_gradient_flow(model, layer_name):
    """Test that gradients can flow through the hooked layer."""
    print("\n" + "="*70)
    print(f"Test 6: Gradient Flow Test - {layer_name}")
    print("="*70)

    base_model = model.model
    if hasattr(base_model, 'module'):
        base_model = base_model.module

    # Access the layer
    try:
        parts = layer_name.split('.')
        target = base_model
        for part in parts:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)
    except Exception as e:
        print(f"Failed to access layer: {e}")
        return False

    # Set up hook that preserves gradients
    layer_outputs = []

    def hook_fn(module, input, output):
        layer_outputs.append(output)

    hook = target.register_forward_hook(hook_fn)

    print("Running forward pass with gradient tracking...")
    try:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2? Answer: 4"}]}
        ]

        inputs = model.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        ).to(base_model.device)

        with torch.set_grad_enabled(True):
            outputs = base_model(**inputs)
            logits = outputs.logits

            # Get log probability of last token
            last_token_logits = logits[0, -2, :]  # Position predicting last token
            log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
            target_token = inputs['input_ids'][0, -1]
            log_prob = log_probs[target_token]

            print(f"Log probability of target token: {log_prob.item():.4f}")

            # Get activation
            if layer_outputs:
                if isinstance(layer_outputs[-1], tuple):
                    activation = layer_outputs[-1][0]
                else:
                    activation = layer_outputs[-1]

                print(f"Activation shape: {activation.shape}")
                print(f"Activation requires_grad: {activation.requires_grad}")

                # Compute gradient
                if activation.requires_grad:
                    grad = torch.autograd.grad(
                        outputs=log_prob,
                        inputs=activation,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True
                    )[0]

                    if grad is not None:
                        print(f"\nGradient computed successfully!")
                        print(f"  Gradient shape: {grad.shape}")
                        print(f"  Gradient min/max: {grad.min().item():.8f} / {grad.max().item():.8f}")
                        print(f"  Gradient abs mean: {grad.abs().mean().item():.8f}")
                        print(f"  Non-zero elements: {(grad != 0).sum().item()} / {grad.numel()}")

                        # Check which positions have gradients
                        grad_per_pos = grad[0].abs().sum(dim=-1)
                        nonzero_pos = (grad_per_pos > 0).nonzero(as_tuple=True)[0]
                        print(f"  Positions with non-zero grad: {len(nonzero_pos)}")
                        if len(nonzero_pos) > 0:
                            print(f"  First few positions: {nonzero_pos[:5].tolist()}")
                            print(f"  Last position (seq_len-1): {grad.shape[1] - 1}")

                        return True
                    else:
                        print("WARNING: Gradient is None!")
                        return False
                else:
                    print("WARNING: Activation does not require grad!")
                    return False
            else:
                print("WARNING: No activations captured!")
                return False

    except Exception as e:
        print(f"Gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        hook.remove()


def find_best_layer(model, layer_paths):
    """Test multiple layers and find the best one for VCR analysis."""
    print("\n" + "="*70)
    print("Test 7: Finding Best Layer for VCR Analysis")
    print("="*70)

    # Filter to just language model layers (not vision)
    lang_layers = [p for p in layer_paths if 'vision' not in p.lower() and 'visual' not in p.lower()]

    if not lang_layers:
        print("No language model layers found!")
        return None

    print(f"Testing {len(lang_layers)} language model layers...")

    working_layers = []
    for layer_name in lang_layers:
        print(f"\n  Testing: {layer_name}")
        if test_layer_hooking(model, layer_name):
            working_layers.append(layer_name)

    if working_layers:
        # Recommend the last layer (typically best for VCR)
        recommended = working_layers[-1]
        print(f"\n" + "="*50)
        print(f"RECOMMENDED LAYER: {recommended}")
        print(f"="*50)
        print(f"\nAll working layers ({len(working_layers)}):")
        for layer in working_layers:
            print(f"  {layer}")
        return recommended
    else:
        print("No working layers found!")
        return None


def main():
    print("GLM-4.1V-9B-Thinking Concept Analyzer Test Suite")
    print("="*70)

    # Test 1: Initialize model
    model = test_model_initialization()
    if model is None:
        print("\nModel initialization failed, cannot proceed with tests")
        sys.exit(1)

    # Test 2: Inspect model structure
    layer_paths = inspect_model_structure(model)

    if not layer_paths:
        print("\nNo layers found, trying common patterns...")
        # Try common layer naming patterns
        common_patterns = [
            "language_model.model.layers.0",
            "model.layers.0",
            "transformer.layers.0",
            "model.model.layers.0",
        ]
        for pattern in common_patterns:
            if test_layer_hooking(model, pattern):
                layer_paths = [pattern]
                break

    if not layer_paths:
        print("\nCould not find any valid layer paths!")
        sys.exit(1)

    # Find best layer
    best_layer = find_best_layer(model, layer_paths)

    if best_layer:
        # Test 3-6 with best layer
        test_layer_hooking(model, best_layer)
        test_activation_collection(model, best_layer)
        test_with_image(model, best_layer)
        test_gradient_flow(model, best_layer)

    print("\n" + "="*70)
    print("Test Suite Complete!")
    print("="*70)

    if best_layer:
        print(f"\nTo use this layer in experiments, run:")
        print(f"  python bootstrap_resample_for_pvalues_glm4.py --layer {best_layer}")


if __name__ == "__main__":
    main()
