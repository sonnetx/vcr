"""
Helper script to inspect R1-Onevision model layer structure.
Run this to find the correct layer names for hooking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.r1_onevision import R1OnevisionAPI

def inspect_model_layers(model_name='R1-Onevision-7B'):
    """
    Inspect and print R1-Onevision model layer structure.

    Args:
        model_name: R1-Onevision model variant
    """
    print(f"Loading {model_name}...")
    r1_model = R1OnevisionAPI(model_name=model_name)

    print("\n" + "="*70)
    print("MODEL STRUCTURE")
    print("="*70)

    # Get the underlying model
    model = r1_model.model

    print(f"\nModel type: {type(model).__name__}")
    print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")

    # Print top-level attributes
    print("\n" + "-"*70)
    print("Top-level attributes:")
    print("-"*70)
    for name in dir(model):
        if not name.startswith('_'):
            attr = getattr(model, name)
            if not callable(attr):
                print(f"  {name}: {type(attr).__name__}")

    # Language model layers
    print("\n" + "-"*70)
    print("Language Model Layers:")
    print("-"*70)

    # Try different possible paths
    lang_layers = None
    layer_path = None

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        lang_layers = model.model.layers
        layer_path = 'model.model.layers'
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        lang_layers = model.language_model.model.layers
        layer_path = 'model.language_model.model.layers'
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        lang_layers = model.model.language_model.layers
        layer_path = 'model.model.language_model.layers'

    if lang_layers is not None:
        num_layers = len(lang_layers)
        print(f"  Found {num_layers} language model layers")
        print(f"  Access as: {layer_path}.0 to {layer_path}.{num_layers-1}")

        # Show structure of first and last layer
        print(f"\n  First layer ({layer_path}.0) structure:")
        first_layer = lang_layers[0]
        for name in dir(first_layer):
            if not name.startswith('_'):
                attr = getattr(first_layer, name)
                if not callable(attr):
                    print(f"    {name}: {type(attr).__name__}")

        print(f"\n  Last layer ({layer_path}.{num_layers-1}) structure:")
        last_layer = lang_layers[num_layers-1]
        for name in dir(last_layer):
            if not name.startswith('_'):
                attr = getattr(last_layer, name)
                if not callable(attr):
                    print(f"    {name}: {type(attr).__name__}")
    else:
        print("  Checking model structure...")
        if hasattr(model, 'model'):
            print(f"  model.model attributes: {[a for a in dir(model.model) if not a.startswith('_')][:20]}")
        print("  Language model layers not found at expected locations")
        print("  Check ALL NAMED MODULES below for language layers")

    # Vision encoder layers
    print("\n" + "-"*70)
    print("Vision Encoder Layers:")
    print("-"*70)
    if hasattr(model, 'visual'):
        print(f"  Found visual encoder")
        print(f"  Type: {type(model.visual).__name__}")

        if hasattr(model.visual, 'blocks'):
            num_visual_layers = len(model.visual.blocks)
            print(f"  Found {num_visual_layers} vision blocks")
            print(f"  Access as: model.visual.blocks.0 to model.visual.blocks.{num_visual_layers-1}")
        elif hasattr(model.visual, 'layers'):
            num_visual_layers = len(model.visual.layers)
            print(f"  Found {num_visual_layers} vision layers")
            print(f"  Access as: model.visual.layers.0 to model.visual.layers.{num_visual_layers-1}")
    else:
        print("  No visual encoder found at model.visual")

    # Recommended layers for hooking
    print("\n" + "="*70)
    print("RECOMMENDED LAYERS FOR HOOKING")
    print("="*70)

    if lang_layers is not None:
        num_layers = len(lang_layers)

        print("\nFor language model analysis:")
        print(f"  Last layer: '{layer_path}.{num_layers-1}'")
        print(f"  Second-to-last: '{layer_path}.{num_layers-2}'")
        print(f"  Middle layer: '{layer_path}.{num_layers//2}'")

        print("\nExample usage:")
        print("  from experiments.r1_concept_analyzer import R1ConceptAnalyzer")
        print("  analyzer = R1ConceptAnalyzer(r1_model, clip)")
        print(f"  analyzer.setup_layer_hook('{layer_path}.{num_layers-1}')")
    else:
        print("\nLanguage layers not found automatically.")
        print("For vision encoder analysis:")
        print("  analyzer.setup_layer_hook('model.visual.blocks.31')  # Last vision block")
        print("\nTo find language layers, check the ALL NAMED MODULES section below.")

    # Print all named modules
    print("\n" + "="*70)
    print("ALL NAMED MODULES (first 50):")
    print("="*70)
    all_modules = list(model.named_modules())
    for i, (name, module) in enumerate(all_modules[:50]):
        print(f"  {name}: {type(module).__name__}")
    if len(all_modules) > 50:
        print(f"  ... and {len(all_modules) - 50} more modules")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inspect R1-Onevision layer structure')
    parser.add_argument('--model', type=str, default='R1-Onevision-7B',
                       choices=['R1-Onevision-7B'],
                       help='R1-Onevision model variant')

    args = parser.parse_args()

    inspect_model_layers(args.model)
