"""
Generate reasoning traces and top significant concepts for manual inspection.

This script runs a single-seed analysis to output:
1. Reasoning traces for sample images
2. Top K concepts identified by gradient-based VCR
3. Side-by-side comparison for manual semantic overlap analysis

Usage:
    python src/experiments/inspect_reasoning_and_concepts.py \
        --model R1-Onevision-7B \
        --num_samples 20 \
        --top_k 30 \
        --output_file reasoning_concept_inspection.txt
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

# Compatibility patch for torch.compiler.is_compiling
# MUST be applied before any transformers imports
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.r1_onevision import R1OnevisionAPI
from experiments.r1_concept_analyzer import R1ConceptAnalyzer
from interpretability.utils import compute_inner_products
from datasets.ddi import DDIDataLoader

# Import CLIP embedder (defined locally in bootstrap script)
from open_clip import create_model_from_pretrained, get_tokenizer
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

class PathDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.preprocess(image)

class CLIPEmbedder:
    def __init__(self, model_name='hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'):
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode_images(self, image_paths, batch_size=32):
        dataset = PathDataset(image_paths, self.preprocess)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding images"):
                if torch.cuda.is_available():
                    batch = batch.cuda()
                embeddings = self.model.encode_image(batch)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_text(self, texts, batch_size=256):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch_texts = texts[i:i+batch_size]
            tokens = self.tokenizer(batch_texts)
            if torch.cuda.is_available():
                tokens = tokens.cuda()

            with torch.no_grad():
                embeddings = self.model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


def generate_inspection_report(
    model_name='R1-Onevision-7B',
    layer_name='model.model.language_model.layers.27',
    num_samples=20,
    top_k=30,
    output_file='reasoning_concept_inspection.txt',
    metadata_path='/scratch/users/sonnet/ddi/ddi_metadata.csv',
    ddi_base_dir='/scratch/users/sonnet/ddi',
    concept_files=['/home/groups/roxanad/sonnet/vcr/src/concept_sets/google-10000-english-no-swears.txt'],
    random_seed=42,
    task_definition='malignant_prob'
):
    """
    Generate inspection report with reasoning traces and top concepts.

    Args:
        model_name: R1-Onevision model variant
        layer_name: Layer to hook for VCR analysis
        num_samples: Number of sample images to inspect
        top_k: Number of top concepts to show
        output_file: Output text file path
        metadata_path: Path to DDI metadata CSV
        ddi_base_dir: Base directory for DDI dataset
        concept_files: List of concept file paths
        random_seed: Random seed for reproducibility
        task_definition: 'malignant_prob' or 'contrastive'
    """

    print("="*80)
    print("REASONING TRACE AND CONCEPT INSPECTION")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Layer: {layer_name}")
    print(f"Samples: {num_samples}")
    print(f"Top K concepts: {top_k}")
    print(f"Output: {output_file}")
    print("="*80)

    # Load models
    print("\n[1/7] Loading models...")
    r1_model = R1OnevisionAPI(model_name=model_name)
    clip = CLIPEmbedder()

    # Create analyzer and setup hook
    print("[2/7] Setting up VCR analyzer...")
    analyzer = R1ConceptAnalyzer(r1_model, clip)
    analyzer.setup_layer_hook(layer_name)
    print(f"  ✓ Hooked layer: {layer_name}")

    # Load data
    print("[3/7] Loading DDI dataset...")
    df = pd.read_csv(metadata_path, index_col=0)
    df['label'] = df['malignant'].map({False: 'Benign', True: 'Malignant'})

    data_loader = DDIDataLoader(
        metadata=df,
        base_dir=ddi_base_dir,
        test_size=0.5,
        random_state=random_seed,
    )

    print(f"  ✓ {data_loader.get_info()}")

    # Get image paths
    train_paths = [os.path.join(ddi_base_dir, path) for path in data_loader.train_df['DDI_file']]
    train_labels = data_loader.train_df['label'].values

    # Limit to num_samples
    sample_paths = train_paths[:num_samples]
    sample_labels = train_labels[:num_samples]

    # Get train dataset for VCR pipeline
    # Use identity preprocessor since R1 will handle image processing
    identity_preprocess = lambda x: x  # Just return the PIL image as-is
    train_dataset, _, _, _, _ = data_loader.get_datasets(
        image_processor=identity_preprocess,
        use_demos=False
    )

    # Get CLIP embeddings
    print("[4/7] Computing CLIP embeddings and similarities...")
    image_emb, text_emb, concept_texts = analyzer.get_embeddings(
        image_paths=train_paths,
        concept_files=concept_files
    )
    sim_matrix = compute_inner_products(text_emb, image_emb)
    print(f"  ✓ {len(concept_texts)} concepts, {len(train_paths)} images")

    # Run full VCR pipeline
    print("[5/7] Running VCR pipeline (this may take a while)...")

    # Collect activations
    print("  - Collecting activations...")
    activations = analyzer.collect_activations(
        dataset=train_dataset,
        system_prompt="You are a medical image analysis assistant specializing in dermatology. Analyze skin lesion images carefully.",
        query_template="Analyze this skin lesion image. Consider features like color, texture, borders, and symmetry. Determine if the lesion is benign or malignant.",
        batch_size=1
    )

    # Train concept model
    print("  - Training concept model...")
    concept_results = analyzer.train_concept_model(
        activations=activations,
        similarity_matrix=sim_matrix
    )
    print(f"    R² score: {concept_results['mean_r2']:.4f}")

    # Extract concept vectors
    concept_vectors = analyzer.extract_concept_vectors()
    concept_weights = analyzer.compute_concept_weights(sim_matrix)

    # Calculate directional derivatives
    print("  - Computing gradient-based sensitivities...")
    weighted_sens, raw_sens = analyzer.calculate_directional_derivatives(
        dataset=train_dataset,
        concept_vectors=concept_vectors,
        concept_weights=concept_weights,
        system_prompt="You are a medical image analysis assistant specializing in dermatology. Analyze skin lesion images carefully.",
        query_template="Analyze this skin lesion image. Consider features like color, texture, borders, and symmetry. Determine if the lesion is benign or malignant.",
        target_completion=" malignant",
        task_score=task_definition
    )
    print(f"    Sensitivity range: [{raw_sens.min():.4f}, {raw_sens.max():.4f}]")
    print(f"    Sensitivity shape: {weighted_sens.shape}")

    # Average sensitivities across samples to get per-concept scores
    # Shape: [num_samples, num_concepts] -> [num_concepts]
    avg_weighted_sens = np.mean(weighted_sens, axis=0)
    avg_raw_sens = np.mean(raw_sens, axis=0)
    print(f"    Averaged sensitivity shape: {avg_weighted_sens.shape}")

    # Get top concepts based on average absolute weighted sensitivity
    top_indices = np.argsort(np.abs(avg_weighted_sens))[-top_k:][::-1]
    top_concepts = [(concept_texts[i], avg_weighted_sens[i], avg_raw_sens[i]) for i in top_indices]

    # Generate reasoning traces for sample images
    print(f"[6/7] Generating reasoning traces for {num_samples} sample images...")
    reasoning_traces = []

    for idx, image_path in enumerate(tqdm(sample_paths, desc="Processing samples")):
        try:
            result = r1_model(
                prompt="Analyze this skin lesion image. Consider features like color, texture, borders, and symmetry. Determine if the lesion is benign or malignant.",
                image_paths=[image_path],
                max_new_tokens=2048,
                return_reasoning=True,
                system_prompt="You are a medical image analysis assistant specializing in dermatology. Analyze skin lesion images carefully.",
                temperature=0.1,
                do_sample=False
            )

            reasoning_traces.append({
                'index': idx,
                'image_path': image_path,
                'label': sample_labels[idx],
                'reasoning': result['reasoning'],
                'answer': result['answer']
            })
        except Exception as e:
            print(f"  Error processing image {idx}: {e}")
            reasoning_traces.append({
                'index': idx,
                'image_path': image_path,
                'label': sample_labels[idx],
                'reasoning': f"[ERROR: {e}]",
                'answer': "[ERROR]"
            })

    # Write inspection report
    print(f"[7/7] Writing inspection report to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("REASONING TRACE AND CONCEPT INSPECTION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {model_name}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Task: {task_definition}\n")
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Concept model R²: {concept_results['mean_r2']:.4f}\n")
        f.write(f"Total concepts: {len(concept_texts)}\n")
        f.write(f"Total training images: {len(train_paths)}\n")
        f.write(f"Sample images inspected: {num_samples}\n\n")

        # Top K concepts
        f.write("="*80 + "\n")
        f.write(f"TOP {top_k} CONCEPTS (by gradient-based sensitivity)\n")
        f.write("="*80 + "\n\n")

        f.write("Rank | Concept                          | Weighted Sens | Raw Sens\n")
        f.write("-"*80 + "\n")
        for rank, (concept, w_sens, r_sens) in enumerate(top_concepts, 1):
            f.write(f"{rank:4d} | {concept:32s} | {w_sens:13.6f} | {r_sens:8.6f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("REASONING TRACES FOR SAMPLE IMAGES\n")
        f.write("="*80 + "\n\n")

        # Reasoning traces
        for trace in reasoning_traces:
            f.write("-"*80 + "\n")
            f.write(f"SAMPLE {trace['index'] + 1}/{num_samples}\n")
            f.write("-"*80 + "\n")
            f.write(f"Image: {os.path.basename(trace['image_path'])}\n")
            f.write(f"True Label: {trace['label']}\n")
            f.write(f"Model Answer: {trace['answer']}\n\n")

            f.write("REASONING:\n")
            f.write(trace['reasoning'] + "\n\n")

        # Instructions for manual analysis
        f.write("\n" + "="*80 + "\n")
        f.write("MANUAL INSPECTION GUIDE\n")
        f.write("="*80 + "\n\n")

        f.write("For each reasoning trace, manually check for semantic overlap with top concepts:\n\n")
        f.write("1. DIRECT MATCHES: Does the reasoning mention the exact concept words?\n")
        f.write("   Example: If top concept is 'asymmetry', does reasoning mention 'asymmetry'?\n\n")

        f.write("2. SEMANTIC OVERLAP: Does the reasoning discuss related ideas?\n")
        f.write("   Example: Top concept 'irregular' vs reasoning mentions 'uneven borders'\n\n")

        f.write("3. IMPLICIT USAGE: Does the reasoning use the concept without naming it?\n")
        f.write("   Example: Top concept 'color' vs reasoning describes 'dark brown areas'\n\n")

        f.write("4. PATTERNS TO LOOK FOR:\n")
        f.write("   - Medical terminology vs everyday language (e.g., 'asymmetric' vs 'lopsided')\n")
        f.write("   - Visual features (color, shape, texture) mentioned in both\n")
        f.write("   - Diagnostic criteria (ABCDE rule: Asymmetry, Border, Color, Diameter, Evolving)\n\n")

        f.write("5. QUESTIONS TO ASK:\n")
        f.write("   - Are top concepts actually used by the model's reasoning?\n")
        f.write("   - Are there important concepts in reasoning NOT in the top list?\n")
        f.write("   - Do high-sensitivity concepts align with dermatology best practices?\n\n")

        f.write("="*80 + "\n")
        f.write("TOP CONCEPTS (for easy reference during manual inspection)\n")
        f.write("="*80 + "\n\n")

        # Print top concepts in columns for easy reference
        concepts_per_row = 4
        for i in range(0, len(top_concepts), concepts_per_row):
            row_concepts = top_concepts[i:i+concepts_per_row]
            row_text = " | ".join([f"{rank:2d}. {concept:20s}" for rank, (concept, _, _) in
                                   enumerate(row_concepts, start=i+1)])
            f.write(row_text + "\n")

    print(f"\n✓ Inspection report written to: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Open {output_file} in a text editor")
    print(f"2. Manually review reasoning traces for semantic overlap with top concepts")
    print(f"3. Note patterns, mismatches, and interesting findings")
    print(f"4. Use insights to refine concept selection or analysis approach")

    # Also save structured data for programmatic analysis
    output_json = output_file.replace('.txt', '.json')
    structured_data = {
        'model': model_name,
        'layer': layer_name,
        'task_definition': task_definition,
        'random_seed': random_seed,
        'concept_model_r2': float(concept_results['mean_r2']),
        'top_concepts': [
            {
                'rank': rank,
                'concept': concept,
                'weighted_sensitivity': float(w_sens),
                'raw_sensitivity': float(r_sens)
            }
            for rank, (concept, w_sens, r_sens) in enumerate(top_concepts, 1)
        ],
        'reasoning_traces': reasoning_traces
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2)

    print(f"\n✓ Structured data saved to: {output_json}")

    return output_file, output_json


def main():
    parser = argparse.ArgumentParser(
        description='Generate reasoning traces and top concepts for manual inspection'
    )

    parser.add_argument('--model', type=str, default='R1-Onevision-7B',
                       choices=['R1-Onevision-7B'],
                       help='R1-Onevision model variant')
    parser.add_argument('--layer', type=str, default='model.language_model.layers.27',
                       help='Layer to hook for VCR analysis')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of sample images to inspect')
    parser.add_argument('--top_k', type=int, default=30,
                       help='Number of top concepts to show')
    parser.add_argument('--output_file', type=str, default='reasoning_concept_inspection.txt',
                       help='Output text file path')
    parser.add_argument('--metadata_path', type=str,
                       default='/scratch/users/sonnet/ddi/ddi_metadata.csv',
                       help='Path to DDI metadata CSV')
    parser.add_argument('--ddi_base_dir', type=str,
                       default='/scratch/users/sonnet/ddi',
                       help='Base directory for DDI dataset')
    parser.add_argument('--concept_files', type=str, nargs='+',
                       default=['/home/groups/roxanad/sonnet/vcr/src/concept_sets/google-10000-english-no-swears.txt'],
                       help='Paths to concept files')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--task_definition', type=str, default='malignant_prob',
                       choices=['malignant_prob', 'contrastive'],
                       help='Task definition for sensitivity calculation')

    args = parser.parse_args()

    generate_inspection_report(
        model_name=args.model,
        layer_name=args.layer,
        num_samples=args.num_samples,
        top_k=args.top_k,
        output_file=args.output_file,
        metadata_path=args.metadata_path,
        ddi_base_dir=args.ddi_base_dir,
        concept_files=args.concept_files,
        random_seed=args.random_seed,
        task_definition=args.task_definition
    )


if __name__ == '__main__':
    main()
