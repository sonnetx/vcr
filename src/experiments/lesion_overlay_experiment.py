import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from einops import repeat
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpretability.vcr import (
    ConceptAnalyzer,
    PromptTemplate
)
from models.flamingo import FlamingoAPI
from interpretability.utils import CLIPEmbedder, ImageDataset


@dataclass
class PromptConfig:
    """Configuration for prompt templates."""
    base_prompt: str = "Based on the image, this lesion is benign.<|endofchunk|>Based on the image, this lesion is malignant.<|endofchunk|>"
    demo_template: str = "<image>Based on the image, this lesion is {label}.<|endofchunk|>"
    query_template: str = "<image>Based on the image, this lesion is"
    completion: str = " malignant"
    use_demos: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for lesion overlay experiment."""
    results_dir: str
    model_name: str
    background_dir: str
    lesion_dir: str
    original_dir: str
    prompt: PromptConfig = field(default_factory=PromptConfig)
    task_definition: str = 'malignant_prob'  # 'malignant_prob' or 'contrastive'
    batch_size: int = 1


class OverlayDataset(Dataset):
    """Dataset for overlaid lesion+background images."""
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return {
            'image': self.preprocess(image),
            'image_path': str(self.image_paths[idx])
        }


def overlay_images(background_path: str, lesion_path: str, alpha: float = 0.5) -> Image.Image:
    """
    Overlay lesion image on background image.

    Args:
        background_path: Path to background image
        lesion_path: Path to lesion image
        alpha: Blending factor (0 = background only, 1 = lesion only)

    Returns:
        PIL Image of overlaid result
    """
    background = Image.open(background_path).convert('RGB')
    lesion = Image.open(lesion_path).convert('RGB')

    # Resize lesion to match background if needed
    if background.size != lesion.size:
        lesion = lesion.resize(background.size, Image.Resampling.LANCZOS)

    # Blend images
    overlaid = Image.blend(background, lesion, alpha)
    output_dir = 'overlaid_images'

    os.makedirs(output_dir, exist_ok=True) 

    background_filename = os.path.basename(background_path)
    lesion_filename = os.path.basename(lesion_path)

    background_name = background_filename.split('.')[0]
    lesion_name = lesion_filename.split('.')[0]

    final_filename = f"{background_name}_{lesion_name}.png"

    overlaid_filepath = os.path.join(output_dir, final_filename)

    overlaid.save(overlaid_filepath)

    return overlaid


def compute_image_difference(image1_path: str, image2_path: str) -> np.ndarray:
    """
    Compute pixel-wise difference between two images.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image

    Returns:
        Numpy array of pixel differences
    """
    img1 = np.array(Image.open(image1_path).convert('RGB'), dtype=np.float32)
    img2 = np.array(Image.open(image2_path).convert('RGB'), dtype=np.float32)

    # Resize if needed
    if img1.shape != img2.shape:
        img2_pil = Image.fromarray(img2.astype(np.uint8))
        img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS)
        img2 = np.array(img2_pil, dtype=np.float32)

    # Compute difference
    difference = np.abs(img1 - img2)
    return difference


def find_corresponding_original(lesion_path: str, original_dir: str) -> Optional[str]:
    """
    Find the corresponding original image for a lesion image by matching filename.

    Args:
        lesion_path: Path to lesion image
        original_dir: Directory containing original images

    Returns:
        Path to corresponding original image, or None if not found
    """
    lesion_name = Path(lesion_path).name
    original_path = Path(original_dir) / lesion_name

    if original_path.exists():
        return str(original_path)

    # Try without extension and look for any matching file
    lesion_stem = Path(lesion_path).stem
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        candidate = Path(original_dir) / f"{lesion_stem}{ext}"
        if candidate.exists():
            return str(candidate)

    return None


def compute_log_probabilities(
    analyzer: ConceptAnalyzer,
    image_batch: torch.Tensor,
    prompt_template: PromptTemplate,
    demo_paths: Optional[List[str]] = None,
    demo_labels: Optional[List[str]] = None,
    task_definition: str = 'malignant_prob'
) -> float:
    """
    Compute log probabilities for an image batch following run_single_seed_experiment logic.

    Args:
        analyzer: ConceptAnalyzer instance
        image_batch: Batch of images
        prompt_template: Prompt template
        demo_paths: Optional demo image paths
        demo_labels: Optional demo labels
        task_definition: 'malignant_prob' or 'contrastive'

    Returns:
        Log probability or log probability difference
    """
    # Prepare image batch format
    if len(image_batch.shape) == 4:
        image_batch = image_batch.unsqueeze(1).unsqueeze(2)

    # Add demo images if provided
    if demo_paths is not None:
        processed_imgs = analyzer.process_images_for_model(demo_paths)
        demo_batch = torch.stack(processed_imgs)
        stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=image_batch.shape[0])
        image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)

    # Build prompt
    prompt_batch = [prompt_template.build_prompt(demo_labels if demo_labels else None)]

    with torch.no_grad():
        if task_definition == 'contrastive':
            completion_a = ' malignant'
            completion_b = ' benign'

            # Compute log probabilities for both completions
            log_prob_a = analyzer.compute_model_outputs(
                image_batch, prompt_batch, completion_a
            ).item()

            log_prob_b = analyzer.compute_model_outputs(
                image_batch, prompt_batch, completion_b
            ).item()

            result = log_prob_a - log_prob_b

        elif task_definition == 'malignant_prob':
            completion = ' malignant'
            result = analyzer.compute_model_outputs(
                image_batch, prompt_batch, completion
            ).item()
        else:
            raise ValueError(f"Unknown task_definition: {task_definition}")

    return result


def run_lesion_overlay_experiment(config: ExperimentConfig):
    """
    Main experiment function that:
    1. Overlays lesion images on background images
    2. Computes log probabilities for each combination
    3. Stratifies by background image
    4. Computes differences with original images
    """
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / 'experiment_config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Load model and analyzer
    print(f"Loading model: {config.model_name}")
    clip = CLIPEmbedder()
    analyzer = ConceptAnalyzer(config.model_name, clip)

    # Build prompt template
    prompt_template = PromptTemplate(
        config.prompt.base_prompt,
        config.prompt.demo_template,
        config.prompt.query_template
    )

    # Get all background and lesion images
    background_dir = Path(config.background_dir)
    lesion_dir = Path(config.lesion_dir)
    original_dir = Path(config.original_dir)

    background_paths = sorted(list(background_dir.glob('*.jpg')) +
                             list(background_dir.glob('*.png')) +
                             list(background_dir.glob('*.jpeg')))
    lesion_paths = sorted(list(lesion_dir.glob('*.jpg')) +
                         list(lesion_dir.glob('*.png')) +
                         list(lesion_dir.glob('*.jpeg')))

    print(f"Found {len(background_paths)} background images")
    print(f"Found {len(lesion_paths)} lesion images")

    # Store results stratified by background
    results_by_background = defaultdict(list)
    overlay_cache_dir = results_dir / 'overlaid_images'
    overlay_cache_dir.mkdir(exist_ok=True)

    # Compute log probabilities for all combinations
    print("\nComputing log probabilities for overlaid images...")

    for bg_idx, bg_path in enumerate(tqdm(background_paths, desc="Background images")):
        bg_name = bg_path.stem

        for lesion_idx, lesion_path in enumerate(tqdm(lesion_paths, desc=f"Lesions (bg={bg_name})", leave=False)):
            lesion_name = lesion_path.stem

            # Create overlay
            overlaid = overlay_images(str(bg_path), str(lesion_path))

            # Save overlaid image
            overlay_filename = f"{bg_name}_X_{lesion_name}.png"
            overlay_path = overlay_cache_dir / overlay_filename
            overlaid.save(overlay_path)

            # Process image for model
            processed_image = analyzer.image_processor(overlaid)
            image_batch = processed_image.unsqueeze(0).cuda()

            # Compute log probability
            log_prob = compute_log_probabilities(
                analyzer,
                image_batch,
                prompt_template,
                demo_paths=None,
                demo_labels=None,
                task_definition=config.task_definition
            )

            # Store result
            results_by_background[bg_name].append({
                'background': bg_name,
                'lesion': lesion_name,
                'log_prob': log_prob,
                'overlay_path': str(overlay_path)
            })

    # Save stratified results
    print("\nSaving stratified results...")
    stratified_results_file = results_dir / 'stratified_results.json'
    with open(stratified_results_file, 'w') as f:
        json.dump({k: v for k, v in results_by_background.items()}, f, indent=2)

    # Convert to DataFrame for easier analysis
    all_results = []
    for bg_name, lesion_results in results_by_background.items():
        all_results.extend(lesion_results)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(results_dir / 'all_overlay_results.csv', index=False)

    # Compute summary statistics by background
    print("\nComputing summary statistics...")
    summary_stats = []
    for bg_name, lesion_results in results_by_background.items():
        log_probs = [r['log_prob'] for r in lesion_results]
        summary_stats.append({
            'background': bg_name,
            'num_lesions': len(log_probs),
            'mean_log_prob': np.mean(log_probs),
            'std_log_prob': np.std(log_probs),
            'min_log_prob': np.min(log_probs),
            'max_log_prob': np.max(log_probs)
        })

    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(results_dir / 'summary_by_background.csv', index=False)

    # Compute image differences
    print("\nComputing image differences with originals...")
    difference_results = []
    diff_cache_dir = results_dir / 'difference_maps'
    diff_cache_dir.mkdir(exist_ok=True)

    for lesion_path in tqdm(lesion_paths, desc="Processing lesion differences"):
        lesion_name = lesion_path.stem

        # Find corresponding original
        original_path = find_corresponding_original(str(lesion_path), str(original_dir))

        if original_path is None:
            print(f"Warning: No original found for {lesion_name}")
            continue

        # Compute lesion vs original difference
        lesion_diff = compute_image_difference(str(lesion_path), original_path)
        lesion_diff_path = diff_cache_dir / f"{lesion_name}_diff.npy"
        np.save(lesion_diff_path, lesion_diff)

        difference_results.append({
            'image_name': lesion_name,
            'image_type': 'lesion',
            'original_path': original_path,
            'difference_path': str(lesion_diff_path),
            'mean_diff': np.mean(lesion_diff),
            'max_diff': np.max(lesion_diff),
            'total_diff': np.sum(lesion_diff)
        })

    for bg_path in tqdm(background_paths, desc="Processing background differences"):
        bg_name = bg_path.stem

        # Find corresponding original
        original_path = find_corresponding_original(str(bg_path), str(original_dir))

        if original_path is None:
            print(f"Warning: No original found for {bg_name}")
            continue

        # Compute background vs original difference
        bg_diff = compute_image_difference(str(bg_path), original_path)
        bg_diff_path = diff_cache_dir / f"{bg_name}_diff.npy"
        np.save(bg_diff_path, bg_diff)

        difference_results.append({
            'image_name': bg_name,
            'image_type': 'background',
            'original_path': original_path,
            'difference_path': str(bg_diff_path),
            'mean_diff': np.mean(bg_diff),
            'max_diff': np.max(bg_diff),
            'total_diff': np.sum(bg_diff)
        })

    # Save difference results
    df_differences = pd.DataFrame(difference_results)
    df_differences.to_csv(results_dir / 'image_differences.csv', index=False)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved in: {results_dir}")
    print(f"Processed {len(background_paths)} backgrounds × {len(lesion_paths)} lesions")
    print(f"Computed {len(difference_results)} image differences")
    print(f"{'='*60}")

    return results_by_background, difference_results


def main():
    parser = argparse.ArgumentParser(description='Run lesion overlay experiment')
    parser.add_argument('--model', type=str, default='OpenFlamingo-3B-Instruct',
                       choices=['OpenFlamingo-3B-Instruct', 'OpenFlamingo-4B'],
                       help='Model to use')
    parser.add_argument('--background_dir', type=str, required=True,
                       help='Directory containing background images')
    parser.add_argument('--lesion_dir', type=str, required=True,
                       help='Directory containing lesion images')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory containing original unaltered images')
    parser.add_argument('--results_dir', type=str, default='lesion_overlay_results',
                       help='Directory to save results')
    parser.add_argument('--task_definition', type=str, default='malignant_prob',
                       choices=['malignant_prob', 'contrastive'],
                       help='Task definition for computing log probabilities')
    parser.add_argument('--use_demos', action='store_true',
                       help='Use in-context learning demos')

    args = parser.parse_args()

    # Prompt configuration
    prompt_config = PromptConfig(
        base_prompt=(
            "You are a medical image analysis assistant. For each skin lesion image, "
            "choose between Benign and Malignant.\n\nExamples:\n\n"
        ),
        demo_template="<image>\nThe lesion is {label}.\n\n",
        query_template="<image>\nThe lesion is",
        completion="Malignant",
        use_demos=args.use_demos
    )

    # Create experiment config
    exp_config = ExperimentConfig(
        results_dir=args.results_dir,
        model_name=args.model,
        background_dir=args.background_dir,
        lesion_dir=args.lesion_dir,
        original_dir=args.original_dir,
        prompt=prompt_config,
        task_definition=args.task_definition
    )

    # Run experiment
    run_lesion_overlay_experiment(exp_config)


if __name__ == '__main__':
    main()
