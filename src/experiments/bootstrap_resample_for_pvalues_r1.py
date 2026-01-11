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
import pickle
from collections import Counter
from dataclasses import asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.r1_onevision import R1OnevisionAPI
from experiments.r1_concept_analyzer import R1ConceptAnalyzer
from interpretability.utils import compute_inner_products
from datasets.ddi import DDIDataLoader

from experiments.experiment_utils import (
    PromptConfig,
    ExperimentConfig,
    extract_concepts_from_reasoning,
    analyze_reasoning_concept_overlap,
    load_experiment_data
)

from open_clip import create_model_from_pretrained, get_tokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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
    def __init__(self, model_name='ViT-L-14', pretrained='openai', batch_size=32):
        self.model, self.preprocess = create_model_from_pretrained(model_name, pretrained=pretrained)
        self.tokenizer = get_tokenizer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def encode_images(self, image_paths):
        dataset = PathDataset(image_paths, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)

        embeddings = []
        for batch in tqdm(dataloader, desc="Getting image embeddings"):
            batch = batch.to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(batch)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings)

    def encode_text(self, texts):
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Getting text embeddings"):
            batch = texts[i:i + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_text(tokens)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings)

##
## Generate top concepts for DDI dataset with R1-Onevision
##

def run_single_seed_experiment(config_dict, df_preprocessed, random_seed, shared_data=None):
    """
    Run experiment with pre-configured dataframe for a single random seed using R1-Onevision.

    Args:
        config_dict: Experiment configuration
        df_preprocessed: DataFrame with 'label' column already set up
        random_seed: Random seed for this experiment
        shared_data: Pre-computed data that can be reused across seeds (similarity matrix, etc.)
    """

    config_dict = config_dict.copy()
    config_dict['random_state'] = random_seed

    results_dir = Path(config_dict['results_dir'])
    seed_dir = results_dir / f'seed_{random_seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading R1-Onevision model: {config_dict['model_name']}")
    r1_model = R1OnevisionAPI(model_name=config_dict['model_name'])

    print(f"Loading CLIP embedder...")
    clip = CLIPEmbedder()

    print(f"Setting up R1ConceptAnalyzer with layer hook...")
    analyzer = R1ConceptAnalyzer(r1_model, clip)

    layer_name = config_dict.get('layer_name', 'model.model.language_model.layers.27')
    analyzer.setup_layer_hook(layer_name)
    print(f"  Hooked layer: {layer_name}")

    # Load DDI data using preprocessed dataframe with current seed
    # Note: DDIDataLoader expects capitalized labels (Benign/Malignant)
    df_for_loader = df_preprocessed.copy()
    df_for_loader['label'] = df_for_loader['label'].str.capitalize()  # benign -> Benign, malignant -> Malignant

    data_loader = DDIDataLoader(
        metadata=df_for_loader,
        base_dir=config_dict['ddi_base_dir'],
        test_size=config_dict.get('test_size', 0.5),
        demo_size=config_dict.get('demo_size', 0.02),
        random_state=random_seed, 
        filter_skin_tone=config_dict.get('filter_skin_tone'),
    )

    print(f"Dataset info for seed {random_seed}:", data_loader.get_info())

    train_paths = [os.path.join(config_dict['ddi_base_dir'], path) for path in data_loader.train_df['DDI_file']]
    train_labels = data_loader.train_df['label'].values

    # Use identity preprocessor since R1 will handle image processing
    identity_preprocess = lambda x: x 
    train_dataset, _, _, _, _ = data_loader.get_datasets(
        image_processor=identity_preprocess,
        use_demos=False
    )

    print("Computing CLIP embeddings and similarity matrix...")

    concept_files = config_dict['concept_files']

    image_emb, text_emb, concept_texts = analyzer.get_embeddings(
        image_paths=train_paths,
        concept_files=concept_files
    )

    print(f"Loaded {len(concept_texts)} concepts from {len(concept_files)} files")

    sim_matrix = compute_inner_products(text_emb, image_emb)

    with open(seed_dir / 'concept_texts.json', 'w') as f:
        json.dump(concept_texts, f)

    np.save(seed_dir / 'similarity_matrix.npy', sim_matrix)
    np.save(seed_dir / 'image_emb.npy', image_emb)
    np.save(seed_dir / 'text_emb.npy', text_emb)

    print(f"Computing training set predictions and reasoning traces for seed {random_seed}...")

    choice_differences = []
    reasoning_traces = []

    for idx, image_path in enumerate(tqdm(train_paths, desc="Processing images")):
        try:
            result = r1_model(
                prompt=config_dict['prompt']['query_template'],
                image_paths=[image_path],
                max_new_tokens=2048,
                return_reasoning=True,
                system_prompt=config_dict['prompt']['system_prompt'],
                temperature=0.1, 
                do_sample=False
            )

            reasoning = result['reasoning']
            answer = result['answer']
            full_output = result['full_output']

            reasoning_traces.append(reasoning)

            if config_dict['task_definition'] == 'contrastive':
                choices = ['malignant', 'benign']
                logprobs = r1_model.get_choice_logprobs(
                    prompt=config_dict['prompt']['query_template'],
                    choices=[' ' + c for c in choices],
                    image_paths=[image_path]
                )
                choice_diff = logprobs[' malignant'] - logprobs[' benign']
            elif config_dict['task_definition'] == 'malignant_prob':
                logprobs = r1_model.get_choice_logprobs(
                    prompt=config_dict['prompt']['query_template'],
                    choices=[' malignant'],
                    image_paths=[image_path]
                )
                choice_diff = logprobs[' malignant']

            choice_differences.append(choice_diff)

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            choice_differences.append(0.0) 
            reasoning_traces.append("")

    choice_differences = np.array(choice_differences)
    np.save(seed_dir / 'choice_differences.npy', choice_differences)

    # Save reasoning traces
    with open(seed_dir / 'reasoning_traces.json', 'w') as f:
        json.dump(reasoning_traces, f)

    # === VCR PIPELINE: Collect activations and train concept model ===
    print("Collecting activations from hooked layer...")
    activations = analyzer.collect_activations(
        dataset=train_dataset,
        system_prompt=config_dict['prompt']['system_prompt'],
        query_template=config_dict['prompt']['query_template'],
        batch_size=1
    )

    print(f"Collected activations shape: {activations.shape}")

    # Train concept model: CLIP similarities → activations
    print("Training concept model (Ridge regression)...")
    concept_results = analyzer.train_concept_model(
        activations=activations,
        similarity_matrix=sim_matrix
    )
    print(f"  Concept model R² score: {concept_results['mean_r2']:.4f}")

    # Extract concept direction vectors
    concept_vectors = analyzer.extract_concept_vectors()
    print(f"  Concept vectors shape: {concept_vectors.shape}")

    # Compute concept weights (variance-based)
    sim_matrix_np = sim_matrix.cpu().numpy() if isinstance(sim_matrix, torch.Tensor) else sim_matrix
    concept_weights = analyzer.compute_concept_weights(sim_matrix)

    # === GRADIENT-BASED DIRECTIONAL DERIVATIVES ===
    print("Computing gradient-based directional derivatives...")
    weighted_sens, raw_sens = analyzer.calculate_directional_derivatives(
        dataset=train_dataset,
        concept_vectors=concept_vectors,
        concept_weights=concept_weights,
        system_prompt=config_dict['prompt']['system_prompt'],
        query_template=config_dict['prompt']['query_template'],
        target_completion=" malignant",
        task_score=config_dict['task_definition']
    )

    print(f"  Sensitivity shape: {weighted_sens.shape}")
    print(f"  Raw sensitivity range: [{raw_sens.min():.4f}, {raw_sens.max():.4f}]")
    print(f"  Weighted sensitivity range: [{weighted_sens.min():.4f}, {weighted_sens.max():.4f}]")

    # Save results for this seed (2D arrays: [num_samples, num_concepts])
    np.save(seed_dir / 'weighted_sens.npy', weighted_sens)
    np.save(seed_dir / 'raw_sens.npy', raw_sens)
    np.save(seed_dir / 'concept_weights.npy', concept_weights)

    # Analyze reasoning traces
    print("Analyzing reasoning trace concept overlap...")

    # Average sensitivities across samples to get per-concept scores
    # Shape: [num_samples, num_concepts] -> [num_concepts]
    avg_weighted_sens = np.mean(weighted_sens, axis=0)

    # Get top K concepts by average absolute weighted sensitivity
    top_k = config_dict.get('top_k_concepts', 20)
    top_concept_indices = np.argsort(np.abs(avg_weighted_sens))[-top_k:][::-1]
    top_concepts = [concept_texts[i] for i in top_concept_indices]

    # Analyze overlap with reasoning traces
    reasoning_analysis = analyze_reasoning_concept_overlap(
        reasoning_traces=reasoning_traces,
        top_concepts=top_concepts,
        all_concepts=concept_texts
    )

    # Save reasoning analysis
    with open(seed_dir / 'reasoning_analysis.json', 'w') as f:
        json.dump(reasoning_analysis, f, indent=2)

    print(f"\nReasoning Analysis Summary:")
    print(f"  Top {top_k} concepts by sensitivity: {top_concepts[:5]}...")
    print(f"  Concepts mentioned in reasoning: {list(reasoning_analysis['reasoning_concept_counts'].keys())[:5]}...")
    print(f"  Overlap: {reasoning_analysis['overlap_count']} / {top_k} ({reasoning_analysis['overlap_percentage']:.1f}%)")

    with open(seed_dir / 'exp_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Return concept texts, sensitivities, and reasoning analysis
    return concept_texts, weighted_sens, reasoning_analysis


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run VCR experiment with DDI dataset using R1-Onevision')
    parser.add_argument('--model', type=str, default='R1-Onevision-7B',
                       choices=['R1-Onevision-7B'],
                       help='R1-Onevision model to use')
    parser.add_argument('--filter_skin_tone', type=str, default='All',
                       help='Filter by skin tone: All, 12, or 56')
    parser.add_argument('--random_seeds', type=int, nargs='+', default=list(range(5)),
                       help='Random seeds to test (default: 5 seeds)')
    parser.add_argument('--task_definition', type=str, default='malignant_prob',
                       choices=['malignant_prob', 'contrastive'])
    parser.add_argument('--top_k_concepts', type=int, default=20,
                       help='Number of top concepts to analyze')
    parser.add_argument('--layer', type=str, default='model.language_model.layers.27',
                       help='Layer to hook for VCR analysis (default: last language layer)')
    args = parser.parse_args()

    # ===== EXPERIMENT CONFIGURATION =====

    # Random seeds to test
    random_seeds = args.random_seeds

    # Prompt configuration for R1-Onevision
    prompt_config = PromptConfig(
        system_prompt="You are a medical image analysis assistant specializing in dermatology. Analyze skin lesion images carefully.",
        query_template="Analyze this skin lesion image. Consider features like color, texture, borders, and symmetry. Determine if the lesion is benign or malignant.",
        use_demos=False  # R1 models don't use the same ICL format
    )

    # Data preprocessing function
    def preprocess_ddi_dataframe(df):
        """
        Preprocess the DDI dataframe to add 'label' column.
        """
        benign_label = "benign"
        malignant_label = "malignant"

        df['label'] = df['malignant'].map({
            False: benign_label,
            True: malignant_label
        })

        return df

    # Data splitting configuration
    data_config = {
        'test_size': 0.5,
        'demo_size': 0.02,
    }

    # Paths and model configuration
    skin_tone_suffix = f"_skin{args.filter_skin_tone}" if args.filter_skin_tone != 'All' else ''

    base_config = {
        'results_dir': f'{args.model}_DDI_R1{skin_tone_suffix}_{args.task_definition}',
        'model_name': args.model,
        'metadata_path': '/scratch/users/sonnet/ddi/ddi_metadata.csv',
        'ddi_base_dir': "/scratch/users/sonnet/ddi",
        'concept_files': ['/home/groups/roxanad/sonnet/vcr/src/concept_sets/google-10000-english-no-swears.txt'],
        'filter_skin_tone': None if args.filter_skin_tone == 'All' else int(args.filter_skin_tone),
        'task_definition': args.task_definition,
        'top_k_concepts': args.top_k_concepts,
        'layer_name': args.layer,  # Layer to hook for VCR analysis
    }

    # ===== END CONFIGURATION =====

    # Create main results directory
    results_dir = Path(base_config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess the dataframe once
    print("Loading and preprocessing DDI metadata...")
    df = pd.read_csv(base_config['metadata_path'], index_col=0)
    df_preprocessed = preprocess_ddi_dataframe(df)

    print(f"Preprocessed dataframe shape: {df_preprocessed.shape}")
    print(f"Label distribution: {df_preprocessed['label'].value_counts().to_dict()}")

    # Create experiment config
    exp_config = ExperimentConfig(
        prompt=prompt_config,
        **base_config,
        **data_config
    )

    exp_config_dict = asdict(exp_config)

    # Save overall experiment config
    with open(results_dir / 'experiment_config.json', 'w') as f:
        json.dump({
            'model': args.model,
            'random_seeds': random_seeds,
            'config': exp_config_dict
        }, f, indent=2)

    # Run experiments for each random seed
    all_sensitivities = []
    all_reasoning_analyses = []
    concept_texts = None

    for i, seed in enumerate(random_seeds):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(random_seeds)} with random seed: {seed}")
        print(f"Model: {args.model}")
        print(f"{'='*60}")

        # Run the experiment for this seed
        seed_concept_texts, sensitivities, reasoning_analysis = run_single_seed_experiment(
            exp_config_dict, df_preprocessed, seed, None
        )

        # Store results
        if concept_texts is None:
            concept_texts = seed_concept_texts
        all_sensitivities.append(sensitivities)
        all_reasoning_analyses.append(reasoning_analysis)

        print(f"Completed seed: {seed}")

    # Aggregate results across seeds
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print(f"{'='*60}")

    # Average sensitivity scores
    avg_sensitivities = np.mean(all_sensitivities, axis=0)
    std_sensitivities = np.std(all_sensitivities, axis=0)

    # Save aggregated results
    np.save(results_dir / 'avg_sensitivities.npy', avg_sensitivities)
    np.save(results_dir / 'std_sensitivities.npy', std_sensitivities)

    # Get overall top concepts
    top_k = args.top_k_concepts
    top_indices = np.argsort(np.abs(avg_sensitivities))[-top_k:][::-1]
    top_concepts_overall = [concept_texts[i] for i in top_indices]

    # Aggregate reasoning analysis
    overlap_counts = [ra['overlap_count'] for ra in all_reasoning_analyses]
    overlap_percentages = [ra['overlap_percentage'] for ra in all_reasoning_analyses]

    # Get most consistently mentioned concepts in reasoning
    all_reasoning_concepts = Counter()
    for ra in all_reasoning_analyses:
        all_reasoning_concepts.update(ra['reasoning_concept_counts'])

    aggregated_analysis = {
        'top_concepts_overall': top_concepts_overall,
        'avg_overlap_count': np.mean(overlap_counts),
        'std_overlap_count': np.std(overlap_counts),
        'avg_overlap_percentage': np.mean(overlap_percentages),
        'std_overlap_percentage': np.std(overlap_percentages),
        'most_common_reasoning_concepts': [c for c, _ in all_reasoning_concepts.most_common(50)],
        'reasoning_concept_frequencies': dict(all_reasoning_concepts.most_common(100)),
    }

    with open(results_dir / 'aggregated_reasoning_analysis.json', 'w') as f:
        json.dump(aggregated_analysis, f, indent=2)

    print(f"\nAggregated Reasoning Analysis:")
    print(f"  Top {top_k} concepts (by avg sensitivity): {top_concepts_overall[:5]}...")
    print(f"  Average overlap: {aggregated_analysis['avg_overlap_count']:.1f} ± {aggregated_analysis['std_overlap_count']:.1f}")
    print(f"  Average overlap %: {aggregated_analysis['avg_overlap_percentage']:.1f}% ± {aggregated_analysis['std_overlap_percentage']:.1f}%")
    print(f"  Most frequent in reasoning: {aggregated_analysis['most_common_reasoning_concepts'][:10]}")

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved in: {results_dir}")
    print(f"Analyzed {len(random_seeds)} random seeds with model: {args.model}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
