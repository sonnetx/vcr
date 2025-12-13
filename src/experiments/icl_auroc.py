import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
from tqdm import tqdm
import random
import pickle
import numpy as np
import pandas as pd
from models.flamingo import FlamingoAPI
from sklearn.metrics import roc_auc_score

class DDIExperiment:
    def __init__(
        self,
        base_directory: str = "/scratch/users/sonnet/ddi/",
        demo_metadata_path: str = "/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_demo.csv",
        test_metadata_path: str = "/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_test.csv",
        filter_rare: bool = False,
        flip_labels: bool = False,
    ):
        self.base_directory = base_directory
        self.demo_metadata_path = demo_metadata_path
        self.test_metadata_path = test_metadata_path
        self.filter_rare = filter_rare
        self.flip_labels = flip_labels
        self.rare_disease_list = ['subcutaneous-t-cell-lymphoma','focal-acral-hyperkeratosis', 'eccrine-poroma', 'inverted-follicular-keratosis', 'kaposi-sarcoma', 'metastatic-carcinoma','mycosis-fungoides','acquired-digital-fibrokeratoma', 'atypical-spindle-cell-nevus-of-reed', 'verruciform-xanthoma', 'morphea', 'nevus-lipomatosus-superficialis', 'pigmented-spindle-cell-nevus-of-reed', 'arteriovenous-hemangioma', 'syringocystadenoma-papilliferum', 'trichofolliculoma', 'coccidioidomycosis', 'leukemia-cutis', 'sebaceous-carcinoma', 'blastic-plasmacytoid-dendritic-cell-neoplasm', 'glomangioma', 'dermatomyositis', 'cellular-neurothekeoma', 'graft-vs-host-disease', 'xanthogranuloma', 'chondroid-syringoma', 'angioleiomyoma']

        
    def create_demo_balanced(self, num_malignant, skin_type, benign_multiplier=3, random_state=42):
        """Create demonstration examples with specified ratio of benign:malignant samples."""
        demo_frame = pd.read_csv(self.demo_metadata_path, index_col=0)
        if self.filter_rare:
            demo_frame = demo_frame[~demo_frame.disease.isin(self.rare_disease_list)]
        num_benign = benign_multiplier * num_malignant
        
        if skin_type == 'both':
            final_demo_frame = self.create_demo(num_benign, num_malignant, num_benign, num_malignant, random_state=random_state)
        elif skin_type == 12:
            final_demo_frame = self.create_demo(2*num_benign, 2*num_malignant, 0, 0, random_state=random_state)
        elif skin_type == 56:
            final_demo_frame = self.create_demo(0, 0, 2*num_benign, 2*num_malignant, random_state=random_state)

        return final_demo_frame

    def create_demo(self, fst12_ben, fst12_mal, fst56_ben, fst56_mal, random_state=42):
        """Create demonstration examples frame from the dataset."""
        demo_metadata = pd.read_csv(self.demo_metadata_path, index_col=0)
        total_samples = fst12_ben + fst12_mal + fst56_ben + fst56_mal
        
        selected_frames = []

        fst56_data = demo_metadata[demo_metadata.skin_tone == 56]
        
        fst56_mal_population = fst56_data[fst56_data.malignant == True]
        actual_fst56_mal_size = min(fst56_mal, len(fst56_mal_population))
        
        fst56_mal_frame = fst56_mal_population.sample(
            n=actual_fst56_mal_size, 
            random_state=random_state
        )
        selected_frames.append(fst56_mal_frame)

        fst56_ben_population = fst56_data[fst56_data.malignant == False]
        actual_fst56_ben_size = min(fst56_ben, len(fst56_ben_population))
        
        fst56_ben_frame = fst56_ben_population.sample(
            n=actual_fst56_ben_size, 
            random_state=random_state
        )
        selected_frames.append(fst56_ben_frame)

        fst12_data = demo_metadata[demo_metadata.skin_tone == 12]

        fst12_mal_population = fst12_data[fst12_data.malignant == True]
        actual_fst12_mal_size = min(fst12_mal, len(fst12_mal_population))
        
        fst12_mal_frame = fst12_mal_population.sample(
            n=actual_fst12_mal_size, 
            random_state=random_state
        )
        selected_frames.append(fst12_mal_frame)
        
        fst12_ben_population = fst12_data[fst12_data.malignant == False]
        actual_fst12_ben_size = min(fst12_ben, len(fst12_ben_population))
        
        fst12_ben_frame = fst12_ben_population.sample(
            n=actual_fst12_ben_size, 
            random_state=random_state
        )
        selected_frames.append(fst12_ben_frame)

        combined_frame = pd.concat(selected_frames)
        
        final_demo_frame = combined_frame.sample(
            n=len(combined_frame), 
            random_state=random_state
        )
        
        return final_demo_frame

    def run_experiment(
        self,
        num_malignant: int, 
        skin_type: str,
        output_dir: str,
        benign_multiplier: int = 3,
        random_state: int = 42,
        num_qns_per_round: int = 1,
        model="OpenFlamingo-4B",
    ):
        """Run a single experiment with specified parameters."""
        print('Building Model')
        api = FlamingoAPI(model_name=model)
        CHOICES = ["Benign", "Malignant"]
        EXP_NAME = f"ddi_NormalLabel_{num_malignant}_{skin_type}_b{benign_multiplier}_s{random_state}"
        
        demo_frame = self.create_demo_balanced(num_malignant, skin_type, benign_multiplier, random_state)
        test_df = pd.read_csv(self.test_metadata_path, index_col=0)
        if self.filter_rare:
            test_df = test_df[~test_df.disease.isin(self.rare_disease_list)]
        
        print(f"{EXP_NAME}, test size = {len(test_df)}")
        
        results = {}
        
        # no longer need to reshuffle test_df as they're being given 1 at a time anyway
#         test_df = test_df.sample(frac=1, random_state=66)
        
        # Create demo examples
        demo_examples = []
        for _, row in demo_frame.iterrows():
            if self.flip_labels:
                ans_choice = "Malignant" if not row.malignant else "Benign"
            else:
                ans_choice = "Benign" if not row.malignant else "Malignant"
            image_path = os.path.join(self.base_directory, row.DDI_file)
            demo_examples.append((image_path, ans_choice))
        
        # Predict test examples one at a time
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Normal Label Prediction"):
            
            # Base prompt
            prompt = (
                "You are a medical image analysis assistant. For each skin lesion image, "
                "choose between Benign and Malignant.\n\nExamples:\n\n"
            )
            
            # add demos
            image_paths = [ex[0] for ex in demo_examples]
            for demo in demo_examples:
                prompt += f"<image>\nThe lesion is {demo[1]}.\n\n"
            
#             qns_idx = []
#             for idx, row in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
#                 qns_idx.append(row.Index)

            # add test example
            image_paths.append(os.path.join(self.base_directory, row.DDI_file))
            prompt += f"<image>\nThe lesion is"
            
            prediction, logprobs = api.get_best_choice(prompt, CHOICES, image_paths)
#             print(f"Prediction: {prediction}")
#             print(f"Log probabilities: {logprobs}")

            results[idx] = {
                'prediction': prediction,
                'logprobs': logprobs,
                'prompt': prompt,
                'skin_tone': row.skin_tone,
                'ground_truth': 1 if row.malignant else 0,
                'image_paths': image_paths
            }

            # Save results periodically in experiment directory
            output_path = os.path.join(output_dir, f"{EXP_NAME}.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(results, f)
                
        # Calculate AUROC scores
        neg_class = "Benign"
        pos_class = "Malignant"

        # Convert predictions to probabilities
        def get_prob_positive(logprobs):
            prob_pos = np.exp(logprobs[pos_class]) / (np.exp(logprobs[pos_class]) + np.exp(logprobs[neg_class]))
            return prob_pos

        # Extract data
        all_probs = []
        all_labels = []
        fst12_probs = []
        fst12_labels = []
        fst56_probs = []
        fst56_labels = []

        for _, case in results.items():
            prob = get_prob_positive(case['logprobs'])
            label = case['ground_truth']
            skin_tone = case['skin_tone']

            all_probs.append(prob)
            all_labels.append(label)

            if skin_tone == 12:
                fst12_probs.append(prob)
                fst12_labels.append(label)
            else:
                fst56_probs.append(prob)
                fst56_labels.append(label)

        # Calculate AUROCs
        auroc_scores = {
            'all_auroc': roc_auc_score(all_labels, all_probs),
            'fst12_auroc': roc_auc_score(fst12_labels, fst12_probs),
            'fst56_auroc': roc_auc_score(fst56_labels, fst56_probs),
            'positive_class': pos_class
        }

        # Add AUROC scores to results
        results['auroc_scores'] = auroc_scores
        print("All AUROC: ", results['auroc_scores']['all_auroc'])
        print("FST12 AUROC: ", results['auroc_scores']['fst12_auroc'])
        print("FST56 AUROC: ", results['auroc_scores']['fst56_auroc'])
        
        return EXP_NAME, results
                
    def run_all_experiments(
        self,
        max_malignant: int = 5,
        random_seeds: range = range(5),
        skin_types: list = [12, 56, 'both'],
        base_output_dir: str = 'results',
        model='OpenFlamingo-4B'
    ):
        """Run full suite of experiments with different parameters."""
        
        results = {}
        
        # Create experiment directory
        exp_dir = os.path.join(base_output_dir, f"{model}_{''.join(str(s) for s in skin_types)}_{max_malignant}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Run experiments
        for num_malignant in range(0, max_malignant + 1, 5):
            for skin_type in skin_types:
                for seed in random_seeds:
                    exp_name, exp_results = self.run_experiment(
                        num_malignant=num_malignant,
                        skin_type=skin_type,
                        random_state=seed,
                        output_dir=exp_dir,
                        model=model
                    )
                    results[exp_name] = exp_results
                    
                    # Save updated results after each experiment
                    summary_path = os.path.join(exp_dir, "all_results_ddi.pkl")
                    with open(summary_path, "wb") as f:
                        pickle.dump(results, f)
        
        return results

def main():
    # Create base results directory
    base_output_dir = '/scratch/users/sonnet/vcr/icl_auroc'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize experiment
    experiment = DDIExperiment(filter_rare=False)
    
    models = ['OpenFlamingo-4B', 'OpenFlamingo-3B-Instruct']
    skin_types_exps = [12, 56, 'both']
    for model in models:
        for skin_type in skin_types_exps:
            experiment.run_all_experiments(
                max_malignant=20,
                random_seeds=range(3),
                skin_types=[skin_type],
                base_output_dir=base_output_dir,
                model=model
            )

if __name__ == "__main__":
    main()