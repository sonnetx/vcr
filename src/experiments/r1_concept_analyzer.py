"""
Adapted ConceptAnalyzer for R1-Onevision models.

R1-Onevision is based on Qwen2.5-VL, which has a different architecture than Flamingo.
This module adapts the VCR approach to work with the R1 model's layer structure.
"""

import torch

# Compatibility patch for torch.compiler.is_compiling
# MUST be applied before any model operations
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from einops import repeat
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import torch.nn.functional as F

class R1ConceptAnalyzer:
    """
    ConceptAnalyzer adapted for R1-Onevision models.

    Key differences from Flamingo ConceptAnalyzer:
    - R1 uses Qwen2.5-VL architecture 
    - Different layer naming convention
    - Chat-based prompt format 
    """

    def __init__(self, r1_model, clip_embedder):
        """
        Initialize analyzer with R1-Onevision model and CLIP embedder.

        Args:
            r1_model: R1OnevisionAPI instance
            clip_embedder: CLIPEmbedder instance for concept similarities
        """
        # Keep naming consistent with original ConceptAnalyzer
        self.model = r1_model  # R1OnevisionAPI wrapper (analogous to FlamingoAPI)
        self.model_name = r1_model.model_id if hasattr(r1_model, 'model_id') else 'R1-Onevision'
        self.clip = clip_embedder
        self.image_processor = r1_model.processor  # Use same name as original

        # Layer hooking - match original naming where applicable
        self.wrapped_layer = None  # The hooked module (matches original)
        self.concept_model = None  # Trained Ridge model
        self.concept_vectors = None  # Extracted concept vectors

        # R1-specific: hook management (not in original but needed for R1's approach)
        self._activations = []  # Temporary storage for activations
        self._hook_handle = None  # Current hook handle

    def get_layer_names(self):
        """
        Get available layer names in R1-Onevision model.

        R1-Onevision (Qwen2.5-VL) structure:
        - model.visual (vision encoder)
        - model.model.layers[i] (language model layers)

        Returns:
            List of layer names
        """
        layers = []

        # Access the underlying Qwen model from the API wrapper
        base_model = self.model.model

        # Language model layers (similar to Flamingo's lang_encoder)
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            num_layers = len(base_model.model.layers)
            for i in range(num_layers):
                layers.append(f'model.model.layers.{i}')

        # Vision encoder layers
        if hasattr(base_model, 'visual'):
            if hasattr(base_model.visual, 'blocks'):
                num_visual_layers = len(base_model.visual.blocks)
                for i in range(num_visual_layers):
                    layers.append(f'model.visual.blocks.{i}')

        return layers

    def setup_layer_hook(self, layer_name):
        """
        Set up activation hook on specified layer.

        Args:
            layer_name: Name of layer to hook (e.g., 'model.model.layers.31')
        """
        # Remove existing hook if any
        if self._hook_handle is not None:
            self._hook_handle.remove()

        self._activations = []

        # Access the underlying Qwen model from the API wrapper
        base_model = self.model.model
        if hasattr(base_model, 'module'):
            base_model = base_model.module

        module = base_model
        try:
            for attr in layer_name.split('.'):
                if attr.isdigit():
                    module = module[int(attr)]
                else:
                    module = getattr(module, attr)
        except AttributeError as e:
            print(f"Error: Could not find layer '{layer_name}'")
            print(f"Available attributes at this level: {[a for a in dir(module) if not a.startswith('_')][:20]}")
            print(f"\nTo find valid layer names, run:")
            print(f"  python src/experiments/inspect_r1_layers.py --model {self.model.model_id}")
            raise AttributeError(f"Layer '{layer_name}' not found in model. {e}")

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]  # Usually first element is the hidden states
            else:
                act = output

            self._activations.append(act.detach().cpu())

        self._hook_handle = module.register_forward_hook(hook_fn)
        self.wrapped_layer = module  # Store the wrapped module (matches original naming)
        self.layer_name = layer_name

        print(f"Hook registered on layer: {layer_name}")

    def get_embeddings(self, image_paths, concept_files):
        """
        Get CLIP embeddings for images and concepts.

        Args:
            image_paths: List of paths to images
            concept_files: List of paths to concept text files

        Returns:
            image_emb, text_emb, concept_texts
        """
        concept_texts = []
        for concept_file in concept_files:
            with open(concept_file, 'r') as f:
                concepts = [line.strip() for line in f if line.strip()]
                concept_texts.extend(concepts)

        image_emb = self.clip.encode_images(image_paths)
        text_emb = self.clip.encode_text(concept_texts)

        return image_emb, text_emb, concept_texts

    def collect_activations(self, dataset, system_prompt, query_template, batch_size=1):
        """
        Collect activations from the hooked layer.

        Args:
            dataset: Dataset with images
            system_prompt: System prompt for R1
            query_template: Query template for R1
            batch_size: Batch size (usually 1 for R1)

        Returns:
            Array of activations
        """
        if self.wrapped_layer is None:
            raise ValueError("Must call setup_layer_hook before collecting activations")

        # Reset activations list (the hook from setup_layer_hook will populate this)
        self._activations = []

        # Custom collate function to handle PIL images
        def collate_fn(batch):
            """Custom collate that doesn't try to stack PIL images"""
            return {
                'image': [item['image'] for item in batch],
                'label': [item['label'] for item in batch],
                'image_path': [item['image_path'] for item in batch]
            }

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Access the underlying model from the API wrapper
        base_model = self.model.model
        base_model.eval()

        for batch in tqdm(dataloader, desc="Collecting activations"):
            image_batch = batch['image']

            with torch.no_grad():
                # Check if qwen_vl_utils is available to determine how to handle images
                try:
                    from qwen_vl_utils import process_vision_info
                    use_qwen_vl_utils = True
                except ImportError:
                    use_qwen_vl_utils = False

                if use_qwen_vl_utils:
                    # Use image paths for process_vision_info (expects strings)
                    image_paths = batch['image_path']

                    messages = [{
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    }, {
                        "role": "user",
                        "content": []
                    }]

                    for img_path in image_paths:
                        messages[1]["content"].append({"type": "image", "image": img_path})

                    messages[1]["content"].append({"type": "text", "text": query_template})

                    text = self.image_processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.image_processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(base_model.device)
                else:
                    # Use PIL Images directly for processor
                    if isinstance(image_batch, torch.Tensor):
                        images = []
                        for img_tensor in image_batch:
                            img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
                            img_array = (img_array * 255).astype('uint8')
                            images.append(Image.fromarray(img_array))
                    else:
                        images = image_batch

                    messages = [{
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    }, {
                        "role": "user",
                        "content": []
                    }]

                    for img in images:
                        messages[1]["content"].append({"type": "image", "image": img})

                    messages[1]["content"].append({"type": "text", "text": query_template})

                    inputs = self.image_processor(
                        text=messages,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                    ).to(base_model.device)

                _ = base_model(**inputs)

        # Process activations: extract last token from each before concatenating
        # This handles variable sequence lengths across batches
        processed_activations = []
        for act in self._activations:
            # Shape: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
            if len(act.shape) == 3:
                # Take last token for each sample in batch
                last_token_act = act[:, -1, :]
            else:
                last_token_act = act
            processed_activations.append(last_token_act)

        # Now concatenate - all tensors are (batch, hidden_dim)
        activations_array = torch.cat(processed_activations, dim=0)

        # Convert to float32 before converting to numpy (BFloat16 not supported by numpy)
        if activations_array.dtype == torch.bfloat16:
            activations_array = activations_array.float()

        return activations_array.numpy()

    def train_concept_model(self, activations, similarity_matrix, alpha=1.0):
        """
        Train linear model to predict concept similarities from activations.

        Args:
            activations: Array of shape (n_samples, hidden_dim)
            similarity_matrix: Tensor of shape (n_concepts, n_samples)
            alpha: Ridge regression regularization parameter

        Returns:
            Dictionary with training results
        """
        # Convert inputs to numpy if needed
        if isinstance(activations, torch.Tensor):
            X = activations.numpy()
        else:
            X = activations

        if isinstance(similarity_matrix, torch.Tensor):
            Y = similarity_matrix.T.numpy()  # Transpose to [samples, concepts]
        else:
            Y = similarity_matrix.T

        print(f"Training concept model: X={X.shape}, Y={Y.shape}")

        # Train Ridge model to predict similarities from activations
        model = Ridge(alpha=alpha)
        model.fit(X, Y)

        # Predict and evaluate
        Y_pred = model.predict(X)

        # Calculate R² for each concept
        r2_scores = []
        for i in range(Y.shape[1]):
            r2 = r2_score(Y[:, i], Y_pred[:, i])
            r2_scores.append(r2)

        r2_scores = np.array(r2_scores)

        print(f"Mean R² score: {np.mean(r2_scores):.4f}")
        print(f"Median R² score: {np.median(r2_scores):.4f}")

        self.concept_model = model
        self.r2_scores = r2_scores

        return {
            'model': model,
            'r2_scores': r2_scores,
            'mean_r2': np.mean(r2_scores),
            'predictions': Y_pred
        }

    def extract_concept_vectors(self):
        """
        Extract and normalize concept vectors from trained model.

        Returns:
            Tensor of shape (n_concepts, hidden_dim)
        """
        if not hasattr(self, 'concept_model'):
            raise ValueError("Must train concept model first")

        # Ridge with multiple outputs stores coefficients as a matrix
        coef_matrix = self.concept_model.coef_

        # For Ridge with multiple outputs, coef_ has shape (n_targets, n_features)
        if len(coef_matrix.shape) == 1:
            # Single output case
            concept_vectors = coef_matrix.reshape(1, -1)
        else:
            # Multiple output case: coef_matrix is already (n_concepts, n_features)
            concept_vectors = coef_matrix

        # Normalize each concept vector to unit length
        norms = np.linalg.norm(concept_vectors, axis=1, keepdims=True)
        concept_vectors = concept_vectors / norms

        self.concept_vectors = torch.tensor(concept_vectors, dtype=torch.float32)
        return self.concept_vectors

    def compute_concept_weights(self, similarity_matrix, weight_type='variance'):
        """
        Compute importance weights for concepts.

        Args:
            similarity_matrix: Tensor of concept similarities [concepts, samples]
            weight_type: Type of weighting ('variance', 'uniform', etc.)

        Returns:
            Tensor of concept weights
        """
        # Convert to numpy if needed
        if isinstance(similarity_matrix, torch.Tensor):
            Y = similarity_matrix.T.numpy()  # [samples, concepts]
        else:
            Y = similarity_matrix.T

        if weight_type == 'variance':
            weights = np.var(Y, axis=0)
        elif weight_type == 'uniform':
            weights = np.ones(Y.shape[1])
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")

        return torch.tensor(weights, dtype=torch.float32)

    def calculate_directional_derivatives(
        self,
        dataset,
        concept_vectors,
        concept_weights,
        system_prompt,
        query_template,
        target_completion=" malignant",
        task_score='malignant_prob'
    ):
        """
        Calculate sensitivity of model output to concept directions using gradients.

        This follows the same approach as the original VCR paper:
        1. Forward pass to get activation h at hooked layer
        2. Compute log P(target_completion)
        3. Compute gradient: ∂log P(y) / ∂h
        4. Project gradient onto concept directions: sensitivity_c = gradient · concept_vector_c

        Args:
            dataset: Dataset with images
            concept_vectors: Array of shape (n_concepts, hidden_dim)
            concept_weights: Array of concept importance weights
            system_prompt: System prompt for R1
            query_template: Query template
            target_completion: Target token(s) to measure probability for
            task_score: 'malignant_prob' or 'contrastive'

        Returns:
            weighted_sensitivities, raw_sensitivities
        """
        if self.wrapped_layer is None:
            raise ValueError("Must call setup_layer_hook first")

        # Access the underlying model from the API wrapper
        base_model = self.model.model

        # Move to device and convert to model's dtype (R1 uses BFloat16)
        # Get model dtype from first parameter
        model_dtype = next(base_model.parameters()).dtype

        concept_vectors = concept_vectors.to(device=base_model.device, dtype=model_dtype)
        concept_weights = concept_weights.to(device=base_model.device, dtype=model_dtype)

        all_raw_sensitivities = []
        all_weighted_sensitivities = []

        base_model.eval()

        for batch in tqdm(dataset, desc="Computing directional derivatives"):
            torch.cuda.empty_cache()

            # Use image path when available (for process_vision_info)
            # Otherwise convert to PIL Image
            if 'image_path' in batch:
                image = batch['image_path']
            elif isinstance(batch['image'], torch.Tensor):
                img_tensor = batch['image']
                img_array = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                img_array = (img_array * 255).astype('uint8')
                image = Image.fromarray(img_array)
            else:
                image = batch['image']

            if task_score == 'contrastive':
                # === Contrastive: compute gradient for both completions ===

                # Forward pass 1: malignant
                # Register temporary hook that doesn't detach (for gradient computation)
                layer_outputs_malignant = []
                def hook_fn_malignant(module, input, output):
                    if isinstance(output, tuple):
                        layer_outputs_malignant.append(output[0])
                    else:
                        layer_outputs_malignant.append(output)

                hook_malignant = self.wrapped_layer.register_forward_hook(hook_fn_malignant)

                log_prob_malignant = self._forward_with_completion(
                    image, system_prompt, query_template, " malignant"
                )

                # Get gradient w.r.t. activation
                activation_malignant = layer_outputs_malignant[-1]

                grad_malignant = torch.autograd.grad(
                    outputs=log_prob_malignant,
                    inputs=activation_malignant,
                    create_graph=False,
                    retain_graph=False
                )[0]

                hook_malignant.remove()

                # Clear intermediate tensors to free memory
                del log_prob_malignant

                # Forward pass 2: benign
                layer_outputs_benign = []
                def hook_fn_benign(module, input, output):
                    if isinstance(output, tuple):
                        layer_outputs_benign.append(output[0])
                    else:
                        layer_outputs_benign.append(output)

                hook_benign = self.wrapped_layer.register_forward_hook(hook_fn_benign)

                log_prob_benign = self._forward_with_completion(
                    image, system_prompt, query_template, " benign"
                )

                activation_benign = layer_outputs_benign[-1]

                grad_benign = torch.autograd.grad(
                    outputs=log_prob_benign,
                    inputs=activation_benign,
                    create_graph=False,
                    retain_graph=False
                )[0]

                hook_benign.remove()

                # Clear intermediate tensors to free memory
                del log_prob_benign

                # Contrastive gradient
                contrastive_grad = grad_malignant - grad_benign

                # Flatten gradient to [seq_len, hidden_dim] (matching original)
                # Shape: [batch, seq_len, hidden_dim] -> [seq_len, hidden_dim]
                flattened_grad = contrastive_grad.view(contrastive_grad.size(1), -1)

                # Clean up
                del activation_malignant, activation_benign, grad_malignant, grad_benign, contrastive_grad
                del layer_outputs_malignant[:], layer_outputs_benign[:]
                torch.cuda.empty_cache()

            elif task_score == 'malignant_prob':
                # === Single completion: compute gradient for target only ===

                # Register temporary hook that doesn't detach (for gradient computation)
                layer_outputs = []
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        layer_outputs.append(output[0])
                    else:
                        layer_outputs.append(output)

                hook = self.wrapped_layer.register_forward_hook(hook_fn)

                log_prob = self._forward_with_completion(
                    image, system_prompt, query_template, target_completion
                )

                activation = layer_outputs[-1]

                grad = torch.autograd.grad(
                    outputs=log_prob,
                    inputs=activation,
                    create_graph=False,
                    retain_graph=False
                )[0]

                hook.remove()

                # Clear intermediate tensors to free memory
                del log_prob

                # Flatten gradient to [seq_len, hidden_dim] (matching original)
                # Shape: [batch, seq_len, hidden_dim] -> [seq_len, hidden_dim]
                flattened_grad = grad.view(grad.size(1), -1)

                # Clean up
                del activation, layer_outputs[:]
                torch.cuda.empty_cache()

            # Compute directional derivatives: gradient · concept_vector
            # [seq_len, hidden_dim] @ [n_concepts, hidden_dim].T = [seq_len, n_concepts]
            raw_sensitivities = torch.matmul(flattened_grad, concept_vectors.T)
            weighted_sensitivities = raw_sensitivities * concept_weights.unsqueeze(0)

            # Store results for final token (like original VCR implementation)
            # Take last token: [seq_len, n_concepts] -> [n_concepts]
            # Convert to float32 before numpy (BFloat16 not supported)
            raw_last_token = raw_sensitivities[-1, :].cpu().detach()
            if raw_last_token.dtype == torch.bfloat16:
                raw_last_token = raw_last_token.float()
            all_raw_sensitivities.append(raw_last_token.numpy())

            weighted_last_token = weighted_sensitivities[-1, :].cpu().detach()
            if weighted_last_token.dtype == torch.bfloat16:
                weighted_last_token = weighted_last_token.float()
            all_weighted_sensitivities.append(weighted_last_token.numpy())

            if 'grad' in locals():
                del grad
            del raw_sensitivities, weighted_sensitivities
            torch.cuda.empty_cache()

        # Stack results (matching original implementation)
        all_weighted = np.vstack(all_weighted_sensitivities)
        all_raw = np.vstack(all_raw_sensitivities)

        return all_weighted, all_raw

    def _forward_with_completion(self, image, system_prompt, query_template, completion):
        """
        Forward pass to compute log probability of completion.

        Args:
            image: PIL Image or image path (string)
            system_prompt: System prompt
            query_template: Query template
            completion: Completion to measure probability for

        Returns:
            Log probability tensor (requires_grad=True)
        """
        full_text = query_template + completion

        messages = [{
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }, {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_text}
            ]
        }]

        # Process inputs
        base_model = self.model.model
        try:
            from qwen_vl_utils import process_vision_info

            text = self.image_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # We're providing the completion
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.image_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(base_model.device)
        except ImportError:
            # If image is a path, load it as PIL Image
            if isinstance(image, str):
                image = Image.open(image)
            inputs = self.image_processor(
                text=messages,
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(base_model.device)

        outputs = base_model(**inputs)
        logits = outputs.logits

        # Get log probability of completion tokens
        # Find where completion starts
        completion_tokens = self.image_processor.tokenizer.encode(completion, add_special_tokens=False)

        # Sum log probabilities of completion tokens
        total_log_prob = 0
        input_ids = inputs["input_ids"][0]

        # Find completion tokens in input_ids and compute log probs only for needed positions/tokens
        for token_id in completion_tokens:
            # Find position in input
            for pos in range(len(input_ids) - 1):
                if input_ids[pos + 1] == token_id:
                    # Compute log_softmax only for this position to save memory
                    # Shape: [vocab_size]
                    position_logits = logits[0, pos, :]
                    log_prob_at_pos = F.log_softmax(position_logits, dim=-1)
                    total_log_prob = total_log_prob + log_prob_at_pos[token_id]
                    break

        return total_log_prob

    def _get_logits(self, images, system_prompt, query_template, target_completion):
        """Helper to get logits for target completion."""
        logprobs = self.model.get_choice_logprobs(
            prompt=query_template,
            choices=[target_completion],
            image_paths=images
        )
        return logprobs[target_completion]


def create_r1_analyzer(model_name='R1-Onevision-7B'):
    """
    Convenience function to create R1ConceptAnalyzer with model and CLIP.

    Args:
        model_name: R1-Onevision model variant

    Returns:
        R1ConceptAnalyzer instance
    """
    from models.r1_onevision import R1OnevisionAPI
    from experiments.bootstrap_resample_for_pvalues_r1 import CLIPEmbedder

    print(f"Loading R1-Onevision model: {model_name}")
    r1_model = R1OnevisionAPI(model_name=model_name)

    print("Loading CLIP embedder...")
    clip = CLIPEmbedder()

    analyzer = R1ConceptAnalyzer(r1_model, clip)

    return analyzer
