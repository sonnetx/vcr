"""
Adapted ConceptAnalyzer for R1-Onevision models.

R1-Onevision is based on Qwen2.5-VL.
This module adapts the VCR approach to work with the R1 model's layer structure.
"""

import torch

# Compatibility patch for torch.compiler.is_compiling
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
from torch import nn


class R1LayerOverride(nn.Module):
    """
    Layer wrapper that allows hooking and gradient computation.

    The layer is replaced in the model, so all forward passes go through this wrapper.

    Important: This wrapper proxies attribute access to the original module,
    which is required for models like Qwen that access layer attributes directly.
    """
    def __init__(self, original_module):
        super().__init__()
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, '_original_module', original_module)
        object.__setattr__(self, '_override', None)

    def __getattr__(self, name):
        """Proxy attribute access to the original module."""
        if name in ('_original_module', '_override'):
            return object.__getattribute__(self, name)
        try:
            return getattr(object.__getattribute__(self, '_original_module'), name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Handle attribute setting."""
        if name in ('_original_module', '_override'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._original_module, name, value)

    @property
    def original_module(self):
        return object.__getattribute__(self, '_original_module')

    @property
    def override(self):
        return object.__getattribute__(self, '_override')

    @override.setter
    def override(self, value):
        object.__setattr__(self, '_override', value)

    def forward(self, *args, **kwargs):
        """Forward pass through the original module."""
        output = self._original_module(*args, **kwargs)

        if self._override is not None:
            if isinstance(output, tuple):
                new_output = list(output)
                if isinstance(new_output[0], torch.Tensor):
                    new_output[0] = new_output[0] * 0 + self._override
                return tuple(new_output)
            elif isinstance(output, torch.Tensor):
                return output * 0 + self._override

        return output


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
        self.model = r1_model
        self.model_name = r1_model.model_id if hasattr(r1_model, 'model_id') else 'R1-Onevision'
        self.clip = clip_embedder
        self.image_processor = r1_model.processor  # Use same name as original

        self.wrapped_layer = None  # The wrapped layer module (R1LayerOverride instance)
        self.concept_model = None  # Trained Ridge model
        self.concept_vectors = None 
        self._activations = []  

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

        base_model = self.model.model

        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            num_layers = len(base_model.model.layers)
            for i in range(num_layers):
                layers.append(f'model.model.layers.{i}')

        if hasattr(base_model, 'visual'):
            if hasattr(base_model.visual, 'blocks'):
                num_visual_layers = len(base_model.visual.blocks)
                for i in range(num_visual_layers):
                    layers.append(f'model.visual.blocks.{i}')

        return layers

    def setup_layer_hook(self, layer_name):
        """
        Set up activation hook on specified layer by wrapping it with R1LayerOverride.

        the layer is wrapped and replaced
        in the model, so all forward passes go through the wrapper.

        Args:
            layer_name: Name of layer to hook (e.g., 'model.model.layers.31')
        """
        self._activations = []
        self.layer_name = layer_name

        base_model = self.model.model
        if hasattr(base_model, 'module'):
            base_model = base_model.module

        parts = layer_name.split('.')

        target_module = base_model
        try:
            for attr in parts:
                if attr.isdigit():
                    target_module = target_module[int(attr)]
                else:
                    target_module = getattr(target_module, attr)
        except AttributeError as e:
            print(f"Error: Could not find layer '{layer_name}'")
            print(f"Available attributes at this level: {[a for a in dir(target_module) if not a.startswith('_')][:20]}")
            print(f"\nTo find valid layer names, run:")
            print(f"  python src/experiments/inspect_r1_layers.py --model {self.model.model_id}")
            raise AttributeError(f"Layer '{layer_name}' not found in model. {e}")

        self.wrapped_layer = R1LayerOverride(target_module)

        parent_module = base_model
        for attr in parts[:-1]:
            if attr.isdigit():
                parent_module = parent_module[int(attr)]
            else:
                parent_module = getattr(parent_module, attr)

        last_part = parts[-1]
        if last_part.isdigit():
            parent_module[int(last_part)] = self.wrapped_layer
        else:
            setattr(parent_module, last_part, self.wrapped_layer)

        print(f"Wrapped layer: {layer_name}")

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

        self._activations = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            self._activations.append(act.detach().cpu())

        hook_handle = self.wrapped_layer.register_forward_hook(hook_fn)

        def collate_fn(batch):
            """Custom collate that doesn't try to stack PIL images"""
            return {
                'image': [item['image'] for item in batch],
                'label': [item['label'] for item in batch],
                'image_path': [item['image_path'] for item in batch]
            }

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        base_model = self.model.model
        if hasattr(base_model, 'module'):
            base_model = base_model.module
        base_model.eval()

        for batch in tqdm(dataloader, desc="Collecting activations"):
            image_batch = batch['image']

            with torch.no_grad(), torch.cuda.amp.autocast():
                # Clear cache before each batch to prevent OOM
                torch.cuda.empty_cache()
                try:
                    from qwen_vl_utils import process_vision_info
                    use_qwen_vl_utils = True
                except ImportError:
                    use_qwen_vl_utils = False

                if use_qwen_vl_utils:
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

                # Clear inputs immediately after forward pass
                del inputs
                torch.cuda.empty_cache()

        hook_handle.remove()

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
        if isinstance(activations, torch.Tensor):
            X = activations.numpy()
        else:
            X = activations

        if isinstance(similarity_matrix, torch.Tensor):
            Y = similarity_matrix.T.numpy()
        else:
            Y = similarity_matrix.T

        print(f"Training concept model: X={X.shape}, Y={Y.shape}")

        model = Ridge(alpha=alpha)
        model.fit(X, Y)

        Y_pred = model.predict(X)

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

        coef_matrix = self.concept_model.coef_

        if len(coef_matrix.shape) == 1:
            concept_vectors = coef_matrix.reshape(1, -1)
        else:
            concept_vectors = coef_matrix

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

        base_model = self.model.model
        if hasattr(base_model, 'module'):
            base_model = base_model.module

        model_dtype = next(base_model.parameters()).dtype

        concept_vectors = concept_vectors.to(device=base_model.device, dtype=model_dtype)
        concept_weights = concept_weights.to(device=base_model.device, dtype=model_dtype)

        all_raw_sensitivities = []
        all_weighted_sensitivities = []

        base_model.eval()

        for batch in tqdm(dataset, desc="Computing directional derivatives"):
            torch.cuda.empty_cache()

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
                    # Store full output to preserve computation graph
                    layer_outputs_malignant.append(output)

                hook_malignant = self.wrapped_layer.register_forward_hook(hook_fn_malignant)

                log_prob_malignant = self._forward_with_completion(
                    image, system_prompt, query_template, " malignant"
                )

                # Get gradient w.r.t. activation
                if isinstance(layer_outputs_malignant[-1], tuple):
                    activation_malignant = layer_outputs_malignant[-1][0]
                else:
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
                    # Store full output to preserve computation graph
                    layer_outputs_benign.append(output)

                hook_benign = self.wrapped_layer.register_forward_hook(hook_fn_benign)

                log_prob_benign = self._forward_with_completion(
                    image, system_prompt, query_template, " benign"
                )

                if isinstance(layer_outputs_benign[-1], tuple):
                    activation_benign = layer_outputs_benign[-1][0]
                else:
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

                contrastive_grad = grad_malignant - grad_benign

                # Flatten gradient to [seq_len, hidden_dim] (matching original)
                # Shape: [batch, seq_len, hidden_dim] -> [seq_len, hidden_dim]
                flattened_grad = contrastive_grad.view(contrastive_grad.size(1), -1)

                del activation_malignant, activation_benign, grad_malignant, grad_benign, contrastive_grad
                del layer_outputs_malignant[:], layer_outputs_benign[:]
                torch.cuda.empty_cache()

            elif task_score == 'malignant_prob':
                # === Single completion: compute gradient for target only ===

                # Register temporary hook that doesn't detach (for gradient computation)
                layer_outputs = []
                def hook_fn(module, input, output):
                    # Store full output to preserve computation graph
                    layer_outputs.append(output)

                hook = self.wrapped_layer.register_forward_hook(hook_fn)

                log_prob = self._forward_with_completion(
                    image, system_prompt, query_template, target_completion
                )

                # Debug: check what we captured
                if len(layer_outputs) == 0:
                    print(f"WARNING: No activations captured by hook!")
                    hook.remove()
                    continue

                # Extract first element if tuple (like original implementation)
                if isinstance(layer_outputs[-1], tuple):
                    activation = layer_outputs[-1][0]
                else:
                    activation = layer_outputs[-1]

                # Debug: verify gradient tracking
                if not log_prob.requires_grad:
                    print(f"WARNING: log_prob does not require grad!")
                if not activation.requires_grad:
                    print(f"WARNING: activation does not require grad! Shape: {activation.shape}")

                grad = torch.autograd.grad(
                    outputs=log_prob,
                    inputs=activation,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True  # Add this to see if activation is actually used
                )[0]

                # Debug: check if gradient is None (meaning activation wasn't used)
                if grad is None:
                    print(f"WARNING: grad is None - activation not in computation graph!")
                    hook.remove()
                    del layer_outputs[:]
                    continue

                # Debug: print first sample's gradient stats
                if len(all_raw_sensitivities) == 0:
                    print(f"DEBUG: First sample gradient stats:")
                    print(f"  log_prob value: {log_prob.item():.6f}")
                    print(f"  activation shape: {activation.shape}")
                    print(f"  activation requires_grad: {activation.requires_grad}")
                    print(f"  grad shape: {grad.shape}")
                    print(f"  grad min/max: {grad.min().item():.8f} / {grad.max().item():.8f}")
                    print(f"  grad abs mean: {grad.abs().mean().item():.8f}")
                    print(f"  grad nonzero count: {(grad != 0).sum().item()} / {grad.numel()}")

                    # Find which token positions have non-zero gradients
                    grad_per_token = grad[0].abs().sum(dim=-1)  # [seq_len]
                    nonzero_mask = grad_per_token > 0
                    nonzero_positions = nonzero_mask.nonzero(as_tuple=True)[0]
                    print(f"  Number of positions with nonzero grad: {len(nonzero_positions)}")
                    if len(nonzero_positions) > 0:
                        print(f"  Non-zero gradient positions: {nonzero_positions.tolist()[:10]}...")  # First 10
                        print(f"  Last token position (seq_len-1): {grad.shape[1] - 1}")
                        print(f"  Grad at last position abs sum: {grad[0, -1].abs().sum().item():.8f}")
                        print(f"  Grad at nonzero pos[0]={nonzero_positions[0].item()} abs sum: {grad[0, nonzero_positions[0]].abs().sum().item():.8f}")

                hook.remove()

                # Clear intermediate tensors to free memory
                del log_prob

                # Flatten gradient to [seq_len, hidden_dim] (matching original)
                # Shape: [batch, seq_len, hidden_dim] -> [seq_len, hidden_dim]
                flattened_grad = grad.view(grad.size(1), -1)

                del activation, layer_outputs[:]
                torch.cuda.empty_cache()

            # Compute directional derivatives: gradient · concept_vector
            # [seq_len, hidden_dim] @ [n_concepts, hidden_dim].T = [seq_len, n_concepts]
            raw_sensitivities = torch.matmul(flattened_grad, concept_vectors.T)
            weighted_sensitivities = raw_sensitivities * concept_weights.unsqueeze(0)

            # Extract sensitivities from second-to-last position (index -2)
            # The gradient flows to the position that predicts the final token,
            # which is always seq_len - 2 (the position before the last token)
            # Convert to float32 before numpy (BFloat16 not supported)
            raw_at_pred_pos = raw_sensitivities[-2, :].cpu().detach()
            if raw_at_pred_pos.dtype == torch.bfloat16:
                raw_at_pred_pos = raw_at_pred_pos.float()
            all_raw_sensitivities.append(raw_at_pred_pos.numpy())

            weighted_at_pred_pos = weighted_sensitivities[-2, :].cpu().detach()
            if weighted_at_pred_pos.dtype == torch.bfloat16:
                weighted_at_pred_pos = weighted_at_pred_pos.float()
            all_weighted_sensitivities.append(weighted_at_pred_pos.numpy())

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

        # Process inputs - use same base_model reference as setup_layer_hook
        base_model = self.model.model
        if hasattr(base_model, 'module'):
            base_model = base_model.module
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
            if isinstance(image, str):
                image = Image.open(image)
            inputs = self.image_processor(
                text=messages,
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(base_model.device)

        # Enable gradients for backprop (even in eval mode)
        with torch.set_grad_enabled(True):
            outputs = base_model(**inputs)
            logits = outputs.logits

        # Get log probability of completion tokens
        # Find where completion starts in the input sequence
        completion_tokens = self.image_processor.tokenizer.encode(completion, add_special_tokens=False)
        input_ids = inputs["input_ids"][0]

        # Find the start position of the completion in the input
        # The completion should be at the end of the input sequence
        seq_len = len(input_ids)
        num_completion_tokens = len(completion_tokens)
        completion_start = seq_len - num_completion_tokens

        # Sum log probabilities of completion tokens at their correct positions
        # Initialize as None to ensure we get a proper tensor
        log_prob = None
        for j, token_id in enumerate(completion_tokens):
            pos = completion_start + j - 1  # Position to predict token at completion_start + j
            if pos >= 0 and pos < logits.shape[1]:
                position_logits = logits[0, pos, :]
                log_prob_at_pos = F.log_softmax(position_logits, dim=-1)
                token_log_prob = log_prob_at_pos[token_id]
                if log_prob is None:
                    log_prob = token_log_prob
                else:
                    log_prob = log_prob + token_log_prob

        if log_prob is None:
            log_prob = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return log_prob

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
