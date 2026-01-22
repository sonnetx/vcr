import torch

# Compatibility patch for torch.compiler.is_compiling
# CRITICAL: This MUST be applied before transformers import
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
import requests
from typing import List, Dict, Union, Optional, Tuple
import os
import re


class GLM4VisionAPI:
    def __init__(
        self,
        model_name='GLM-4.1V-9B-Thinking',
        hf_token=None,
        use_flash_attention=False,
    ):
        """
        Args:
            model_name: Model variant to use. Options:
                - 'GLM-4.1V-9B-Thinking': 9B thinking/reasoning model (default)
            hf_token: Hugging Face authentication token (optional)
                     Get yours at https://huggingface.co/settings/tokens
                     Or use: huggingface-cli login
            use_flash_attention: Whether to use flash attention 2 for better memory efficiency
        """

        valid_models = {
            'GLM-4.1V-9B-Thinking'
        }
        assert model_name in valid_models, f"Error: Model '{model_name}' is not implemented. Valid models are: {', '.join(valid_models)}"

        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        self.model_name = model_name

        # Model ID mapping
        model_id_map = {
            'GLM-4.1V-9B-Thinking': 'zai-org/GLM-4.1V-9B-Thinking',
        }

        self.model_id = model_id_map[model_name]
        self.hf_token = hf_token

        if self.hf_token is None:
            self.hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')

        print(f"Loading GLM-4.1V model: {self.model_id}")
        print("This may take a while for the first download...")

        # Load model using Glm4vForConditionalGeneration
        print("Loading model with Glm4vForConditionalGeneration...")
        from transformers import Glm4vForConditionalGeneration

        try:
            load_kwargs = {
                'torch_dtype': torch.bfloat16,
                'device_map': "auto",
                'trust_remote_code': True,
                'token': self.hf_token,
            }

            if use_flash_attention:
                load_kwargs['attn_implementation'] = "flash_attention_2"
                print("Using flash attention 2...")

            self.model = Glm4vForConditionalGeneration.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            # Enable gradient checkpointing to reduce memory during backward pass
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
            print("Model loaded successfully")
        except Exception as model_error:
            print(f"Error loading model: {model_error}")
            raise

        print("\nLoading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.hf_token,
            use_fast=True,
        )
        print("Processor loaded successfully")

        # Get tokenizer from processor
        self.tokenizer = self.processor.tokenizer

        print("\nSuccessfully loaded GLM-4.1V model and processor!")

        # Configure tokenizer
        if hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = "left"
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        print(f"Model ready on device!")

    def load_and_resize_image(self, path: Union[str, Image.Image], max_pixels: int = 16777216) -> Image.Image:
        """Load and resize image while maintaining aspect ratio
        Note: GLM-4.1V supports up to 4K resolution (~16M pixels for 4096x4096)"""
        try:
            if isinstance(path, str):
                if path.startswith('http://') or path.startswith('https://'):
                    # Load from URL
                    img = Image.open(requests.get(path, headers={"User-Agent": "GLM4VisionAPI"}, stream=True).raw)
                else:
                    # Load from local file
                    img = Image.open(path)
            elif isinstance(path, Image.Image):
                img = path
            else:
                raise ValueError(f"Unsupported image type: {type(path)}")

            img = img.convert('RGB')

            # Resize if image exceeds max pixels
            current_pixels = img.size[0] * img.size[1]
            if current_pixels > max_pixels:
                ratio = (max_pixels / current_pixels) ** 0.5
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            return img
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            raise

    def preprocess_images(self, images: List[Union[str, Image.Image]]) -> List[Image.Image]:
        """Preprocess a list of images"""
        try:
            processed_images = []
            for image in images:
                img = self.load_and_resize_image(image)
                processed_images.append(img)
            return processed_images
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def extract_reasoning(self, text: str) -> Tuple[str, str]:
        """
        Extract reasoning and final answer from GLM-4.1V-Thinking model output.

        GLM uses <think>...</think> tags followed by <answer>...</answer> tags.

        Args:
            text: Raw model output

        Returns:
            Tuple of (reasoning, final_answer)
        """
        # Pattern 1: <think>...</think> followed by <answer>...</answer> (PRIMARY for GLM-4.1V-Thinking)
        # This is the actual format GLM uses
        think_answer_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)(?:</answer>|$)'
        think_answer_match = re.search(think_answer_pattern, text, re.DOTALL)

        if think_answer_match:
            reasoning = think_answer_match.group(1).strip()
            final_answer = think_answer_match.group(2).strip()
            return reasoning, final_answer

        # Pattern 2: R1-style <think>...<answer> without </think> closing tag
        think_answer_no_close_pattern = r'<think>(.*?)<answer>(.*?)(?:</answer>|$)'
        think_answer_no_close_match = re.search(think_answer_no_close_pattern, text, re.DOTALL)

        if think_answer_no_close_match:
            reasoning = think_answer_no_close_match.group(1).strip()
            final_answer = think_answer_no_close_match.group(2).strip()
            return reasoning, final_answer

        # Pattern 3: Proper <think>...</think> format without <answer> tags
        think_pattern = r'<think>(.*?)</think>\s*(.*?)$'
        think_match = re.search(think_pattern, text, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            final_answer = think_match.group(2).strip()
            # Strip any remaining <answer> tags if present
            final_answer = re.sub(r'^<answer>\s*', '', final_answer)
            final_answer = re.sub(r'\s*</answer>$', '', final_answer)
            return reasoning, final_answer

        # Pattern 4: <think> tag without closing (incomplete response)
        think_only_pattern = r'<think>(.*?)$'
        think_only_match = re.search(think_only_pattern, text, re.DOTALL)

        if think_only_match:
            reasoning = think_only_match.group(1).strip()
            # Check if there's an <answer> tag within the content
            answer_in_think = re.search(r'<answer>(.*?)(?:</answer>|$)', reasoning, re.DOTALL)
            if answer_in_think:
                # Split reasoning and answer
                reasoning = reasoning[:reasoning.index('<answer>')].strip()
                final_answer = answer_in_think.group(1).strip()
                return reasoning, final_answer
            # Try to extract conclusion from reasoning
            conclusion_pattern = r'(.*?)(?:Therefore|Thus|In conclusion|To summarize|Final answer)[,:]?\s*(.*?)$'
            conclusion_match = re.search(conclusion_pattern, reasoning, re.DOTALL | re.IGNORECASE)
            if conclusion_match and conclusion_match.group(2).strip():
                return conclusion_match.group(1).strip(), conclusion_match.group(2).strip()
            # Default: return last sentence as answer
            final_answer = reasoning.split('.')[-1].strip() if '.' in reasoning else reasoning
            return reasoning, final_answer

        # Pattern 4: Look for explicit "Reasoning:" or "Analysis:" sections
        reasoning_patterns = [
            r'(?:Reasoning|Analysis|Thought process):\s*(.*?)(?:\n\n|$)(.*)',
            r'(?:Let me think|Let\'s think).*?\.\s*(.*?)(?:\n\n|$)(.*)',
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                final_answer = match.group(2).strip() if len(match.groups()) > 1 else text
                return reasoning, final_answer

        # Pattern 5: Conclusion markers without tags
        conclusion_pattern = r'(.*?)(?:Therefore|Thus|In conclusion|To summarize|Final answer)[,:]?\s*(.*)'
        conclusion_match = re.search(conclusion_pattern, text, re.DOTALL | re.IGNORECASE)

        if conclusion_match:
            reasoning = conclusion_match.group(1).strip()
            final_answer = conclusion_match.group(2).strip()
            if reasoning and final_answer:
                return reasoning, final_answer

        # Default: No reasoning found, return empty reasoning and full text as answer
        return "", text.strip()

    def get_choice_logprobs(self, prompt: str, choices: List[str], image_paths: List[Union[str, Image.Image]] = []) -> Dict[str, float]:
        """Calculate log probabilities for each choice"""

        choice_logprobs = {}
        images = self.preprocess_images(image_paths) if image_paths else []

        for choice in choices:
            full_text = f"{prompt} {choice}"

            # Build messages with images
            user_content = []
            for img in images:
                user_content.append({
                    "type": "image",
                    "image": img
                })
            user_content.append({
                "type": "text",
                "text": full_text
            })

            messages = [{"role": "user", "content": user_content}]

            try:
                # GLM uses apply_chat_template with tokenize=True, return_dict=True
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits[0]

                # Get prompt-only tokens for comparison
                prompt_content = []
                for img in images:
                    prompt_content.append({"type": "image", "image": img})
                prompt_content.append({"type": "text", "text": prompt})
                prompt_messages = [{"role": "user", "content": prompt_content}]

                try:
                    prompt_inputs = self.processor.apply_chat_template(
                        prompt_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    prompt_tokens_len = prompt_inputs["input_ids"].shape[1]
                except:
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                    prompt_tokens_len = len(prompt_tokens)

                full_tokens = inputs["input_ids"][0]
                choice_start = prompt_tokens_len - 1

                choice_logprob = 0
                for idx in range(choice_start, min(len(full_tokens) - 1, len(logits) - 1)):
                    token_logits = logits[idx]
                    next_token = full_tokens[idx + 1]
                    token_probs = torch.log_softmax(token_logits, dim=-1)
                    if next_token < len(token_probs):
                        choice_logprob += token_probs[next_token].item()

                choice_logprobs[choice] = choice_logprob

            except Exception as e:
                print(f"Error processing choice '{choice}': {e}")
                choice_logprobs[choice] = float('-inf')

        return choice_logprobs

    def get_best_choice(self, prompt: str, choices: List[str], image_paths: List[Union[str, Image.Image]] = []) -> Tuple[str, Dict[str, float]]:
        """Get the most likely choice based on log probabilities"""
        logprobs = self.get_choice_logprobs(prompt, choices, image_paths)
        best_choice = max(logprobs.items(), key=lambda x: x[1])[0]
        return best_choice, logprobs

    def __call__(
        self,
        prompt: str,
        image_paths: List[Union[str, Image.Image]] = [],
        max_new_tokens: int = 8192,
        num_beams: int = 1,
        min_length: int = 1,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
        return_reasoning: bool = True,
    ) -> Union[str, Dict[str, str]]:
        """
        Generate text response given prompt and optional images

        Args:
            prompt: Text prompt
            image_paths: List of image paths, URLs, or PIL Images
            max_new_tokens: Maximum number of new tokens to generate (default 8192)
            num_beams: Number of beams for beam search
            min_length: Minimum length of generated sequence
            temperature: Sampling temperature (default 1.0 for thinking models)
            do_sample: Whether to use sampling (default True for thinking models)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            system_prompt: Optional system prompt
            return_reasoning: If True, return dict with 'reasoning', 'answer', and 'full_output'.
                            If False, return only the final answer string.

        Returns:
            If return_reasoning=True: Dict with keys:
                - 'reasoning': Extracted reasoning process
                - 'answer': Final answer
                - 'full_output': Complete model output
            If return_reasoning=False: String with final answer only
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths] if image_paths else []

        # Load and preprocess images
        images = self.preprocess_images(image_paths) if image_paths else []

        # Build messages
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # Build user content with images and text
        user_content = []
        for img in images:
            user_content.append({
                "type": "image",
                "image": img
            })
        user_content.append({
            "type": "text",
            "text": prompt
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            # GLM uses apply_chat_template with tokenize=True, return_dict=True
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
        except Exception as e:
            print(f"Error during input processing: {e}")
            raise

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
                min_length=min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output - use processor.decode and skip_special_tokens=False to preserve <think> tags
        input_length = inputs["input_ids"].shape[1]
        output_text = self.processor.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=False
        )

        # Clean up end tokens but preserve think tags
        full_output = output_text.replace('<|endoftext|>', '').replace('<|user|>', '').replace('<|assistant|>', '').strip()

        if return_reasoning:
            reasoning, answer = self.extract_reasoning(full_output)
            return {
                'reasoning': reasoning,
                'answer': answer,
                'full_output': full_output
            }
        else:
            _, answer = self.extract_reasoning(full_output)
            return answer
