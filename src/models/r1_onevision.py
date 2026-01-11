import torch

# Compatibility patch for torch.compiler.is_compiling
# CRITICAL: This MUST be applied before transformers import
# The Qwen model code calls torch.compiler.is_compiling() during model loading
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
import requests
from typing import List, Dict, Union, Optional, Tuple
import os
import re

class R1OnevisionAPI:
    def __init__(
        self,
        model_name='R1-Onevision-7B',
        hf_token=None,
    ):
        """
        Args:
            model_name: Model variant to use. Options:
                - 'R1-Onevision-7B': 7B reasoning model (default)
            hf_token: Hugging Face authentication token (optional)
                     Get yours at https://huggingface.co/settings/tokens
                     Or use: huggingface-cli login
        """

        valid_models = {
            'R1-Onevision-7B'
        }
        assert model_name in valid_models, f"Error: Model '{model_name}' is not implemented. Valid models are: {', '.join(valid_models)}"

        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        self.model_name = model_name

        # Model ID mapping
        model_id_map = {
            'R1-Onevision-7B': 'Fancy-MLLM/R1-Onevision-7B',
        }

        self.model_id = model_id_map[model_name]
        self.hf_token = hf_token

        if self.hf_token is None:
            self.hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')

        print(f"Loading R1-Onevision model: {self.model_id}")
        print("This may take a while for the first download...")

        # Load model using Qwen2_5_VLForConditionalGeneration
        # (R1-Onevision is based on Qwen2.5-VL)
        print("Loading model with Qwen2_5_VLForConditionalGeneration...")
        from transformers import Qwen2_5_VLForConditionalGeneration

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=self.hf_token,
            )
            # Enable gradient checkpointing to reduce memory during backward pass
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            print("✓ Model loaded successfully")
        except Exception as model_error:
            print(f"Error loading model: {model_error}")
            raise

        print("\nLoading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.hf_token,
        )
        print("✓ Processor loaded successfully")

        # Get tokenizer
        self.tokenizer = self.processor.tokenizer

        print("\n✓ Successfully loaded R1-Onevision model and processor!")

        # Configure tokenizer
        if hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = "left"
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        print(f"Model loaded successfully!")

    def load_and_resize_image(self, path: Union[str, Image.Image], max_size: int = 518) -> Image.Image:
        """Load and resize image while maintaining aspect ratio
        Note: R1-Onevision uses 518x518 images by default"""
        try:
            if isinstance(path, str):
                if path.startswith('http://') or path.startswith('https://'):
                    # Load from URL
                    img = Image.open(requests.get(path, headers={"User-Agent": "R1OnevisionAPI"}, stream=True).raw)
                else:
                    # Load from local file
                    img = Image.open(path)
            elif isinstance(path, Image.Image):
                img = path
            else:
                raise ValueError(f"Unsupported image type: {type(path)}")

            img = img.convert('RGB')

            if img.size[0] > max_size or img.size[1] > max_size:
                ratio = min(max_size/img.size[0], max_size/img.size[1])
                new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
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
        Extract reasoning and final answer from model output.
        R1-Onevision uses <think> and <answer> tags to separate reasoning from final answer.

        Args:
            text: Raw model output

        Returns:
            Tuple of (reasoning, final_answer)
        """
        # Pattern 1: Look for <think> and <answer> tags (R1-Onevision format)
        # Format: <think>reasoning here<answer>final answer here
        think_answer_pattern = r'<think>(.*?)<answer>(.*?)(?:</answer>|$)'
        think_answer_match = re.search(think_answer_pattern, text, re.DOTALL)

        if think_answer_match:
            reasoning = think_answer_match.group(1).strip()
            final_answer = think_answer_match.group(2).strip()
            return reasoning, final_answer

        # Pattern 2: Look for <think> tag only (no <answer> tag)
        # In this case, everything after <think> is reasoning, and we extract conclusion
        think_only_pattern = r'<think>(.*?)$'
        think_only_match = re.search(think_only_pattern, text, re.DOTALL)

        if think_only_match:
            reasoning = think_only_match.group(1).strip()
            # Try to extract final answer from reasoning using conclusion markers
            conclusion_pattern = r'(.*?)(?:Therefore|Thus|In conclusion|To summarize|Final answer)[,:]?\s*(.*?)$'
            conclusion_match = re.search(conclusion_pattern, reasoning, re.DOTALL | re.IGNORECASE)
            if conclusion_match and conclusion_match.group(2).strip():
                final_answer = conclusion_match.group(2).strip()
                reasoning = conclusion_match.group(1).strip()
            else:
                # No clear conclusion, return last sentence/paragraph as answer
                final_answer = reasoning.split('.')[-1].strip() if '.' in reasoning else reasoning
            return reasoning, final_answer

        # Pattern 3: Legacy <think>...</think> tags (for backwards compatibility)
        think_closing_pattern = r'<think>(.*?)</think>(.*?)$'
        think_closing_match = re.search(think_closing_pattern, text, re.DOTALL)

        if think_closing_match:
            reasoning = think_closing_match.group(1).strip()
            final_answer = think_closing_match.group(2).strip()
            if not final_answer:
                final_answer = text
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

        # Pattern 5: If no explicit reasoning markers, try to split by common patterns
        # Look for transitions like "Therefore", "Thus", "In conclusion"
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

            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]

            for img in images:
                messages[0]["content"].append({
                    "type": "image",
                    "image": img
                })

            messages[0]["content"].append({
                "type": "text",
                "text": full_text
            })

            try:
                # Try using qwen_vl_utils if available
                try:
                    from qwen_vl_utils import process_vision_info

                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                except ImportError:
                    # Fallback: Use processor directly
                    inputs = self.processor(
                        text=messages,
                        images=images if images else None,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                logits = outputs.logits[0]

                prompt_only_messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                for img in images:
                    prompt_only_messages[0]["content"].append({
                        "type": "image",
                        "image": img
                    })
                prompt_only_messages[0]["content"].append({
                    "type": "text",
                    "text": prompt
                })

                try:
                    prompt_text = self.processor.apply_chat_template(
                        prompt_only_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
                except:
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)

                full_tokens = inputs["input_ids"][0]

                choice_start = len(prompt_tokens) - 1

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
        max_new_tokens: int = 4096,
        num_beams: int = 1,
        min_length: int = 1,
        temperature: float = 0.7,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
        return_reasoning: bool = True,
    ) -> Union[str, Dict[str, str]]:
        """
        Generate text response given prompt and optional images

        Args:
            prompt: Text prompt
            image_paths: List of image paths or PIL Images
            max_new_tokens: Maximum number of new tokens to generate (default 4096 for reasoning)
            num_beams: Number of beams for beam search
            min_length: Minimum length of generated sequence
            temperature: Sampling temperature
            do_sample: Whether to use sampling
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

        images = self.preprocess_images(image_paths) if image_paths else []

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

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
            # Try using qwen_vl_utils if available
            try:
                from qwen_vl_utils import process_vision_info

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            except ImportError:
                print("Warning: qwen_vl_utils not found, using fallback processing")
                inputs = self.processor(
                    text=messages,
                    images=images if images else None,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
        except Exception as e:
            print(f"Error during input processing: {e}")
            raise

        with torch.no_grad():
            generated = self.model.generate(
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

        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated[0][input_length:]
        full_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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
