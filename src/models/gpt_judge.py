"""
GPT LLM Judge wrapper using the OpenAI API.

Provides a callable interface compatible with the concept_presence_analysis pipeline.
"""

import os
import time
from typing import Dict, Optional

import openai


class GPTJudge:
    """Wrapper around OpenAI API for use as an LLM judge."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def __call__(
        self,
        prompt: str,
        image_paths=None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        do_sample: bool = False,
        return_reasoning: bool = True,
    ) -> Dict[str, str]:
        """
        Query GPT and return response in the same format as ClaudeJudge/R1OnevisionAPI.

        Returns:
            Dict with 'reasoning', 'answer', and 'full_output' keys.
        """
        messages = [{"role": "user", "content": prompt}]
        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_completion_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=messages,
                )
                answer_text = response.choices[0].message.content

                return {
                    "reasoning": "",
                    "answer": answer_text,
                    "full_output": answer_text,
                }

            except (openai.RateLimitError, openai.APIStatusError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt + 1
                print(f"  Rate limit/API error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
