"""
Claude Sonnet LLM Judge wrapper using the Anthropic API.

Provides a callable interface compatible with the concept_presence_analysis pipeline.
"""

import os
import time
from typing import Dict, Optional

import anthropic


class ClaudeJudge:
    """Wrapper around Anthropic API for use as an LLM judge."""

    def __init__(self, model_name: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

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
        Query Claude and return response in the same format as R1OnevisionAPI.

        Returns:
            Dict with 'reasoning', 'answer', and 'full_output' keys.
        """
        messages = [{"role": "user", "content": prompt}]
        max_retries = 5

        for attempt in range(max_retries):
            try:
                if return_reasoning:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_new_tokens,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": max(1024, max_new_tokens // 2),
                        },
                        messages=messages,
                    )
                    reasoning_text = ""
                    answer_text = ""
                    for block in response.content:
                        if block.type == "thinking":
                            reasoning_text += block.thinking
                        elif block.type == "text":
                            answer_text += block.text

                    return {
                        "reasoning": reasoning_text,
                        "answer": answer_text,
                        "full_output": reasoning_text + "\n" + answer_text,
                    }
                else:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        messages=messages,
                    )
                    answer_text = response.content[0].text
                    return {
                        "reasoning": "",
                        "answer": answer_text,
                        "full_output": answer_text,
                    }

            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt + 1
                print(f"  Rate limit/API error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
