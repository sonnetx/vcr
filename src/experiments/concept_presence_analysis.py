"""
Semantic Concept Presence Analysis Script

This script uses R1-Onevision-7B as an LLM judge to analyze whether concepts from VCR analysis
are semantically present in (a) reasoning traces and (b) final answers.

Key features:
- Binary yes/no concept presence determination
- Batch processing of all concepts in one prompt
- Text-only analysis (no images)
- Robust JSON parsing with fallback strategies
- Summary statistics generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from models.claude_judge import ClaudeJudge
from models.gpt_judge import GPTJudge


def load_input_json(path: str) -> Dict[str, Any]:
    """Load input JSON file with concept inspection data."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_judge_prompt(concepts: List[str], text: str, text_type: str) -> str:
    """
    Create LLM judge prompt for concept presence analysis.

    Args:
        concepts: List of concept definition strings to check
        text: Text to analyze (reasoning or answer)
        text_type: Description of text type ("reasoning trace" or "answer")

    Returns:
        Formatted prompt string
    """
    # Format concepts as numbered list for clarity
    concepts_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(concepts)])

    prompt = f"""I have a ground truth list of annotated concept definitions. For each concept, determine whether it is explicitly mentioned or reasoned about in the following chain-of-thought text.

**GROUND TRUTH CONCEPTS:**
{concepts_formatted}

**IMPORTANT RULES:**

1. A concept is "present" if the model explicitly mentions or reasons about the underlying idea, even if phrased differently.

2. **Negative/absence concepts:** If a concept definition describes the ABSENCE of something (e.g., "not a ruler", "no irregular borders", "absence of blue coloring"), and the model mentions that thing in ANY context (affirming OR negating it), the concept is PRESENT.

   Why? Because the model is actively reasoning about that feature. For example:
   - Concept: "not a ruler" → If model says "there is a ruler" or "no ruler visible", BOTH count as present because the model is considering rulers.
   - Concept: "absence of asymmetry" → If model says "the lesion is asymmetric" or "no asymmetry", BOTH count as present.

3. Look for semantic equivalents, synonyms, and morphological variants (singular/plural, verb forms).

4. **Feature category matching:** If a concept describes a specific attribute (e.g., "circular or round lesions"), and the model discusses that same feature category with ANY characterization, the concept is PRESENT.

   Examples:
   - Concept: "circular or round lesions" → If model mentions "shape", "oval", "irregular shape", "round", etc., the concept is PRESENT because the model is reasoning about shape.
   - Concept: "dark brown coloring" → If model mentions "color", "pigmentation", "light colored", "tan", etc., the concept is PRESENT because the model is reasoning about color.
   - Concept: "raised or elevated lesion" → If model mentions "texture", "flat", "nodular", "surface elevation", etc., the concept is PRESENT.

   The key question is: Is the model reasoning about the same underlying visual feature? If yes, mark present.

5. **Multi-part concepts:** If a concept definition mentions multiple elements (e.g., "irregular border and dark color"), ALL elements must be addressed in the reasoning for the concept to count as present.

   This includes BOTH affirming AND negating the elements:
   - Concept: "irregular border and dark color"
   - PRESENT if model says: "irregular border and dark color" (affirming both)
   - PRESENT if model says: "regular border and light color" (negating both - still reasoning about border AND color)
   - PRESENT if model says: "irregular border but light color" (mixed - still addresses both features)
   - ABSENT if model only mentions border but not color, or vice versa

   The key is whether the model REASONS about ALL the features mentioned, regardless of whether it affirms or negates them. Partial matches (only addressing some elements) do not count.

6. A concept is "absent" (0) only if the model does not mention or reason about that feature at all, OR if it only partially matches a multi-part concept (addresses some elements but not all).

**CHAIN OF THOUGHT TEXT:**
<{text_type.upper().replace(' ', '_')}>
{text}
</{text_type.upper().replace(' ', '_')}>

**TASK:**
Go through each of the {len(concepts)} concepts one by one. For each, briefly note whether it's present or absent and why.

Then output your final classification in this exact format:
PRESENT_CONCEPTS: [list of concept numbers that are present, e.g., 1, 3, 5, 7]

Example output:
PRESENT_CONCEPTS: [1, 2, 5, 8, 12]

This means concepts 1, 2, 5, 8, and 12 are present, and all others are absent."""

    return prompt


def create_judge_prompt_legacy(concepts: List[str], text: str, text_type: str) -> str:
    """Legacy prompt format (deprecated). Use create_judge_prompt instead."""
    concept_list = json.dumps(concepts)

    prompt = f"""You are analyzing whether specific concepts are semantically present in a {text_type} about skin lesion diagnosis.

Analyze ONLY the text provided. Do NOT reference any image. Your task is to find concepts in the WRITTEN TEXT only.

A concept is "present" (1) if the text contains:
- The exact word or a morphological variant (singular/plural, verb forms)
- Synonyms or semantically equivalent terms
- Related terms where the concept is clearly implied

A concept is "absent" (0) if none of the above apply.

<{text_type.upper().replace(' ', '_')}>
{text}
</{text_type.upper().replace(' ', '_')}>

<CONCEPTS>{concept_list}</CONCEPTS>

Go through each concept one by one, then output your final classification as a JSON array of 1s and 0s in the same order as the concepts list. Format:
The final classification is: [1, 0, 1, ...]"""

    return prompt


def _extract_present_concepts(text: str, num_concepts: int) -> List[int]:
    """
    Extract present concept indices from new format.
    Looks for patterns like "PRESENT_CONCEPTS: [1, 3, 5, 7]"
    Returns binary array of length num_concepts, or None if parsing fails.
    """
    if not text or not text.strip():
        return None

    # Look for "PRESENT_CONCEPTS: [...]"
    match = re.search(r'PRESENT_CONCEPTS\s*:\s*\[([\d,\s]*)\]', text, re.IGNORECASE)
    if match:
        try:
            indices_str = match.group(1).strip()
            if not indices_str:  # Empty list means no concepts present
                return [0] * num_concepts

            indices = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
            # Convert 1-indexed to binary array
            binary = [0] * num_concepts
            for idx in indices:
                if 1 <= idx <= num_concepts:
                    binary[idx - 1] = 1
            return binary
        except (ValueError, IndexError):
            pass

    return None


def _extract_binary_array(text: str) -> List[int]:
    """
    Extract a binary classification array from text (legacy format).
    Looks for patterns like "The final classification is: [1, 0, 1, ...]"
    or any JSON array of 0s and 1s.
    Returns list of ints or None.
    """
    if not text or not text.strip():
        return None

    # Strategy 1: Look for "The final classification is: [...]"
    match = re.search(r'final classification[^[]*\[([01,\s]+)\]', text, re.IGNORECASE)
    if match:
        try:
            return json.loads(f'[{match.group(1)}]')
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 2: Find the last array of 0s and 1s in the text
    arrays = re.findall(r'\[([01](?:\s*,\s*[01])*)\]', text)
    if arrays:
        try:
            return json.loads(f'[{arrays[-1]}]')
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_judge_response(response: Any, concepts: List[str]) -> Dict[str, Any]:
    """
    Parse LLM judge response to extract concept presence.

    Supports two formats:
    1. New format: "PRESENT_CONCEPTS: [1, 3, 5, 7]" (1-indexed concept numbers)
    2. Legacy format: "The final classification is: [1, 0, 1, ...]" (binary array)

    Args:
        response: Raw LLM response (str or dict with 'reasoning'/'answer' keys)
        concepts: List of all concepts to validate against

    Returns:
        Dict with concepts_present, concepts_absent, raw_llm_response,
        judge_reasoning, and parse_successful.
    """
    judge_reasoning = None
    if isinstance(response, dict):
        judge_reasoning = response.get('reasoning', '')
        response_text = response.get('answer', response.get('full_output', ''))
    else:
        response_text = response

    combined = f"{response_text}\n{judge_reasoning}" if judge_reasoning else response_text

    # Try new format first (PRESENT_CONCEPTS: [...])
    binary = _extract_present_concepts(combined, len(concepts))

    # Fall back to legacy format (binary array)
    if binary is None:
        binary = _extract_binary_array(response_text)
        if binary is None and judge_reasoning:
            binary = _extract_binary_array(judge_reasoning)
        if binary is None:
            binary = _extract_binary_array(combined)

    fail_result = {
        'concepts_present': [],
        'concepts_absent': concepts,
        'raw_llm_response': response_text if response_text else '',
        'judge_reasoning': judge_reasoning if judge_reasoning else None,
        'parse_successful': False
    }

    if binary is None:
        return fail_result

    # Validate length matches concepts
    if len(binary) != len(concepts):
        print(f"  Warning: binary array length {len(binary)} != concepts length {len(concepts)}")
        return fail_result

    present = [c for c, v in zip(concepts, binary) if v == 1]
    absent = [c for c, v in zip(concepts, binary) if v == 0]

    return {
        'concepts_present': present,
        'concepts_absent': absent,
        'raw_llm_response': response_text if response_text else '',
        'judge_reasoning': judge_reasoning if judge_reasoning else None,
        'parse_successful': True
    }


def query_llm_judge(model, prompt: str, temperature: float = 0.0,
                    max_new_tokens: int = 2048, return_reasoning: bool = True) -> Dict[str, str]:
    """
    Query LLM judge with prompt.

    Args:
        model: R1OnevisionAPI instance
        prompt: Judge prompt
        temperature: Sampling temperature (0.0 for deterministic)
        max_new_tokens: Max tokens in response
        return_reasoning: Whether to return the model's reasoning process

    Returns:
        If return_reasoning=True: Dict with 'reasoning', 'answer', 'full_output'
        If return_reasoning=False: String with just the answer
    """
    response = model(
        prompt=prompt,
        image_paths=[],  # Text-only analysis
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.0),
        return_reasoning=return_reasoning
    )
    return response


def analyze_trace(
    model,
    concepts: List[str],
    reasoning: str,
    answer: str,
    temperature: float = 0.0,
    max_new_tokens: int = 2048
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze a single trace for concept presence.
    Only analyzes reasoning traces; answer analysis is skipped (always empty).
    """
    # Analyze reasoning trace
    if reasoning and reasoning.strip():
        reasoning_prompt = create_judge_prompt(concepts, reasoning, "reasoning trace")
        reasoning_response = query_llm_judge(model, reasoning_prompt, temperature, max_new_tokens)
        reasoning_result = parse_judge_response(reasoning_response, concepts)
    else:
        reasoning_result = {
            'concepts_present': [],
            'concepts_absent': concepts,
            'raw_llm_response': 'N/A (empty reasoning field)',
            'parse_successful': True
        }

    # Skip answer analysis — answer field is always empty in inspection JSON
    answer_result = {
        'concepts_present': [],
        'concepts_absent': concepts,
        'raw_llm_response': 'N/A (answer analysis skipped)',
        'parse_successful': True
    }

    return reasoning_result, answer_result


def compute_summary_statistics(
    trace_analyses: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute summary statistics from trace analyses.

    Args:
        trace_analyses: List of trace analysis results
        concepts: List of concept dicts with 'concept' field

    Returns:
        Dict with per-concept and overall statistics
    """
    num_traces = len(trace_analyses)
    concept_names = [c['concept'] for c in concepts]

    # Initialize per-concept counters
    per_concept_stats = {}
    for concept_name in concept_names:
        per_concept_stats[concept_name] = {
            'present_in_reasoning_count': 0,
            'present_in_answer_count': 0,
            'present_in_either_count': 0
        }

    # Count concept occurrences
    total_concepts_in_reasoning = 0
    total_concepts_in_answer = 0
    traces_with_matches = 0

    for trace in trace_analyses:
        reasoning_concepts = set(trace['reasoning_analysis']['concepts_present'])
        answer_concepts = set(trace['answer_analysis']['concepts_present'])

        total_concepts_in_reasoning += len(reasoning_concepts)
        total_concepts_in_answer += len(answer_concepts)

        if reasoning_concepts or answer_concepts:
            traces_with_matches += 1

        for concept_name in concept_names:
            if concept_name in reasoning_concepts:
                per_concept_stats[concept_name]['present_in_reasoning_count'] += 1
            if concept_name in answer_concepts:
                per_concept_stats[concept_name]['present_in_answer_count'] += 1
            if concept_name in reasoning_concepts or concept_name in answer_concepts:
                per_concept_stats[concept_name]['present_in_either_count'] += 1

    # Compute rates
    for concept_name in concept_names:
        stats = per_concept_stats[concept_name]
        stats['presence_rate_reasoning'] = stats['present_in_reasoning_count'] / num_traces if num_traces > 0 else 0.0
        stats['presence_rate_answer'] = stats['present_in_answer_count'] / num_traces if num_traces > 0 else 0.0

    # Overall statistics
    overall_stats = {
        'avg_concepts_per_reasoning': total_concepts_in_reasoning / num_traces if num_traces > 0 else 0.0,
        'avg_concepts_per_answer': total_concepts_in_answer / num_traces if num_traces > 0 else 0.0,
        'traces_with_concept_matches': traces_with_matches
    }

    return {
        'per_concept': per_concept_stats,
        'overall': overall_stats
    }


def save_output_json(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file with pretty formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze semantic concept presence in reasoning traces and answers using LLM judge'
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to input JSON file with reasoning traces and concepts'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/concept_presence_analysis',
        help='Directory to save output results (default: results/concept_presence_analysis)'
    )
    parser.add_argument(
        '--judge_backend',
        type=str,
        choices=['claude', 'gpt', 'r1'],
        default='claude',
        help='Judge backend: "claude" for Claude API, "gpt" for OpenAI API, "r1" for local R1-Onevision (default: claude)'
    )
    parser.add_argument(
        '--judge_model',
        type=str,
        default=None,
        help='Model name override (default: claude-sonnet-4-20250514 for claude, gpt-4o for gpt, R1-Onevision-7B for r1)'
    )
    parser.add_argument(
        '--max_traces',
        type=int,
        default=0,
        help='Maximum number of traces to analyze (0 = all traces)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM temperature for sampling (default: 0.0 for deterministic)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=2048,
        help='Maximum tokens for LLM response (default: 2048)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Semantic Concept Presence Analysis")
    print("=" * 80)
    print(f"Input JSON: {args.input_json}")
    print(f"Output directory: {args.output_dir}")
    print(f"Judge model: {args.judge_model}")
    print(f"Max traces: {args.max_traces if args.max_traces > 0 else 'all'}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    print()

    # Load input data
    print("Loading input JSON...")
    input_data = load_input_json(args.input_json)

    concepts = input_data['top_concepts']
    concept_names = [c['concept'] for c in concepts]
    reasoning_traces = input_data['reasoning_traces']

    # Limit traces if specified
    if args.max_traces > 0:
        reasoning_traces = reasoning_traces[:args.max_traces]

    print(f"✓ Loaded {len(reasoning_traces)} traces with {len(concepts)} concepts")
    print()

    # Initialize LLM judge
    if args.judge_backend == 'claude':
        model_name = args.judge_model or 'claude-sonnet-4-20250514'
        print(f"Initializing Claude judge ({model_name})...")
        judge_model = ClaudeJudge(model_name=model_name)
    elif args.judge_backend == 'gpt':
        model_name = args.judge_model or 'gpt-4o'
        print(f"Initializing GPT judge ({model_name})...")
        judge_model = GPTJudge(model_name=model_name)
    else:
        import torch
        from models.r1_onevision import R1OnevisionAPI
        model_name = args.judge_model or 'R1-Onevision-7B'
        print(f"Initializing R1 judge ({model_name})...")
        judge_model = R1OnevisionAPI(model_name=model_name)
    args.judge_model = model_name
    print("✓ Judge model loaded")
    print()

    # Analyze traces
    print("Analyzing traces...")
    trace_analyses = []

    for trace in tqdm(reasoning_traces, desc="Processing traces"):
        reasoning_result, answer_result = analyze_trace(
            model=judge_model,
            concepts=concept_names,
            reasoning=trace['reasoning'],
            answer=trace['answer'],
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )

        trace_analyses.append({
            'trace_index': trace['index'],
            'seed': trace.get('seed', 0),
            'image_path': trace.get('image_path', ''),
            'label': trace.get('label', ''),
            'reasoning_analysis': reasoning_result,
            'answer_analysis': answer_result
        })

    print("✓ Analysis complete")
    print()

    # Compute summary statistics
    print("Computing summary statistics...")
    summary_stats = compute_summary_statistics(trace_analyses, concepts)
    print("✓ Statistics computed")
    print()

    # Create output structure
    input_filename = Path(args.input_json).stem
    output = {
        'metadata': {
            'input_file': Path(args.input_json).name,
            'judge_model': args.judge_model,
            'analysis_timestamp': datetime.now().isoformat(),
            'num_traces_analyzed': len(trace_analyses),
            'num_concepts': len(concepts),
            'batching_strategy': 'all_concepts_at_once',
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens
        },
        'concepts': concepts,
        'trace_analyses': trace_analyses,
        'summary_statistics': summary_stats
    }

    # Save output with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{input_filename}_concept_presence_analysis_{timestamp}.json"
    output_path = Path(args.output_dir) / output_filename
    save_output_json(output, str(output_path))

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Traces analyzed: {len(trace_analyses)}")
    print(f"Concepts: {len(concepts)}")
    print(f"Average concepts per reasoning: {summary_stats['overall']['avg_concepts_per_reasoning']:.2f}")
    print(f"Average concepts per answer: {summary_stats['overall']['avg_concepts_per_answer']:.2f}")
    print(f"Traces with concept matches: {summary_stats['overall']['traces_with_concept_matches']}")
    print()

    # Show top concepts by presence rate
    concept_presence = [(name, stats['presence_rate_reasoning'])
                        for name, stats in summary_stats['per_concept'].items()]
    concept_presence.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 concepts by presence in reasoning:")
    for i, (concept, rate) in enumerate(concept_presence[:10], 1):
        print(f"  {i:2d}. {concept:20s} {rate:.1%}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
