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
import torch

from models.r1_onevision import R1OnevisionAPI


def load_input_json(path: str) -> Dict[str, Any]:
    """Load input JSON file with concept inspection data."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_judge_prompt(concepts: List[str], text: str, text_type: str) -> str:
    """
    Create LLM judge prompt for concept presence analysis.

    Args:
        concepts: List of concept strings to check
        text: Text to analyze (reasoning or answer)
        text_type: Description of text type ("reasoning trace" or "answer")

    Returns:
        Formatted prompt string
    """
    concept_list = ", ".join([f'"{c}"' for c in concepts])

    prompt = f"""You are analyzing whether specific concepts are semantically present in a {text_type} about skin lesion diagnosis.

**Concepts to check:**
{concept_list}

**Text to analyze:**
{text}

**Instructions:**
For each concept, determine if it is semantically present in the text. Be INCLUSIVE and mark a concept as "present" if ANY of the following apply:

1. **Direct mention**: The exact word appears in the text
   - Example: "nail" in concept list, "nail bed" or "nails" in text → mark "nail" as present

2. **Synonyms or related terms**: Semantically equivalent words
   - Example: "footwear" in concept list, "shoes" in text → mark "footwear" as present

3. **Part-whole relationships**: Text mentions something that includes or is part of the concept
   - If text mentions "nail", "toenail", "fingernail", or "nail bed" → mark BOTH "nail" AND "nails" as present
   - If text mentions "toe", "toenail", or "subungual" → mark "toe", "foot", "nail", and "nails" as present
   - If text mentions body parts → mark related singular/plural forms as present

4. **Strong contextual associations**: Text discusses something strongly associated with the concept
   - If text mentions "trauma", "pain", "infection", "lesion", or "abscess" → mark "hurt" as present
   - If text mentions "discoloration" in nail context → mark "color" and "paint" as present
   - If text discusses foot/toe pathology → mark "foot", "toe", "shoe" as present

5. **Clinical context**: For medical texts, consider clinical relationships
   - Nail conditions (onychomycosis, paronychia, subungual) → mark "nail", "nails", potentially "toe" or "foot"
   - Purple coloration/marking → mark "purple" as present

**Key principle**: When uncertain, INCLUDE the concept rather than exclude it. We want to capture all plausible semantic connections.

Return ONLY a JSON object with this format:
{{
  "present": ["concept1", "concept2", ...],
  "absent": ["concept3", "concept4", ...]
}}

Do NOT include explanations or additional text outside the JSON."""

    return prompt


def parse_judge_response(response: Any, concepts: List[str]) -> Dict[str, Any]:
    """
    Parse LLM judge response to extract concept presence.

    Implements robust parsing with multiple fallback strategies:
    1. Direct JSON parsing
    2. Extract from markdown code blocks
    3. Regex extraction of JSON-like structure

    Args:
        response: Raw LLM response (str or dict with 'reasoning'/'answer' keys)
        concepts: List of all concepts to validate against

    Returns:
        Dict with keys:
        - concepts_present: List of concepts marked as present
        - concepts_absent: List of concepts marked as absent
        - raw_llm_response: Original LLM response (answer portion)
        - judge_reasoning: The model's reasoning process (if available)
        - parse_successful: Boolean indicating parse success
    """
    # Extract answer text if response is a dict (when return_reasoning=True)
    judge_reasoning = None
    if isinstance(response, dict):
        judge_reasoning = response.get('reasoning', '')
        response_text = response.get('answer', response.get('full_output', ''))
    else:
        response_text = response

    result = None

    # Try direct JSON parsing
    try:
        result = json.loads(response_text.strip())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting from markdown code blocks
    if result is None:
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Try regex fallback to find JSON-like structure
    if result is None:
        match = re.search(r'\{[^{}]*"present"[^{}]*"absent"[^{}]*\}', response_text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    # Try more permissive regex (allow nested braces)
    if result is None:
        match = re.search(r'\{.*?"present".*?\[.*?\].*?"absent".*?\[.*?\].*?\}', response_text, re.DOTALL)
        if match:
            try:
                # Extract the matched string
                json_str = match.group(0)
                result = json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # If parsing failed, return failure response
    if result is None:
        return {
            'concepts_present': [],
            'concepts_absent': concepts,
            'raw_llm_response': response_text if response_text else '', 
            'judge_reasoning': judge_reasoning if judge_reasoning else None, 
            'parse_successful': False
        }

    # Validate result structure
    if not isinstance(result, dict) or 'present' not in result or 'absent' not in result:
        return {
            'concepts_present': [],
            'concepts_absent': concepts,
            'raw_llm_response': response_text if response_text else '',
            'judge_reasoning': judge_reasoning if judge_reasoning else None,
            'parse_successful': False
        }

    return {
        'concepts_present': result.get('present', []),
        'concepts_absent': result.get('absent', []),
        'raw_llm_response': response_text if response_text else '', 
        'judge_reasoning': judge_reasoning if judge_reasoning else None,
        'parse_successful': True
    }


def query_llm_judge(model: R1OnevisionAPI, prompt: str, temperature: float = 0.0,
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
    model: R1OnevisionAPI,
    concepts: List[str],
    reasoning: str,
    answer: str,
    temperature: float = 0.0,
    max_new_tokens: int = 2048
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze a single trace for concept presence.

    Args:
        model: R1OnevisionAPI instance
        concepts: List of concept strings
        reasoning: Reasoning trace text
        answer: Answer text
        temperature: LLM temperature
        max_new_tokens: Max tokens per response

    Returns:
        Tuple of (reasoning_analysis, answer_analysis) dicts
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

    # Analyze answer
    if answer and answer.strip():
        answer_prompt = create_judge_prompt(concepts, answer, "answer")
        answer_response = query_llm_judge(model, answer_prompt, temperature, max_new_tokens)
        answer_result = parse_judge_response(answer_response, concepts)
    else:
        answer_result = {
            'concepts_present': [],
            'concepts_absent': concepts,
            'raw_llm_response': 'N/A (empty answer field)',
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
        '--judge_model',
        type=str,
        default='R1-Onevision-7B',
        help='Model to use as judge (default: R1-Onevision-7B)'
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
    print(f"Initializing {args.judge_model} as LLM judge...")
    judge_model = R1OnevisionAPI(model_name=args.judge_model)
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
            'image_path': trace['image_path'],
            'label': trace['label'],
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
