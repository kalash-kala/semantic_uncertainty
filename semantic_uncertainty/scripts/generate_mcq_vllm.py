#!/usr/bin/env python3
"""
Generate MCQ questions from semantic uncertainty CSV files using a local vLLM model.

Serves Qwen2.5-32B-Instruct-AWQ (or any HF-compatible model) via vLLM's offline
LLM class and processes each CSV row individually, appending results to a JSON file.

Usage:
    python generate_mcq_vllm.py \\
        --input  /path/to/detail_xxx.csv \\
        --output /path/to/output_mcq.json \\
        --model  /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ \\
        --max-retries 3

    nohup command:
    nohup python3 generate_mcq_vllm.py --input /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5/detail_sciq__llama_entropy_0.5_Correct.csv --output /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5__mcq/detail_sciq__llama_entropy_0.5_Correct_mcq.json --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --max-retries 3 > /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5__mcq/detail_sciq__llama_entropy_0.5_Correct_mcq.log 2>&1 &
"""

import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path

from vllm import LLM, SamplingParams

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — insert yours below (triple-quoted to handle any quote style)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
"""

# ──────────────────────────────────────────────────────────────────────────────
# MCQ generation rules — strict format enforcement
# ──────────────────────────────────────────────────────────────────────────────
MCQ_RULES = """You are a strict MCQ generator. You will be given a question, its ground truth answer, a model-predicted answer, sampled model answers, and a label. Your task is to produce exactly 4 options (A, B, C, D) for a multiple-choice question.

=== LABEL-SPECIFIC RULES ===

LABEL "correct":
  - One option MUST be the exact ground truth answer.
  - The other 3 options must be plausible but WRONG answers related to the topic.
  - None of the 3 distractors may be identical to each other or to the ground truth.

LABEL "AH":
  - One option MUST be the exact ground truth answer.
  - One option MUST be the exact model predicted answer.
  - The remaining 2 options must be plausible but WRONG answers related to the topic.
  - None of the 4 options may be identical to each other.

LABEL "UH":
  - One option MUST be the exact ground truth answer.
  - One option MUST be the exact model predicted answer.
  - The remaining 2 options should be drawn from the sampled model answers if any of them differ from both the ground truth and predicted answer; otherwise generate plausible but WRONG alternatives.
  - None of the 4 options may be identical to each other.

=== GENERAL RULES (apply to ALL labels) ===
  - Options must NOT be paraphrases or plural variants of each other — they must be meaningfully distinct.
  - Options must be relevant to the question being asked.
  - For NUMERIC answers: distractors must be genuinely different numbers. 232 and 232.0 are the SAME — do not use them as separate options. Choose numbers that are numerically close but distinct (e.g. if answer is 4, use 3, 5, 6 — not 4.0 or 04).
  - Do NOT add explanations, prefixes, or any text outside the JSON block.
  - The correct_option field must contain the letter (A, B, C, or D) whose text exactly matches the ground truth answer you placed in the options.
  - The correct_answer field must contain the exact text you placed in the options for the correct option.

=== REQUIRED OUTPUT FORMAT ===
You MUST return a single JSON object and nothing else. No markdown. No code fences. No explanation. Start your response with { and end with }.

The JSON must have exactly these three keys:
  "options"        — a list of exactly 4 objects, each with "label" (A/B/C/D) and "text"
  "correct_option" — a single uppercase letter: A, B, C, or D
  "correct_answer" — the exact text of the correct option

Example of a valid response (for illustration only — do not copy these values):
{"options": [{"label": "A", "text": "mitosis"}, {"label": "B", "text": "meiosis"}, {"label": "C", "text": "apoptosis"}, {"label": "D", "text": "necrosis"}], "correct_option": "A", "correct_answer": "mitosis"}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_list_string(s: str) -> list:
    """Parse a Python list literal string like \"['hearing']\" into a list."""
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        return [str(result)]
    except Exception:
        return [s.strip("[]'\"")] if s else []


def extract_true_answer(ground_truth_str: str) -> str:
    items = parse_list_string(ground_truth_str)
    return items[0].strip() if items else ground_truth_str.strip()


def load_existing_output(output_path: str) -> list:
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                print(f"WARNING: Could not parse existing output at {output_path}. Starting fresh.")
    return []


def save_output(output_path: str, data: list) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_user_prompt(row: dict, true_answer: str) -> str:
    label    = row["label"].strip()
    question = row["question"].strip()
    prediction = row["prediction"].strip()
    sampled  = row.get("sampled_answers", "").strip()

    return (
        f"{MCQ_RULES}\n\n"
        f"=== INPUT ===\n"
        f"Question        : {question}\n"
        f"Ground truth    : {true_answer}\n"
        f"Model prediction: {prediction}\n"
        f"Sampled answers : {sampled}\n"
        f"Label           : {label}\n\n"
        f"=== YOUR JSON RESPONSE ==="
    )


def build_chat_prompt(tokenizer, row: dict, true_answer: str) -> str:
    """Format the prompt using the model's chat template."""
    messages = []

    system_text = SYSTEM_PROMPT.strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})

    messages.append({"role": "user", "content": build_user_prompt(row, true_answer)})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_from_response(text: str) -> str:
    """
    Try to extract a JSON object from the response text.
    Handles cases where the model adds surrounding explanation text.
    """
    text = strip_code_fences(text)

    # Direct parse attempt
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


def validate_mcq_json(data: dict) -> None:
    """Raise ValueError if the parsed JSON does not meet the required schema."""
    if "options" not in data or "correct_option" not in data or "correct_answer" not in data:
        raise ValueError(f"Missing required keys. Got: {list(data.keys())}")

    options = data["options"]
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError(f"'options' must be a list of exactly 4 items. Got {len(options) if isinstance(options, list) else type(options)}")

    labels = [o.get("label") for o in options]
    if labels != ["A", "B", "C", "D"]:
        raise ValueError(f"Option labels must be exactly ['A','B','C','D']. Got {labels}")

    for opt in options:
        if "label" not in opt or "text" not in opt:
            raise ValueError(f"Each option must have 'label' and 'text'. Got: {opt}")
        if not str(opt["text"]).strip():
            raise ValueError(f"Option text must not be empty. Got: {opt}")

    if data["correct_option"] not in {"A", "B", "C", "D"}:
        raise ValueError(f"'correct_option' must be A/B/C/D. Got: {data['correct_option']!r}")

    # Verify correct_option letter actually appears in options
    option_map = {o["label"]: o["text"] for o in options}
    declared_answer = option_map.get(data["correct_option"], "")
    if not declared_answer:
        raise ValueError(f"correct_option '{data['correct_option']}' not found in options.")

    # Warn (not fail) if correct_answer text doesn't match the letter's text
    if data["correct_answer"].strip().lower() != declared_answer.strip().lower():
        raise ValueError(
            f"correct_answer '{data['correct_answer']}' does not match "
            f"option {data['correct_option']} text '{declared_answer}'"
        )


def build_mcq_question_full(question: str, options: list) -> str:
    lines = [question] + [f"{opt['label']}. {opt['text']}" for opt in options]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────────────────────

def process_csv(
    input_path: str,
    output_path: str,
    model_path: str,
    max_retries: int = 3,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_new_tokens: int = 512,
) -> None:
    # ── Load model once ──────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        quantization="awq_marlin",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="float16",
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.\n")

    sampling_params = SamplingParams(
        temperature=0.0,       # greedy — deterministic JSON output
        max_tokens=max_new_tokens,
        stop=["}\n\n", "\n\n\n"],  # stop after JSON block closes
    )

    # ── Resume support ───────────────────────────────────────────────────────
    results = load_existing_output(output_path)
    processed_questions = {r["question"] for r in results}
    print(f"Resuming from {len(results)} already-processed entries.")

    with open(input_path, newline="", encoding="utf-8") as csvfile:
        rows = list(csv.DictReader(csvfile))

    total    = len(rows)
    new_count = 0
    skipped   = 0
    errors    = 0

    for i, row in enumerate(rows):
        question = row["question"].strip()

        if question in processed_questions:
            skipped += 1
            continue

        true_answer   = extract_true_answer(row["ground_truth"])
        greedy_answer = row["prediction"].strip()

        print(f"[{i + 1}/{total}] {question[:75]}...")

        # ── Retry loop ───────────────────────────────────────────────────────
        mcq_data = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                prompt = build_chat_prompt(tokenizer, row, true_answer)
                outputs = llm.generate([prompt], sampling_params)
                raw_text = outputs[0].outputs[0].text.strip()

                print(f"    Raw model output:\n{raw_text[:300]}\n")

                json_str = extract_json_from_response(raw_text)
                mcq_data = json.loads(json_str)
                # validate_mcq_json(mcq_data)  # COMMENTED OUT for debugging
                break  # success

            except Exception as e:
                last_error = e
                print(f"  Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    print("  Retrying with temperature bump...")
                    # Bump temperature slightly on retry so we don't get the same bad output
                    sampling_params_retry = SamplingParams(
                        temperature=0.3 * attempt,
                        max_tokens=max_new_tokens,
                        stop=["}\n\n", "\n\n\n"],
                    )
                    sampling_params_used = sampling_params_retry
                else:
                    sampling_params_used = sampling_params  # reset for next row

        if mcq_data is None:
            print(f"  SKIPPING row {i + 1} after {max_retries} failed attempts. Last error: {last_error}")
            errors += 1
            continue

        # ── Build full entry ─────────────────────────────────────────────────
        options        = mcq_data["options"]
        correct_option = mcq_data["correct_option"]
        correct_answer = mcq_data["correct_answer"]

        entry = {
            "model":           row["model"],
            "question":        question,
            "ground_truth":    row["ground_truth"],
            "prediction":      row["prediction"],
            "sampled_answers": row["sampled_answers"],
            "semantic_entropy": float(row["semantic_entropy"]),
            "correct":         row["correct"].strip().lower() == "true",
            "label":           row["label"].strip(),
            "true_answer":     true_answer,
            "greedy_answer_extracted": greedy_answer,
            "mcq_question_full": build_mcq_question_full(question, options),
            "mcq_meta": {
                "question":       question,
                "options":        options,
                "correct_option": correct_option,
                "correct_answer": correct_answer,
            },
        }

        results.append(entry)
        processed_questions.add(question)
        save_output(output_path, results)  # persist after every row
        new_count += 1

    print(
        f"\nFinished. New: {new_count}  |  Skipped (already done): {skipped}  |  Errors: {errors}"
    )
    print(f"Output: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MCQs from semantic uncertainty CSV using a local vLLM model."
    )
    parser.add_argument("--input",  required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--model",
        default="/home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ",
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.70,
        help="Fraction of GPU memory vLLM may use (default: 0.90)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate per row (default: 1024)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts per row on JSON parse/validation failure (default: 3)",
    )
    args = parser.parse_args()

    process_csv(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        max_retries=args.max_retries,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
