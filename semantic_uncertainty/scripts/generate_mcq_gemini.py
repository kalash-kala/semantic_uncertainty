#!/usr/bin/env python3
"""
Generate MCQ questions from semantic uncertainty CSV files using Gemini Flash.

Usage:
    python generate_mcq_gemini.py \\
        --input  /path/to/detail_xxx.csv \\
        --output /path/to/output_mcq.json \\
        --env    /path/to/.env            \\
        --sleep  1.5
"""

import argparse
import ast
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — insert yours below (triple-quoted to handle any quote style)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
"""

# ──────────────────────────────────────────────────────────────────────────────
# MCQ generation rules injected into every user prompt
# ──────────────────────────────────────────────────────────────────────────────
MCQ_RULES = """Here are some rules that you must follow:

1. For questions with label "correct": create options containing the ground truth.
   The other 3 options should be related to the answer for confusion, with none of
   them being the same as each other or the ground truth.

2. For questions labelled "AH": one option must be the ground truth, one must be
   the predicted answer. The remaining 2 should be related to the first 2 but
   must be distinct from each other and from the ground truth and prediction.
   Do not use options from the sampled model answers.

3. For questions labelled "UH": one option must be the ground truth, one must be
   the predicted answer. The remaining 2 may be drawn from the sampled model
   answers provided they differ from both the ground truth and prediction;
   otherwise create plausible alternatives that are similar but genuinely different.
   Do not use options from the sampled model answers.

4. Options must NOT be paraphrases or simple plurals of each other — they must be
   meaningfully distinct and relevant to the question.

5. For numeric answers: follow the same rules above. Distractors must be distinct
   numbers (e.g. 232 and 232.0 are the same — do not use those as separate options).

Return ONLY a valid JSON object — no markdown, no extra text — in exactly this format:
{
  "options": [
    {"label": "A", "text": "..."},
    {"label": "B", "text": "..."},
    {"label": "C", "text": "..."},
    {"label": "D", "text": "..."}
  ],
  "correct_option": "<letter A/B/C/D whose text matches the ground truth>",
  "correct_answer": "<exact ground truth text as placed in the options>"
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_api_key(env_path: str) -> str:
    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"GEMINI_API_KEY not found in {env_path}")
    return api_key


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
    label = row["label"].strip()
    question = row["question"].strip()
    prediction = row["prediction"].strip()
    sampled = row.get("sampled_answers", "").strip()

    return (
        f"You are generating a multiple-choice question (MCQ) for a science quiz dataset.\n\n"
        f"Question: {question}\n"
        f"Ground truth answer: {true_answer}\n"
        f"Model predicted answer: {prediction}\n"
        f"Sampled model answers: {sampled}\n"
        f"Label: {label}\n\n"
        f"{MCQ_RULES}"
    )


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_gemini(client: genai.Client, model_id: str, user_prompt: str) -> dict:
    """Send a prompt to Gemini Flash and return the parsed JSON dict."""
    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    config_kwargs = {
        "temperature": 0.7,
    }
    if SYSTEM_PROMPT.strip():
        config_kwargs["system_instruction"] = SYSTEM_PROMPT.strip()

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    raw = response.text.strip()
    cleaned = strip_code_fences(raw)
    return json.loads(cleaned)


def build_mcq_question_full(question: str, options: list) -> str:
    lines = [question] + [f"{opt['label']}. {opt['text']}" for opt in options]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────

def process_csv(
    input_path: str,
    output_path: str,
    api_key: str,
    model_id: str = "gemini-2.0-flash",
    sleep_seconds: float = 1.5,
) -> None:
    client = genai.Client(api_key=api_key)

    # Resume: skip questions already in the output file
    results = load_existing_output(output_path)
    processed_questions = {r["question"] for r in results}
    print(f"Resuming from {len(results)} already-processed entries.")

    with open(input_path, newline="", encoding="utf-8") as csvfile:
        rows = list(csv.DictReader(csvfile))

    total = len(rows)
    new_count = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(rows):
        question = row["question"].strip()

        if question in processed_questions:
            skipped += 1
            continue

        true_answer = extract_true_answer(row["ground_truth"])
        greedy_answer = row["prediction"].strip()

        print(f"[{i + 1}/{total}] {question[:70]}...")

        try:
            user_prompt = build_user_prompt(row, true_answer)
            mcq_data = call_gemini(client, model_id, user_prompt)

            options = mcq_data["options"]
            correct_option = mcq_data["correct_option"]
            correct_answer = mcq_data["correct_answer"]

            entry = {
                "model": row["model"],
                "question": question,
                "ground_truth": row["ground_truth"],
                "prediction": row["prediction"],
                "sampled_answers": row["sampled_answers"],
                "semantic_entropy": float(row["semantic_entropy"]),
                "correct": row["correct"].strip().lower() == "true",
                "label": row["label"].strip(),
                "true_answer": true_answer,
                "greedy_answer_extracted": greedy_answer,
                "mcq_question_full": build_mcq_question_full(question, options),
                "mcq_meta": {
                    "question": question,
                    "options": options,
                    "correct_option": correct_option,
                    "correct_answer": correct_answer,
                },
            }

            results.append(entry)
            processed_questions.add(question)
            save_output(output_path, results)  # write after every row so progress is never lost
            new_count += 1

        except Exception as e:
            print(f"  ERROR on row {i + 1} (skipping): {e}")
            errors += 1

        time.sleep(sleep_seconds)

    print(
        f"\nFinished. New: {new_count}  |  Skipped (already done): {skipped}  |  Errors: {errors}"
    )
    print(f"Output: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MCQs from semantic uncertainty CSV using Gemini Flash."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--env",
        default="/home/kalashkala/Perception-R1/.env",
        help="Path to .env file containing GEMINI_API_KEY",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model ID (default: gemini-1.5-flash)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=4.1,
        help="Seconds to sleep between API calls (default: 1.5)",
    )
    args = parser.parse_args()

    api_key = load_api_key(args.env)
    process_csv(
        input_path=args.input,
        output_path=args.output,
        api_key=api_key,
        model_id=args.model,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
