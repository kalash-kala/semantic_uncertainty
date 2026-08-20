#!/usr/bin/env python3
"""
LLM Judge Verdict for Semantic Uncertainty CSVs (Claude API backend).

Uses the Anthropic Claude API as the LLM judge — much faster than local GPU
inference because calls run concurrently.

Compares ground_truth vs low_t_generation for every row and adds a boolean
claude_verdict column (True/False) to the output CSV.

Output is saved as <original_stem>_claude_verdict.csv in the same directory.

Usage:
    python llm_judge_verdict_claude.py
    python llm_judge_verdict_claude.py --model claude-opus-4-7 --max_workers 20
    python llm_judge_verdict_claude.py --model claude-sonnet-4-6 --max_workers 20
    nohup python llm_judge_verdict_claude.py --model claude-sonnet-4-6 --max_workers 20 > claude_verdicts.log 2>&1 &

Set ANTHROPIC_API_KEY in environment before running.

Nohup Usage:
    nohup python3 llm_judge_verdict_claude.py --max_workers 20 > llm_judge_verdict_claude.log 2>&1 &

    Edit FILE_LIST below to specify which CSVs to process.
"""

import os
import sys
import argparse
import ast
import json
import re
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd

# Setup logging with immediate flush for nohup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ─────────────────────────────────────────────────────────────────────────────
# FILE LIST — add one CSV path per entry
# ─────────────────────────────────────────────────────────────────────────────

FILE_LIST = [
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_llama_gsm8k_combined.csv",
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_mistral_gsm8k_combined.csv",
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_qwen_gsm8k_combined.csv",
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_gemma_gsm8k_combined.csv",
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/sciq/uncertainty_run_llama_sciq_combined_llm_verdict.csv"
    "/home/kalashkala/Datasets/Semantic-Uncertainty/triviaqa/uncertainty_run_llama_triviaqa_combined_50K_llm_verdict.csv"
]

# ─────────────────────────────────────────────────────────────────────────────
# Judge prompts
# ─────────────────────────────────────────────────────────────────────────────

def get_judge_prompt(dataset: str):
    """Return (system, user_template) for the given dataset."""

    if dataset == "sciq":
        system = (
            "You are an expert science answer evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it conveys the same scientific meaning as any of the valid answers.\n"
            "Focus on the meaning being conveyed, not the surface form — synonyms, paraphrases, abbreviations, "
            "and different phrasings are acceptable.\n"
            "If a numeric value appears in the ground truth, focus on the value only — formatting differences "
            "such as 5.0 vs 5, $100 vs 100, 1,000 vs 1000, or 4 vs four are the same.\n"
            "A partial or abbreviated answer is acceptable if it unambiguously identifies the correct answer "
            "given the question context.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it matches ANY one of them.\n"
            "If the proposed answer is empty, irrelevant, or does not address the question, answer no.\n"
            "Your entire response must be a single word: yes or no. No explanation."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, does the proposed answer convey the same scientific meaning "
            "as at least one of the valid answers? yes or no."
        )

    elif dataset == "svamp":
        system = (
            "You are a math answer evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it is numerically equivalent to any of the valid answers.\n"
            "Focus on the numerical value only — formatting differences such as 5.0 vs 5, $100 vs 100, "
            "or 1,000 vs 1000 are the same.\n"
            "If the proposed answer contains reasoning steps or working, evaluate only the final numerical value.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it equals ANY one of them.\n"
            "If the proposed answer is empty, irrelevant, or does not address the question, answer no.\n"
            "Your entire response must be a single word: yes or no. No explanation."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, is the proposed answer numerically equivalent to at least "
            "one of the valid answers? yes or no."
        )

    elif dataset == "triviaqa":
        system = (
            "You are an expert factual QA evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it refers to the same entity or concept as any of the valid answers.\n"
            "Focus on the meaning and identity being conveyed — aliases, abbreviations, and common name variations "
            "are acceptable.\n"
            "If the answer is numerical, focus on the value being expressed — formatting differences such as "
            "5.0 vs 5, $100 vs 100, 1,000 vs 1000, or 4 vs four are the same.\n"
            "A partial or abbreviated answer is acceptable if it unambiguously identifies the correct entity "
            "given the question context.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it matches ANY one of them.\n"
            "If the proposed answer is empty, irrelevant, or does not address the question, answer no.\n"
            "Your entire response must be a single word: yes or no. No explanation."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, does the proposed answer refer to the same entity or concept "
            "as at least one of the valid answers? yes or no."
        )

    elif dataset == "gsm8k":
        system = (
            "You are an expert math answer evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it is equivalent to any of the valid answers.\n"
            "Focus on the value being expressed — formatting differences such as 5.0 vs 5, $100 vs 100, "
            "1,000 vs 1000, or 4 vs four are the same.\n"
            "If the proposed answer contains reasoning steps or working (e.g. ends with #### 42), evaluate "
            "only the final numerical value.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it equals ANY one of them.\n"
            "If the proposed answer is empty, irrelevant, or does not address the question, answer no.\n"
            "Your entire response must be a single word: yes or no. No explanation."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, is the proposed answer equivalent to at least one of the "
            "valid answers? yes or no."
        )

    else:
        system = (
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "Determine whether the proposed answer conveys the same meaning as any of the valid answers.\n"
            "Focus on the meaning being conveyed, not surface form — synonyms, paraphrases, and abbreviations "
            "are acceptable.\n"
            "A partial or abbreviated answer is acceptable if it unambiguously identifies the correct answer "
            "given the context.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it matches ANY one of them.\n"
            "If the proposed answer is empty, irrelevant, or does not address the question, answer no.\n"
            "Your entire response must be a single word: yes or no. No explanation."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, does the proposed answer convey the same meaning as at least "
            "one of the valid answers? yes or no."
        )

    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_dataset(filename: str) -> str:
    name = filename.lower()
    if "sciq" in name:
        return "sciq"
    elif "svamp" in name:
        return "svamp"
    elif "triviaqa" in name:
        return "triviaqa"
    return "generic"


def normalize_text(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r'[^a-z0-9\s]', '', x)
    x = re.sub(r'\s+', ' ', x)
    return x


def normalize_math(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r'[$€£¥]', '', x)
    x = re.sub(r'(?<=\d),(?=\d{3})', '', x)
    x = re.sub(r'[a-zA-Z\s]+$', '', x).strip()
    try:
        val = float(x)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return x.lower()


def normalize_answer(x: str, dataset: str) -> str:
    if dataset == "svamp":
        return normalize_math(x)
    return normalize_text(x)


def parse_ground_truth(gt_str: str, dataset: str) -> list[str]:
    try:
        parsed = ast.literal_eval(str(gt_str))
        if isinstance(parsed, list) and parsed:
            return [normalize_answer(x, dataset) for x in parsed if str(x).strip()]
        return [normalize_answer(parsed, dataset)]
    except (ValueError, SyntaxError):
        return [normalize_answer(gt_str, dataset)]


def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse progress file {progress_file}. Starting fresh.")
    return {}


def save_progress(progress_file: str, verdicts: dict) -> None:
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(verdicts, f)


def parse_verdict(text: str) -> bool:
    cleaned = text.strip().lower()
    if re.search(r'\byes\b', cleaned):
        return True
    if re.search(r'\bno\b', cleaned):
        return False
    logger.warning(f"Ambiguous verdict {text!r} — treating as 'no'")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def judge_single_row(
    client: anthropic.Anthropic,
    row: dict,
    dataset: str,
    model: str,
) -> bool:
    """Call the Claude API for one row and return the boolean verdict."""
    system_prompt, user_template = get_judge_prompt(dataset)

    gt_list = parse_ground_truth(row["ground_truth"], dataset)
    if len(gt_list) == 1:
        gt_formatted = f"  1. {gt_list[0]}"
    else:
        gt_formatted = "\n".join(f"  {i + 1}. {ans}" for i, ans in enumerate(gt_list))

    # Pass the raw prediction — Claude handles semantic normalization directly.
    prediction = str(row["low_t_generation"]).strip()

    user_text = user_template.format(
        question=str(row["question"]).strip(),
        ground_truth=gt_formatted,
        prediction=prediction,
    )

    response = client.messages.create(
        model=model,
        max_tokens=16,
        system=system_prompt,
        messages=[{"role": "user", "content": user_text}],
    )

    raw_text = response.content[0].text if response.content else ""
    return parse_verdict(raw_text)


def run_judge(
    df: pd.DataFrame,
    client: anthropic.Anthropic,
    model: str,
    max_workers: int,
    progress_file: str,
    dataset: str,
) -> dict:
    """
    Run LLM judge over all rows concurrently.
    Returns dict mapping str(index) → bool verdict.
    Supports resume via progress_file.
    """
    verdicts = load_progress(progress_file)
    already_done = len(verdicts)
    if already_done:
        logger.info(f"Resuming: {already_done} rows already judged.")

    pending_indices = [idx for idx in df.index if str(idx) not in verdicts]
    pending_rows = df.loc[pending_indices].to_dict("records")

    logger.info(f"Rows to judge: {len(pending_indices)}")

    save_interval = 50
    completed = 0

    def judge_with_index(idx, row):
        verdict = judge_single_row(client, row, dataset, model)
        return idx, verdict

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(judge_with_index, idx, row): idx
            for idx, row in zip(pending_indices, pending_rows)
        }

        for future in as_completed(futures):
            idx, verdict = future.result()
            verdicts[str(idx)] = verdict
            completed += 1

            if completed % save_interval == 0:
                save_progress(progress_file, verdicts)
                total = already_done + len(pending_indices)
                done_so_far = already_done + completed
                logger.info(f"Progress: {done_so_far}/{total} | yes_count: {sum(verdicts.values())}")

    save_progress(progress_file, verdicts)
    return verdicts


def process_single_csv(
    input_path: Path,
    output_path: Path,
    client: anthropic.Anthropic,
    model: str,
    max_workers: int,
    dataset: str,
    verdict_column: str = "claude_verdict",
) -> bool:
    """Process one CSV: judge all rows, add verdict column, save."""
    progress_file = str(output_path.with_suffix(".progress.json"))

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    if len(df) == 0:
        logger.info("Empty CSV. Skipping.")
        return False

    verdicts = run_judge(
        df=df,
        client=client,
        model=model,
        max_workers=max_workers,
        progress_file=progress_file,
        dataset=dataset,
    )

    df[verdict_column] = df.index.map(lambda idx: verdicts.get(str(idx), None))

    confirmed = sum(verdicts.values())
    rejected = len(verdicts) - confirmed

    logger.info("=" * 60)
    logger.info("VERDICT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows judged    : {len(verdicts)}")
    logger.info(f"Claude says correct  : {confirmed}")
    logger.info(f"Claude says incorrect: {rejected}")
    logger.info(f"Accuracy (Claude)    : {confirmed / len(verdicts) * 100:.1f}%")
    logger.info("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    if os.path.exists(progress_file):
        os.remove(progress_file)
        logger.info(f"Removed progress file: {progress_file}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add claude_verdict column to semantic uncertainty CSVs using Claude as judge."
    )
    p.add_argument("--model", default="claude-opus-4-7",
                   help="Claude model ID (default: claude-opus-4-7)")
    p.add_argument("--max_workers", type=int, default=10,
                   help="Number of concurrent API calls (default: 10)")
    p.add_argument("--api_key", type=str, default=None,
                   help="Anthropic API key (default: reads ANTHROPIC_API_KEY env var)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip CSVs that already have a _llm_verdict output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not FILE_LIST:
        logger.error("FILE_LIST is empty. Add CSV paths to FILE_LIST at the top of the script.")
        return

    for f in FILE_LIST:
        if not Path(f).exists():
            raise FileNotFoundError(f"File not found: {f}")

    # max_retries=5: SDK auto-retries 429 and 5xx with exponential backoff
    client = anthropic.Anthropic(api_key=args.api_key, max_retries=5)

    logger.info("=" * 60)
    logger.info(f"CLAUDE VERDICT (API) — {len(FILE_LIST)} file(s) to process")
    logger.info("=" * 60)
    logger.info(f"Model       : {args.model}")
    logger.info(f"Concurrency : {args.max_workers} parallel calls")
    for i, f in enumerate(FILE_LIST, 1):
        logger.info(f"  {i:2d}. {f}")
    logger.info("")

    total_start = time.time()
    processed = 0
    skipped = 0
    failed = 0

    for i, csv_path_str in enumerate(FILE_LIST, 1):
        csv_path = Path(csv_path_str)
        output_path = csv_path.parent / (csv_path.stem + "_claude_verdict.csv")

        if args.skip_existing and output_path.exists():
            logger.info(f"[{i}/{len(FILE_LIST)}] SKIPPING (output exists): {csv_path.name}")
            skipped += 1
            continue

        dataset = detect_dataset(str(csv_path))

        logger.info(f"[{i}/{len(FILE_LIST)}] Processing: {csv_path.name}")
        logger.info(f"Dataset type: {dataset} → Output: {output_path.name}")

        file_start = time.time()
        try:
            success = process_single_csv(
                input_path=csv_path,
                output_path=output_path,
                client=client,
                model=args.model,
                max_workers=args.max_workers,
                dataset=dataset,
                verdict_column="claude_verdict",
            )
            elapsed = time.time() - file_start
            logger.info(f"Completed in {elapsed:.1f}s")
            processed += 1
        except Exception as e:
            elapsed = time.time() - file_start
            logger.error(f"FAILED after {elapsed:.1f}s: {e}", exc_info=True)
            failed += 1

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)
    logger.info(f"Total files  : {len(FILE_LIST)}")
    logger.info(f"Processed    : {processed}")
    logger.info(f"Skipped      : {skipped}")
    logger.info(f"Failed       : {failed}")
    logger.info(f"Total time   : {total_elapsed / 60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
