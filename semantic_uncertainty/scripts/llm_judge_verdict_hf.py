#!/usr/bin/env python3
"""
LLM Judge Verdict for Semantic Uncertainty CSVs (HuggingFace Transformers backend).

Uses AutoModelForCausalLM with device_map="auto" — works for large models like
Llama-3.3-70B that exceed vLLM's KV-cache pre-allocation limit.

Compares ground_truth vs low_t_generation for every row using an LLM judge and
adds a boolean LLM_verdict column (True/False) to the output CSV.

Output is saved as <original_stem>_llm_verdict.csv in the same directory.

Usage:
    python llm_judge_verdict_hf.py --model /data/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --cuda_device 0,1

Nohup Usage:
    nohup python3 llm_judge_verdict_hf.py --model /data/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b --cuda_device 0,1 --batch_size 32 > llm_judge_verdict_gsm8k_hf.log 2>&1 &

    Edit FILE_LIST below to specify which CSVs to process.
"""

import os
import argparse
import ast
import json
import re
import time
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# FILE LIST — add one CSV path per entry
# ─────────────────────────────────────────────────────────────────────────────

FILE_LIST = [
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_llama_gsm8k_combined.csv",
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_mistral_gsm8k_combined.csv",
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_qwen_gsm8k_combined.csv",
    # "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_gemma_gsm8k_combined.csv",
    "/home/kalashkala/semantic_uncertainty/semantic_uncertainty/scripts/uncertainty_run_llama_sciq_combined_empty_verdict.csv"
]

# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt
# ─────────────────────────────────────────────────────────────────────────────

def get_judge_prompt(dataset: str):

    if dataset == "sciq":
        system = (
            "You are an expert science answer evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it conveys the same scientific meaning as any of the valid answers.\n"
            "Focus on the meaning being conveyed, not the surface form — synonyms, paraphrases, and abbreviations are acceptable.\n"
            "If numeric answer is there in ground truth, focus on the value only — formatting differences such as 5.0 vs 5, $100 vs 100, 1,000 vs 1000 or 4 vs four are the same.\n"
            "A partial or abbreviated answer is acceptable if it unambiguously identifies the correct answer given the question context.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it matches ANY one of them.\n"
            "Respond with exactly one word: yes or no."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, does the proposed answer convey the same scientific meaning as at least one of the valid answers? "
            "Respond with exactly one word: yes or no."
        )

    elif dataset == "svamp":
        system = (
            "You are a math answer evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it is numerically equivalent to any of the valid answers.\n"
            "Focus on the numerical value only — formatting differences such as 5.0 vs 5, $100 vs 100, or 1,000 vs 1000 are the same.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it equals ANY one of them.\n"
            "Respond with exactly one word: yes or no."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, is the proposed answer numerically equivalent to at least one of the valid answers? "
            "Respond with exactly one word: yes or no."
        )

    elif dataset == "triviaqa":
        system = (
            "You are an expert factual QA evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it refers to the same entity or concept as any of the valid answers.\n"
            "Focus on the meaning and identity being conveyed — aliases, abbreviations, and common name variations are acceptable.\n"
            "If the answer is numerical then focus on the value being expressed — formatting differences such as 5.0 vs 5, $100 vs 100, or 1,000 vs 1000, 4 vs four are the same.\n"
            "A partial or abbreviated answer is acceptable if it unambiguously identifies the correct entity given the question context.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it matches ANY one of them.\n"
            "Respond with exactly one word: yes or no."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, does the proposed answer refer to the same entity or concept as at least one of the valid answers? "
            "Respond with exactly one word: yes or no."
        )

    elif dataset == "gsm8k":
        system = (
            "You are an expert math answer evaluator.\n"
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "The proposed answer is correct if it is equivalent to any of the valid answers.\n"
            "Focus on the value being expressed — formatting differences such as 5.0 vs 5, $100 vs 100, or 1,000 vs 1000, 4 vs four are the same.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it equals ANY one of them.\n"
            "Respond with exactly one word: yes or no."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, is the proposed answer equivalent to at least one of the valid answers? "
            "Respond with exactly one word: yes or no."
        )

    else:
        system = (
            "IMPORTANT: Evaluate the proposed answer IN THE CONTEXT OF WHAT THE QUESTION ASKS.\n"
            "Determine whether the proposed answer conveys the same meaning as any of the valid answers.\n"
            "Focus on the meaning being conveyed, not surface form — synonyms, paraphrases, and abbreviations are acceptable.\n"
            "A partial or abbreviated answer is acceptable if it unambiguously identifies the correct answer given the context.\n"
            "There may be multiple valid answers listed — the proposed answer is correct if it matches ANY one of them.\n"
            "Respond with exactly one word: yes or no."
        )
        user = (
            "Question: {question}\n"
            "Valid answer(s) (match ANY one):\n{ground_truth}\n"
            "Proposed answer: {prediction}\n\n"
            "In the context of the question asked, does the proposed answer convey the same meaning as at least one of the valid answers? yes or no."
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
                print(f"WARNING: Could not parse progress file {progress_file}. Starting fresh.")
    return {}


def save_progress(progress_file: str, verdicts: dict) -> None:
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(verdicts, f)


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_prompts(rows: list[dict], tokenizer, dataset: str) -> list[str]:
    prompts = []
    system_prompt, user_template = get_judge_prompt(dataset)

    for row in rows:
        gt_list = parse_ground_truth(row["ground_truth"], dataset)
        if len(gt_list) == 1:
            gt_formatted = f"  1. {gt_list[0]}"
        else:
            gt_formatted = "\n".join(f"  {i + 1}. {ans}" for i, ans in enumerate(gt_list))

        user_text = user_template.format(
            question=str(row["question"]).strip(),
            ground_truth=gt_formatted,
            prediction=normalize_answer(row["low_t_generation"], dataset),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    return prompts


def parse_verdict(text: str) -> bool:
    cleaned = text.strip().lower()
    if re.search(r'\byes\b', cleaned):
        return True
    if re.search(r'\bno\b', cleaned):
        return False
    print(f"  WARNING: ambiguous verdict {text!r} — treating as 'no'")
    return False


def run_judge(
    df: pd.DataFrame,
    tokenizer,
    model,
    batch_size: int,
    max_new_tokens: int,
    progress_file: str,
    dataset: str,
) -> dict:
    """
    Run LLM judge over all rows.
    Returns dict mapping str(index) → bool verdict.
    Supports resume via progress_file.
    """
    verdicts = load_progress(progress_file)
    already_done = len(verdicts)
    if already_done:
        print(f"Resuming: {already_done} rows already judged.")

    pending_indices = [idx for idx in df.index if str(idx) not in verdicts]
    pending_rows    = df.loc[pending_indices].to_dict("records")

    print(f"Rows to judge: {len(pending_indices)}")

    first_device = next(model.parameters()).device

    for batch_start in range(0, len(pending_indices), batch_size):
        batch_idx  = pending_indices[batch_start: batch_start + batch_size]
        batch_rows = pending_rows[batch_start: batch_start + batch_size]

        prompts = build_chat_prompts(batch_rows, tokenizer, dataset)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids      = inputs["input_ids"].to(first_device)
        attention_mask = inputs["attention_mask"].to(first_device)
        input_len      = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for idx, out in zip(batch_idx, output_ids):
            new_tokens = out[input_len:]
            raw_text   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            verdicts[str(idx)] = parse_verdict(raw_text)

        save_progress(progress_file, verdicts)
        done_so_far = already_done + batch_start + len(batch_idx)
        total = already_done + len(pending_indices)
        print(f"  Progress: {done_so_far}/{total}  "
              f"(yes so far: {sum(verdicts.values())})")

    return verdicts


def process_single_csv(
    input_path: Path,
    output_path: Path,
    tokenizer,
    model,
    batch_size: int,
    max_new_tokens: int,
    dataset: str,
) -> bool:
    """
    Process one CSV: judge all rows, add LLM_verdict column, save.
    Returns True on success.
    """
    progress_file = str(output_path.with_suffix(".progress.json"))

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    if len(df) == 0:
        print("Empty CSV. Skipping.")
        return False

    verdicts = run_judge(
        df=df,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        progress_file=progress_file,
        dataset=dataset,
    )

    # Add LLM_verdict column
    df["LLM_verdict"] = df.index.map(lambda idx: verdicts.get(str(idx), None))

    confirmed = sum(verdicts.values())
    rejected  = len(verdicts) - confirmed

    print("\n" + "=" * 60)
    print("VERDICT SUMMARY")
    print("=" * 60)
    print(f"  Total rows judged    : {len(verdicts)}")
    print(f"  LLM says correct     : {confirmed}")
    print(f"  LLM says incorrect   : {rejected}")
    print(f"  Accuracy (LLM judge) : {confirmed / len(verdicts) * 100:.1f}%")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Removed progress file: {progress_file}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args):
    model_path = args.model

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from    : {model_path}")
    print("  Using device_map='auto' — distributes layers across available GPUs")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add LLM_verdict column to semantic uncertainty CSVs using an LLM judge (HF backend)."
    )
    p.add_argument("--model",
                   default="/data/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
                   help="HuggingFace model path or name used as the judge")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Inference batch size (default: 8)")
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="Max tokens for judge response (default: 16 — just yes/no)")
    p.add_argument("--cuda_device", type=str, default=None,
                   help="CUDA device index or indices (e.g. '0' or '0,1'). "
                        "Sets CUDA_VISIBLE_DEVICES before loading the model.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip CSVs that already have a _llm_verdict output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not FILE_LIST:
        print("FILE_LIST is empty. Add CSV paths to FILE_LIST at the top of the script.")
        return

    # Validate all paths before loading model
    for f in FILE_LIST:
        if not Path(f).exists():
            raise FileNotFoundError(f"File not found: {f}")

    print(f"{'=' * 60}")
    print(f"LLM VERDICT — {len(FILE_LIST)} file(s) to process")
    print(f"{'=' * 60}")
    for i, f in enumerate(FILE_LIST, 1):
        print(f"  {i:2d}. {f}")
    print()

    tokenizer, model = load_model(args)

    total_start = time.time()
    processed = 0
    skipped = 0
    failed = 0

    for i, csv_path_str in enumerate(FILE_LIST, 1):
        csv_path = Path(csv_path_str)
        output_path = csv_path.parent / (csv_path.stem + "_llm_verdict.csv")

        if args.skip_existing and output_path.exists():
            print(f"\n{'─' * 60}")
            print(f"[{i}/{len(FILE_LIST)}] SKIPPING (output exists): {csv_path.name}")
            print(f"{'─' * 60}")
            skipped += 1
            continue

        dataset = detect_dataset(str(csv_path))

        print(f"\n{'━' * 60}")
        print(f"[{i}/{len(FILE_LIST)}] Processing: {csv_path.name}")
        print(f"  Dataset type: {dataset}  →  Output: {output_path.name}")
        print(f"{'━' * 60}")

        file_start = time.time()
        try:
            success = process_single_csv(
                input_path=csv_path,
                output_path=output_path,
                tokenizer=tokenizer,
                model=model,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                dataset=dataset,
            )
            elapsed = time.time() - file_start
            print(f"  Completed in {elapsed:.1f}s")
            processed += 1
        except Exception as e:
            elapsed = time.time() - file_start
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            failed += 1

    total_elapsed = time.time() - total_start
    print(f"\n{'━' * 60}")
    print(f"ALL DONE")
    print(f"{'━' * 60}")
    print(f"  Total files  : {len(FILE_LIST)}")
    print(f"  Processed    : {processed}")
    print(f"  Skipped      : {skipped}")
    print(f"  Failed       : {failed}")
    print(f"  Total time   : {total_elapsed / 60:.1f} minutes")
    print(f"{'━' * 60}")


if __name__ == "__main__":
    main()
