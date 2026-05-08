#!/usr/bin/env python3
"""
LLM Judge Re-labeler for Semantic Uncertainty CSVs (HuggingFace Transformers backend).

Uses AutoModelForCausalLM with device_map="auto" instead of vLLM — works for large
models like Llama-3.3-70B that exceed vLLM's KV-cache pre-allocation limit.

Processes all rows with labels in ["Correct", "AH", "UH"] using an LLM judge
to verify semantic correctness, then re-labels based on verdict and entropy:

For "Correct" rows:
    - LLM confirms correct                → "Correct" (no change)
    - LLM says incorrect + entropy < threshold  → "AH"
    - LLM says incorrect + entropy >= threshold → "UH"

For "AH"/"UH" rows:
    - LLM confirms correct               → "Correct" (promote)
    - LLM says incorrect                 → keep current label (no change)

The threshold is parsed from the filename (e.g. entropy_0.5) unless overridden
with --threshold.

Usage (single file):
    python llm_judge_relabel_hf.py \\
        --input_csv  /path/to/detail_svamp__llama_entropy_0.5.csv \\
        --output_csv /path/to/detail_svamp__llama_entropy_0.5_relabeled.csv \\
        --model      /data/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \\
        --cuda_device 0,1

Batch mode:
    nohup python3 llm_judge_relabel_hf.py \\
        --batch_dir /home/kalashkala/Datasets/Semantic-Uncertainty/output_test \\
        --model /data/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \\
        --cuda_device 0,1 --skip_existing > relabel_batch_llama70b_hf.log 2>&1 &
"""

import os
import argparse
import ast
import json
import re
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def extract_threshold(filename: str) -> float | None:
    """Parse entropy threshold from filename like detail_svamp__llama_entropy_0.5.csv."""
    m = re.search(r'entropy_(\d+(?:\.\d+)?)', filename)
    return float(m.group(1)) if m else None


def normalize_text(x: str) -> str:
    """Normalize factual answers (sciq, triviaqa): lowercase, strip punctuation and extra spaces."""
    x = str(x).lower().strip()
    x = re.sub(r'[^a-z0-9\s]', '', x)  # remove punctuation
    x = re.sub(r'\s+', ' ', x)         # normalize spaces
    return x


def normalize_math(x: str) -> str:
    """Normalize numerical answers (svamp): strip currency/units/commas, collapse trailing zeros."""
    x = str(x).strip()
    x = re.sub(r'[$€£¥]', '', x)               # strip currency symbols
    x = re.sub(r'(?<=\d),(?=\d{3})', '', x)    # remove thousands commas: 1,000 → 1000
    x = re.sub(r'[a-zA-Z\s]+$', '', x).strip() # strip trailing units/words: "5 apples" → "5"
    try:
        val = float(x)
        # collapse 5.0 → 5, keep 3.14 → 3.14
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return x.lower()


def normalize_answer(x: str, dataset: str) -> str:
    """Route to the appropriate normalizer based on dataset type."""
    if dataset == "svamp":
        return normalize_math(x)
    return normalize_text(x)


def parse_ground_truth(gt_str: str, dataset: str) -> list[str]:
    """Normalize ground truth; always returns a list of valid answers."""
    try:
        parsed = ast.literal_eval(str(gt_str))
        if isinstance(parsed, list) and parsed:
            return [normalize_answer(x, dataset) for x in parsed if str(x).strip()]
        return [normalize_answer(parsed, dataset)]
    except (ValueError, SyntaxError):
        return [normalize_answer(gt_str, dataset)]


def load_progress(progress_file: str) -> dict:
    """Load intermediate results from a progress JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"WARNING: Could not parse progress file {progress_file}. Starting fresh.")
    return {}


def save_progress(progress_file: str, verdicts: dict) -> None:
    """Save intermediate results keyed by DataFrame index."""
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(verdicts, f)


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_prompts(rows: list[dict], tokenizer, dataset: str) -> list[str]:
    """Build chat-formatted prompt strings for each row."""
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
            prediction=normalize_answer(row["prediction"], dataset),
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
    """Return True if the model replied 'yes', False otherwise."""
    cleaned = text.strip().lower()
    # Accept "yes" anywhere in a short reply (model sometimes adds punctuation)
    if re.search(r'\byes\b', cleaned):
        return True
    if re.search(r'\bno\b', cleaned):
        return False
    # Fallback: treat ambiguous responses as not-confirmed
    print(f"  WARNING: ambiguous verdict {text!r} — treating as 'no'")
    return False


def run_judge(
    df_correct: pd.DataFrame,
    tokenizer,
    model,
    batch_size: int,
    max_new_tokens: int,
    progress_file: str,
    dataset: str,
) -> dict:
    """
    Run LLM judge over df_correct rows using HuggingFace model.generate().
    Returns dict mapping str(index) → bool verdict.
    Supports resume via progress_file.
    """
    verdicts = load_progress(progress_file)
    already_done = len(verdicts)
    if already_done:
        print(f"Resuming: {already_done} rows already judged.")

    # Collect rows that still need judging
    pending_indices = [idx for idx in df_correct.index if str(idx) not in verdicts]
    pending_rows    = df_correct.loc[pending_indices].to_dict("records")

    print(f"Rows to judge: {len(pending_indices)}")

    # Determine input device (first GPU in device_map)
    first_device = next(model.parameters()).device

    for batch_start in range(0, len(pending_indices), batch_size):
        batch_idx  = pending_indices[batch_start: batch_start + batch_size]
        batch_rows = pending_rows[batch_start: batch_start + batch_size]

        prompts = build_chat_prompts(batch_rows, tokenizer, dataset)

        # Tokenize with left-padding (required for decoder-only batch generation)
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

        # Persist after every batch so a crash doesn't lose everything
        save_progress(progress_file, verdicts)
        done_so_far = already_done + batch_start + len(batch_idx)
        total = already_done + len(pending_indices)
        print(f"  Progress: {done_so_far}/{total}  "
              f"(yes so far: {sum(verdicts.values())})")

    return verdicts


def assign_labels(
    df: pd.DataFrame,
    verdicts: dict,
    threshold: float,
) -> pd.DataFrame:
    """
    Apply new labels based on LLM verdicts, current label, and entropy.

    For "Correct" rows:
      - LLM says correct   → keep as "Correct"
      - LLM says incorrect → entropy rule:
          * entropy < threshold → "AH"
          * entropy >= threshold → "UH"

    For "AH"/"UH" rows:
      - LLM says correct   → promote to "Correct"
      - LLM says incorrect → keep current label (no change)

    Adds a 'llm_verdict' column (True/False) for transparency.
    """
    df = df.copy()
    if "llm_verdict" not in df.columns:
        df["llm_verdict"] = None

    for idx_str, is_correct in verdicts.items():
        idx = int(idx_str)
        current_label = df.at[idx, "label"]
        df.at[idx, "llm_verdict"] = is_correct

        if is_correct:
            # LLM confirms correct → promote to "Correct"
            df.at[idx, "label"] = "Correct"
        else:
            # LLM says incorrect
            if current_label == "Correct":
                # Demote "Correct" rows based on entropy
                entropy = df.at[idx, "semantic_entropy"]
                df.at[idx, "label"] = "AH" if entropy < threshold else "UH"
            # else: keep "AH"/"UH" as-is (no change)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-CSV processing (used by both single and batch modes)
# ─────────────────────────────────────────────────────────────────────────────

def process_single_csv(
    input_path: Path,
    output_path: Path,
    threshold: float,
    tokenizer,
    model,
    batch_size: int,
    max_new_tokens: int,
    dataset: str,
) -> bool:
    """
    Process one CSV file: judge all Correct/AH/UH rows, relabel, save.
    Returns True on success, False if skipped (no target rows).
    """
    progress_file = str(output_path.with_suffix(".progress.json"))

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Label distribution (before):\n{df['label'].value_counts().to_string()}\n")

    target_labels = ["Correct", "AH", "UH"]
    target_mask = df["label"].isin(target_labels)
    df_target = df[target_mask]
    print(f"Rows with labels in {target_labels}: {len(df_target)}")
    print(f"  Correct: {(df['label']=='Correct').sum()}")
    print(f"  AH     : {(df['label']=='AH').sum()}")
    print(f"  UH     : {(df['label']=='UH').sum()}\n")

    if len(df_target) == 0:
        print("Nothing to re-label. Saving unchanged output.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        return False

    # ── Run judge ─────────────────────────────────────────────────────────────
    verdicts = run_judge(
        df_correct=df_target,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        progress_file=progress_file,
        dataset=dataset,
    )

    # ── Apply new labels ──────────────────────────────────────────────────────
    df_out = assign_labels(df, verdicts, threshold)

    # ── Summary ───────────────────────────────────────────────────────────────
    confirmed = sum(verdicts.values())
    rejected  = len(verdicts) - confirmed

    correct_before = (df["label"] == "Correct").sum()
    correct_after = (df_out["label"] == "Correct").sum()
    promoted_count = correct_after - correct_before

    print("\n" + "=" * 60)
    print("RE-LABELING SUMMARY")
    print("=" * 60)
    print(f"  Total rows processed        : {len(verdicts)}")
    print(f"  LLM-confirmed correct       : {confirmed}")
    print(f"    → Promoted to 'Correct'   : {max(promoted_count, 0)}")
    print(f"  LLM-rejected as incorrect   : {rejected}")
    print(f"\nLabel changes:")
    print(f"  'Correct' before : {correct_before} → after : {correct_after}")
    print(f"  'AH'      before : {(df['label']=='AH').sum()} → after : {(df_out['label']=='AH').sum()}")
    print(f"  'UH'      before : {(df['label']=='UH').sum()} → after : {(df_out['label']=='UH').sum()}")
    print(f"\nFinal label distribution:\n{df_out['label'].value_counts().to_string()}")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Clean up progress file on success
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Removed progress file: {progress_file}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-label 'Correct' rows in a semantic uncertainty CSV using an LLM judge (HF backend)."
    )
    # Single-file mode
    p.add_argument("--input_csv", default=None,
                   help="Path to input CSV (single-file mode)")
    p.add_argument("--output_csv", default=None,
                   help="Path to output CSV (default: <stem>_relabeled.csv next to input)")
    # Batch mode
    p.add_argument("--batch_dir", default=None,
                   help="Directory containing detail_*_entropy_*.csv files. "
                        "Loads model once and processes all CSVs sequentially.")
    p.add_argument("--skip_existing", action="store_true",
                   help="In batch mode, skip CSVs that already have a _relabeled output.")
    # Model & inference
    p.add_argument("--model",
                   default="/data/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
                   help="HuggingFace model path or name used as the judge")
    p.add_argument("--threshold", type=float, default=None,
                   help="Entropy threshold (parsed from filename if omitted)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Inference batch size (default: 8)")
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="Max tokens for judge response (default: 16 — just yes/no)")
    p.add_argument("--cuda_device", type=str, default=None,
                   help="CUDA device index or indices (e.g. '0' or '0,1'). "
                        "Sets CUDA_VISIBLE_DEVICES before loading the model.")
    return p


def load_model(args):
    """Load tokenizer and model with device_map='auto' for multi-GPU support."""
    model_path = args.model

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Left-padding is required for batched generation with decoder-only models
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


def discover_csvs(batch_dir: Path) -> list[Path]:
    """Find all detail_*_entropy_*.csv files that are NOT already relabeled."""
    pattern = "detail_*_entropy_*.csv"
    all_csvs = sorted(batch_dir.glob(pattern))
    # Exclude files that are already relabeled outputs
    return [p for p in all_csvs if "_relabeled" not in p.stem]


def main() -> None:
    import time

    args = build_parser().parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if args.batch_dir is None and args.input_csv is None:
        raise ValueError("Must specify either --input_csv (single file) or --batch_dir (batch mode).")
    if args.batch_dir is not None and args.input_csv is not None:
        raise ValueError("Cannot specify both --input_csv and --batch_dir.")

    # ── BATCH MODE ────────────────────────────────────────────────────────────
    if args.batch_dir is not None:
        batch_path = Path(args.batch_dir)
        if not batch_path.is_dir():
            raise FileNotFoundError(f"Batch directory not found: {batch_path}")

        csv_files = discover_csvs(batch_path)
        if not csv_files:
            print(f"No matching CSVs found in {batch_path}")
            return

        print(f"{'=' * 60}")
        print(f"BATCH MODE — {len(csv_files)} CSVs found in {batch_path}")
        print(f"{'=' * 60}")
        for i, f in enumerate(csv_files, 1):
            print(f"  {i:2d}. {f.name}")
        print()

        # Load model ONCE
        tokenizer, model = load_model(args)

        total_start = time.time()
        processed = 0
        skipped = 0
        failed = 0

        for i, csv_path in enumerate(csv_files, 1):
            output_path = csv_path.parent / (csv_path.stem + "_relabeled.csv")

            # Skip if output already exists
            if args.skip_existing and output_path.exists():
                print(f"\n{'─' * 60}")
                print(f"[{i}/{len(csv_files)}] SKIPPING (output exists): {csv_path.name}")
                print(f"{'─' * 60}")
                skipped += 1
                continue

            # Resolve threshold from filename
            threshold = args.threshold
            if threshold is None:
                threshold = extract_threshold(csv_path.name)
                if threshold is None:
                    print(f"\n[{i}/{len(csv_files)}] ERROR: Cannot parse threshold from '{csv_path.name}' — skipping.")
                    failed += 1
                    continue

            print(f"\n{'━' * 60}")
            print(f"[{i}/{len(csv_files)}] Processing: {csv_path.name}")
            print(f"  Threshold: {threshold}  →  Output: {output_path.name}")
            print(f"{'━' * 60}")

            file_start = time.time()
            try:
                dataset = detect_dataset(str(csv_path))
                print(f"Detected dataset: {dataset}")

                success = process_single_csv(
                    input_path=csv_path,
                    output_path=output_path,
                    threshold=threshold,
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
        print(f"BATCH COMPLETE")
        print(f"{'━' * 60}")
        print(f"  Total files  : {len(csv_files)}")
        print(f"  Processed    : {processed}")
        print(f"  Skipped      : {skipped}")
        print(f"  Failed       : {failed}")
        print(f"  Total time   : {total_elapsed / 60:.1f} minutes")
        print(f"{'━' * 60}")
        return

    # ── SINGLE FILE MODE ──────────────────────────────────────────────────────
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    threshold = args.threshold
    if threshold is None:
        threshold = extract_threshold(input_path.name)
        if threshold is None:
            raise ValueError(
                f"Could not parse entropy threshold from '{input_path.name}'. "
                "Use --threshold to set it explicitly."
            )
    print(f"Entropy threshold : {threshold}")

    output_path = Path(args.output_csv) if args.output_csv else (
        input_path.parent / (input_path.stem + "_relabeled.csv")
    )

    # Load model
    tokenizer, model = load_model(args)

    dataset = detect_dataset(str(input_path))
    print(f"Detected dataset: {dataset}")

    process_single_csv(
        input_path=input_path,
        output_path=output_path,
        threshold=threshold,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
