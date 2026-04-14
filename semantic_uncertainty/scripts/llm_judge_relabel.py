#!/usr/bin/env python3
"""
LLM Judge Re-labeler for Semantic Uncertainty CSVs.

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

Usage:
    python llm_judge_relabel.py \\
        --input_csv  /path/to/detail_svamp__llama_entropy_0.5.csv \\
        --output_csv /path/to/detail_svamp__llama_entropy_0.5_relabeled.csv \\
        --model      /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ

Sample commands for all dataset-model-threshold combos:
(base dir: /home/kalashkala/Datasets/Semantic-Uncertainty/output)

    # ── SVAMP ─────────────────────────────────────────────────────────────────
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__llama_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__llama_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_llama_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__llama_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__llama_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_llama_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__gemma_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__gemma_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_gemma_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__gemma_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__gemma_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_gemma_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__mistral_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__mistral_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_mistral_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__mistral_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__mistral_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_mistral_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__qwen_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__qwen_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_qwen_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__qwen_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_svamp__qwen_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_svamp_qwen_0.3.log 2>&1 &

    # ── SciQ ──────────────────────────────────────────────────────────────────
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__llama_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__llama_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_llama_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__llama_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__llama_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_llama_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__gemma_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__gemma_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_gemma_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__gemma_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__gemma_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_gemma_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__mistral_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__mistral_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_mistral_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__mistral_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__mistral_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_mistral_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__qwen_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__qwen_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_qwen_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__qwen_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_sciq__qwen_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_sciq_qwen_0.3.log 2>&1 &

    # ── TriviaQA ──────────────────────────────────────────────────────────────
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__llama_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__llama_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_llama_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__llama_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__llama_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_llama_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__gemma_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__gemma_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_gemma_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__gemma_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__gemma_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_gemma_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__mistral_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__mistral_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_mistral_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__mistral_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__mistral_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_mistral_0.3.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__qwen_entropy_0.5.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__qwen_entropy_0.5_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_qwen_0.5.log 2>&1 &
    nohup python3 llm_judge_relabel.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__qwen_entropy_0.3.csv --output_csv /home/kalashkala/Datasets/Semantic-Uncertainty/output/detail_triviaqa__qwen_entropy_0.3_relabeled.csv --model /home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ --cuda_device 0 > relabel_triviaqa_qwen_0.3.log 2>&1 &
"""

import os
# Fix MKL threading conflict with libgomp (vLLM subprocess inspection crashes without this)
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Judge prompt
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict answer correctness evaluator. "
    "Given a question, an expected answer, and a proposed answer, "
    "decide if the proposed answer is semantically equivalent to the expected answer. "
    "Respond with exactly one word: yes or no."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Expected answer: {ground_truth}\n"
    "Proposed answer: {prediction}\n\n"
    "Is the proposed answer semantically equivalent to the expected answer? "
    "Respond with exactly one word: yes or no."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_threshold(filename: str) -> float | None:
    """Parse entropy threshold from filename like detail_svamp__llama_entropy_0.5.csv."""
    m = re.search(r'entropy_(\d+(?:\.\d+)?)', filename)
    return float(m.group(1)) if m else None


def parse_ground_truth(gt_str: str) -> str:
    """Return first element if gt_str is a list literal, else return as-is."""
    try:
        parsed = ast.literal_eval(str(gt_str))
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
        return str(parsed).strip()
    except (ValueError, SyntaxError):
        return str(gt_str).strip()


def is_awq_model(model_path: str) -> bool:
    """Detect AWQ quantization from model path or config."""
    if "awq" in model_path.lower():
        return True
    config_path = Path(model_path) / "quantize_config.json"
    return config_path.exists()


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

def build_chat_prompts(rows: list[dict], tokenizer) -> list[str]:
    """Build chat-formatted prompts for each row."""
    prompts = []
    for row in rows:
        gt = parse_ground_truth(row["ground_truth"])
        user_text = JUDGE_USER_TEMPLATE.format(
            question=str(row["question"]).strip(),
            ground_truth=gt,
            prediction=str(row["prediction"]).strip(),
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
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
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int,
    progress_file: str,
) -> dict:
    """
    Run LLM judge over df_correct rows.
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

    for batch_start in range(0, len(pending_indices), batch_size):
        batch_idx  = pending_indices[batch_start: batch_start + batch_size]
        batch_rows = pending_rows[batch_start: batch_start + batch_size]

        prompts = build_chat_prompts(batch_rows, tokenizer)
        outputs = llm.generate(prompts, sampling_params)

        for idx, output in zip(batch_idx, outputs):
            raw_text = output.outputs[0].text
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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-label 'Correct' rows in a semantic uncertainty CSV using an LLM judge."
    )
    p.add_argument("--input_csv",  required=True,
                   help="Path to input CSV, e.g. detail_svamp__llama_entropy_0.5.csv")
    p.add_argument("--output_csv", default=None,
                   help="Path to output CSV (default: <stem>_relabeled.csv next to input)")
    p.add_argument("--model",
                   default="/home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ",
                   help="HuggingFace model path or name used as the judge")
    p.add_argument("--threshold", type=float, default=None,
                   help="Entropy threshold (parsed from filename if omitted)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Inference batch size (default: 64)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                   help="vLLM GPU memory utilization (default: 0.90)")
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="Max tokens for judge response (default: 16 — just yes/no)")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Number of GPUs for tensor parallelism (default: 1)")
    p.add_argument("--cuda_device", type=str, default=None,
                   help="CUDA device index or indices to use (e.g., '0' or '0,1')")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ── Resolve threshold ─────────────────────────────────────────────────────
    threshold = args.threshold
    if threshold is None:
        threshold = extract_threshold(input_path.name)
        if threshold is None:
            raise ValueError(
                f"Could not parse entropy threshold from '{input_path.name}'. "
                "Use --threshold to set it explicitly."
            )
    print(f"Entropy threshold : {threshold}")

    # ── Resolve output path ───────────────────────────────────────────────────
    output_path = Path(args.output_csv) if args.output_csv else (
        input_path.parent / (input_path.stem + "_relabeled.csv")
    )
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
        return

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = args.model
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading LLM from      : {model_path}")
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
    )
    if is_awq_model(model_path):
        llm_kwargs["quantization"] = "awq_marlin"
        llm_kwargs["dtype"] = "float16"
    else:
        llm_kwargs["dtype"] = "bfloat16"

    llm = LLM(**llm_kwargs)
    print("Model loaded.\n")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["\n"],
    )

    # ── Run judge ─────────────────────────────────────────────────────────────
    verdicts = run_judge(
        df_correct=df_target,
        tokenizer=tokenizer,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        progress_file=progress_file,
    )

    # ── Apply new labels ──────────────────────────────────────────────────────
    df_out = assign_labels(df, verdicts, threshold)

    # ── Summary ───────────────────────────────────────────────────────────────
    confirmed = sum(verdicts.values())
    rejected  = len(verdicts) - confirmed

    # Count demotions and promotions
    correct_before = (df["label"] == "Correct").sum()
    correct_after = (df_out["label"] == "Correct").sum()
    promoted_count = correct_after - correct_before  # Can be negative if more demoted than promoted

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


if __name__ == "__main__":
    main()
