#!/usr/bin/env python3
"""Extract hidden states for a subset of rows defined by a sampled CSV.

Variant of extract_open_text_hidden_states.py designed for the recovery-gaps
experiment, where the input CSV is already the balanced subset (e.g.
sampled_1400_uncertainty_run_llama_sciq_combined_llm_verdict.csv) rather than
the full 10K dataset.

For each row in the CSV the script:
  1. Builds the same few-shot prompt as generate_answers_combined.py (loaded
     from experiment_details.pkl so prompts are identical to the original run).
  2. Runs greedy decoding to generate a fresh open-text answer.
  3. Feeds prompt + answer back through the model and extracts hidden states.

Output files (in --output_dir):
  open_hidden_states_arrays_{suffix}.npz   : compressed numpy arrays
  open_hidden_states_{suffix}.json         : per-example metadata + row_id
  open_hidden_states_summary_{suffix}.json : aggregate statistics

NPZ keys (--layer all, default):
  open_hidden_state            : [N, hidden_dim]           last-layer vectors
  open_hidden_state_all_layers : [N, n_layers, hidden_dim] all-layer vectors
  layer_indices                : [n_layers]

Example command
---------------
nohup conda run --no-capture-output -n semantic_uncertainty \\
    env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python extract_open_text_hidden_states_subset.py \\
        --input_csv      /home/kalashkala/recovery-gaps-experiment/data/sampled_1400_uncertainty_run_llama_sciq_combined_llm_verdict.csv \\
        --experiment_pkl /data/kalashkala/semantic_uncertainty_data/uncertainty/sciq__meta-llama__Llama-3.1-8B-Instruct__seed10__pid3815758__20260428_195641/experiment_details.pkl \\
        --output_dir     /home/kalashkala/recovery-gaps-experiment/data/hidden_states_subset \\
        --model_name     meta-llama/Llama-3.1-8B-Instruct \\
        --layer          all \\
        --token_position last \\
        --device         cuda --gpu_id 0 \\
    > extract_hidden_states_subset.log 2>&1 &
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pickle
import random
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

LayerSpec = Union[str, int]

STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']


# ---------------------------------------------------------------------------
# Stopping criteria (matches HuggingfaceModel.StoppingCriteriaSub)
# ---------------------------------------------------------------------------

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, initial_length):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer
        self.initial_length = initial_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generation = self.tokenizer.decode(
            input_ids[0][self.initial_length:], skip_special_tokens=False)
        return any(stop in generation for stop in self.stops)


# ---------------------------------------------------------------------------
# Seeding / device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str, gpu_id: Optional[int] = None) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if gpu_id is not None:
        return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Experiment details
# ---------------------------------------------------------------------------

def load_experiment_details(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Prompt construction (matches generate_answers_combined.py)
# ---------------------------------------------------------------------------

def build_question_input(question: str, context: Optional[str],
                         use_context: bool, brief: str,
                         brief_always: bool) -> str:
    parts = ''
    if brief_always:
        parts += brief
    if use_context and context:
        parts += f"Context: {context}\n"
    parts += f"Question: {question}\n"
    parts += 'Answer:'
    return parts


def split_context_question(combined: str) -> Tuple[str, str]:
    parts = combined.rsplit('\n', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return '', combined.strip()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def parse_ground_truth(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return text


def strip_stop_sequences(answer: str) -> str:
    for stop in STOP_SEQUENCES:
        if answer.endswith(stop):
            return answer[: len(answer) - len(stop)].strip()
    return answer.strip()


def clean_answer(text: str) -> str:
    x = str(text or "").strip()
    x = re.sub(
        r"^\s*(?:<\|im_start\|>\s*)?(?:system|user|assistant)\s*(?:<\|im_end\|>)?\s*",
        "", x, flags=re.IGNORECASE,
    )
    lines = [l.strip() for l in x.splitlines() if l.strip()]
    if lines and lines[0].lower() in {"system", "user", "assistant"}:
        lines = lines[1:]
    x = " ".join(lines).strip()
    x = re.sub(r"^\s*(the\s+answer\s+is|answer\s*:?)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^\s*[:\-\"'`]+\s*", "", x)
    x = x.strip(" \t\n\r\"'`[](){}")
    x = re.sub(r"[.\s]+$", "", x)
    return re.sub(r"\s+", " ", x)


def normalize_answer(text: str) -> str:
    return clean_answer(text).lower()


def answers_match(predicted: str, true_answer: str) -> bool:
    p, t = normalize_answer(predicted), normalize_answer(true_answer)
    if not p or not t:
        return False
    return p == t or p in t or t in p


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_records(path: Path) -> List[Dict[str, Any]]:
    """Load all rows from the sampled CSV. The CSV is already the subset."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            question = row.get("question", "").strip()
            if not question:
                continue
            records.append({
                "row_id":      row.get("id", ""),
                "question":    question,
                "true_answer": parse_ground_truth(row.get("ground_truth", "")),
                "context":     row.get("context", ""),
            })
    return records


# ---------------------------------------------------------------------------
# Model / tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str, device: torch.device
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    inputs.pop('token_type_ids', None)
    n_input = inputs['input_ids'].shape[1]
    pad_id = getattr(tokenizer, 'pad_token_id', None) or tokenizer.eos_token_id

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        stops=STOP_SEQUENCES + [tokenizer.eos_token],
        tokenizer=tokenizer,
        initial_length=n_input,
    )])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            stopping_criteria=stopping_criteria,
        )

    answer = tokenizer.decode(outputs[0][n_input:], skip_special_tokens=True)
    return strip_stop_sequences(answer)


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def parse_layer_spec(layer_arg: str) -> LayerSpec:
    if layer_arg in {"last", "all"}:
        return layer_arg
    try:
        return int(layer_arg)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid layer value: {layer_arg}") from exc


def select_layer_indices(
    hidden_states: Tuple[torch.Tensor, ...], layer: LayerSpec
) -> List[int]:
    n = len(hidden_states)
    if layer == "all":
        return list(range(n))
    if layer == "last":
        return [n - 1]
    if isinstance(layer, int):
        idx = layer if layer >= 0 else n + layer
        if not (0 <= idx < n):
            raise IndexError(f"Layer {idx} out of range for {n} layers")
        return [idx]
    raise ValueError(f"Unsupported layer spec: {layer}")


def extract_hidden_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_prompt: str,
    response: str,
    layer: LayerSpec,
    device: torch.device,
    token_position: str = "last",
) -> Tuple[np.ndarray, List[int]]:
    input_ids = tokenizer(full_prompt + response, return_tensors="pt").input_ids.to(device)
    prompt_len = tokenizer(full_prompt, return_tensors="pt").input_ids.shape[1]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    layer_indices = select_layer_indices(hidden_states, layer)

    if token_position == "last":
        target_idx = -1
    elif token_position == "last_prompt":
        target_idx = min(prompt_len - 1, seq_len - 1)
    elif token_position == "first_answer":
        target_idx = prompt_len if prompt_len < seq_len else -1
    else:
        raise ValueError(f"Unknown token_position: {token_position}")

    selected = [
        hidden_states[i][0, target_idx, :].detach().cpu().float().numpy()
        for i in layer_indices
    ]

    if len(selected) == 1:
        return selected[0], layer_indices
    return np.stack(selected, axis=0), layer_indices


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract open-text hidden states for a sampled subset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_csv",      type=str, required=True,
                        help="Sampled subset CSV (must have id, question, ground_truth columns)")
    parser.add_argument("--experiment_pkl", type=str, required=True,
                        help="experiment_details.pkl from the original generate_answers_combined.py run")
    parser.add_argument("--output_dir",     type=str, default="hidden_states_subset",
                        help="Directory to write output files")
    parser.add_argument("--model_name",     type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer",          type=parse_layer_spec, default="all",
                        help="Layer index, 'last', or 'all'")
    parser.add_argument("--token_position", type=str, default="last",
                        choices=["last", "last_prompt", "first_answer"])
    parser.add_argument("--max_new_tokens", type=int, default=15,
                        help="Overridden by experiment_pkl if available")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--gpu_id",         type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, label in [(csv_path, "CSV"), (Path(args.experiment_pkl), "experiment PKL")]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    # Load experiment details to match original prompt structure exactly
    print(f"Loading experiment details from: {args.experiment_pkl}")
    exp = load_experiment_details(args.experiment_pkl)
    fewshot_prefix = exp['prompt']
    brief         = exp.get('BRIEF', '')
    exp_args      = exp['args']
    use_context   = getattr(exp_args, 'use_context', False)
    brief_always  = getattr(exp_args, 'brief_always', False) and getattr(exp_args, 'enable_brief', True)
    orig_max_new  = getattr(exp_args, 'model_max_new_tokens', None)
    if orig_max_new is not None:
        args.max_new_tokens = orig_max_new
    print(f"  use_context={use_context}, brief_always={brief_always}, max_new_tokens={args.max_new_tokens}")

    print(f"\nLoading records from: {csv_path}")
    records = load_csv_records(csv_path)
    if not records:
        raise ValueError(f"No usable records found in {csv_path}")
    print(f"Loaded {len(records)} records (subset — no further sampling)")

    device = resolve_device(args.device, args.gpu_id)
    print(f"\nLoading model: {args.model_name} on {device}")
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)
    print("Model loaded.\n")

    open_vectors: List[np.ndarray] = []
    metadata: List[Dict[str, Any]]  = []
    layer_indices: Optional[List[int]] = None

    print(f"Processing {len(records)} examples | layer={args.layer} | token_position={args.token_position}")
    print("=" * 80)

    for idx, row in enumerate(records):
        question    = row["question"]
        true_answer = row["true_answer"]
        context     = row.get("context") or None

        if use_context and not context:
            context, question = split_context_question(question)

        question_input = build_question_input(question, context, use_context, brief, brief_always)
        full_prompt    = fewshot_prefix + question_input

        print(f"[{idx+1}/{len(records)}] id={row['row_id']} generating...", end=" ", flush=True)

        try:
            predicted = generate_answer(model, tokenizer, full_prompt, device, args.max_new_tokens)
            vec, curr_layer_indices = extract_hidden_state(
                model, tokenizer, full_prompt, predicted,
                args.layer, device, token_position=args.token_position,
            )
        except Exception as exc:
            print(f"\n  SKIPPED — {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        if layer_indices is None:
            layer_indices = curr_layer_indices

        is_correct = answers_match(predicted, true_answer)
        print(f"[{'OK' if is_correct else 'X'}] pred={predicted[:40]!r}  gt={true_answer[:30]!r}")

        open_vectors.append(vec.astype(np.float32))
        metadata.append({
            "hidden_row_index":  len(open_vectors) - 1,
            "row_id":            row["row_id"],
            "question":          question,
            "true_answer":       true_answer,
            "predicted_answer":  predicted,
            "is_correct":        bool(is_correct),
        })

    if not open_vectors:
        raise ValueError("No hidden states extracted.")

    processed = len(open_vectors)
    skipped   = len(records) - processed
    print(f"\nDone: {processed}/{len(records)} processed, {skipped} skipped")

    suffix = f"{csv_path.stem}_{args.model_name.split('/')[-1]}"
    open_matrix = np.stack(open_vectors, axis=0)

    arrays_path = output_dir / f"open_hidden_states_arrays_{suffix}.npz"
    if args.layer == "all":
        open_last = open_matrix[:, -1, :]
        np.savez_compressed(
            arrays_path,
            open_hidden_state=open_last,
            open_hidden_state_all_layers=open_matrix,
            layer_indices=np.asarray(layer_indices, dtype=np.int32),
        )
    else:
        np.savez_compressed(
            arrays_path,
            open_hidden_state=open_matrix,
            layer_indices=np.asarray(layer_indices, dtype=np.int32),
        )

    n_correct = sum(1 for r in metadata if r["is_correct"])
    summary = {
        "input_csv":      str(csv_path),
        "model_name":     args.model_name,
        "n_examples":     processed,
        "n_skipped":      skipped,
        "layer":          str(args.layer),
        "token_position": args.token_position,
        "max_new_tokens": args.max_new_tokens,
        "seed":           args.seed,
        "n_correct":      n_correct,
        "accuracy":       round(n_correct / processed, 4) if processed else None,
        "layer_indices":  layer_indices,
        "experiment_pkl": args.experiment_pkl,
    }

    (output_dir / f"open_hidden_states_{suffix}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / f"open_hidden_states_summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nOutput files saved to: {output_dir}")
    print(f"  Arrays  : {arrays_path.name}  shape={open_matrix.shape}")
    print(f"  Accuracy: {summary['accuracy']}")


if __name__ == "__main__":
    main()
