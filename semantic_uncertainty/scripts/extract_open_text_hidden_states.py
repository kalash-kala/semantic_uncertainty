#!/usr/bin/env python3
"""Extract hidden states for open-text QA from a CSV dataset.

Reads a CSV file with columns: question, ground_truth (and optionally id, context).
For each row:
  1. Builds an open-text prompt using the SAME few-shot structure as
     generate_answers_combined.py (loaded from experiment_details.pkl).
  2. Runs greedy decoding (do_sample=False) to generate a fresh answer.
  3. Feeds prompt + generated answer back through the model to extract hidden states.

IMPORTANT: When --experiment_pkl is provided, the prompt structure exactly matches
the generate_answers_combined.py pipeline:
  - Few-shot prefix from the original run
  - Raw text tokenization (NO chat template) matching HuggingfaceModel.predict()
  - Same stop-sequence stripping
  - Same max_new_tokens

Layer behavior:
  --layer last   : save last-layer hidden state vector (default)
  --layer <int>  : save a specific layer index
  --layer all    : save vectors for all layers; NPZ includes both
                   open_hidden_state_all_layers and the last-layer
                   open_hidden_state for compatibility

Token position (which token's hidden state to extract):
  --token_position last          : last token of the full sequence (default)
  --token_position last_prompt   : last token of the prompt
  --token_position first_answer  : first generated answer token

================================================================================
USAGE EXAMPLES
================================================================================

Basic run with experiment_pkl (recommended -- matches generate_answers_combined.py):
  python extract_open_text_hidden_states.py \\
    --input_csv /path/to/uncertainty_run_llama_sciq_combined_llm_verdict.csv \\
    --model_name meta-llama/Llama-3.1-8B-Instruct \\
    --experiment_pkl /path/to/experiment_details.pkl \\
    --output_dir ./hidden_states_open_only \\
    --layer all \\
    --device auto

SCIQ (10,000 samples) on GPU 0:
  nohup python extract_open_text_hidden_states.py --input_csv /home/kalashkala/Datasets/Semantic-Uncertainty/sciq/uncertainty_run_llama_sciq_combined_llm_verdict.csv --model_name meta-llama/Llama-3.1-8B-Instruct --experiment_pkl /data/kalashkala/semantic_uncertainty_data/uncertainty/sciq__meta-llama__Llama-3.1-8B-Instruct__seed10__pid.../experiment_details.pkl --output_dir /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_open_only --layer all --device cuda --gpu_id 0 --seed 42 > /home/kalashkala/sciq_open_hidden_states.out 2>&1 &

Legacy run WITHOUT experiment_pkl (zero-shot, chat template -- old behavior):
  python extract_open_text_hidden_states.py \\
    --input_csv /path/to/data.csv \\
    --model_name meta-llama/Llama-3.1-8B-Instruct \\
    --output_dir ./hidden_states_open_only \\
    --layer all \\
    --device auto

OUTPUT FILES (in --output_dir):
  open_hidden_states_arrays_{dataset}_{model}.npz   : compressed numpy arrays
  open_hidden_states_{dataset}_{model}.json         : per-example metadata
  open_hidden_states_summary_{dataset}_{model}.json : aggregate statistics

NPZ keys (--layer all):
  open_hidden_state            : [N, hidden_dim]  last-layer vectors
  open_hidden_state_all_layers : [N, n_layers, hidden_dim]
  layer_indices                : [n_layers]

NPZ keys (--layer last or specific int):
  open_hidden_state : [N, hidden_dim]
  layer_indices     : [1]
"""

from __future__ import annotations

import argparse
import ast
import csv
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

# Stop sequences matching generate_answers_combined.py / base_model.py
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']


class StoppingCriteriaSub(StoppingCriteria):
    """Matches HuggingfaceModel.StoppingCriteriaSub — stops when any stop string
    appears anywhere in the newly generated text."""
    def __init__(self, stops, tokenizer, initial_length):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer
        self.initial_length = initial_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generation = self.tokenizer.decode(
            input_ids[0][self.initial_length:], skip_special_tokens=False)
        for stop in self.stops:
            if stop in generation:
                return True
        return False


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
    if choice == "cuda":
        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto mode
    if gpu_id is not None:
        return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Experiment details loading
# ---------------------------------------------------------------------------

def load_experiment_details(pkl_path: str) -> Dict[str, Any]:
    """Load experiment_details.pkl from a generate_answers_combined.py run."""
    with open(pkl_path, "rb") as f:
        details = pickle.load(f)
    return details


# ---------------------------------------------------------------------------
# Prompt construction (matching generate_answers_combined.py)
# ---------------------------------------------------------------------------

def build_question_input(question: str, context: Optional[str],
                         use_context: bool, brief: str,
                         brief_always: bool) -> str:
    """Build the per-question input, matching utils.get_make_prompt (default).

    This replicates:
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\\n"
            prompt += f"Question: {question}\\n"
            if answer:
                prompt += f"Answer: {answer}\\n\\n"
            else:
                prompt += 'Answer:'
            return prompt
    """
    parts = ''
    if brief_always:
        parts += brief
    if use_context and context:
        parts += f"Context: {context}\n"
    parts += f"Question: {question}\n"
    parts += 'Answer:'
    return parts


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def parse_ground_truth(text: str) -> str:
    """Parse ground_truth from list-string format like \"['answer']\" to \"answer\"."""
    if not text:
        return ""
    text = text.strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return text


def split_context_question(combined: str) -> Tuple[str, str]:
    """Split a combined svamp-style string into (context, question).

    The svamp CSV stores context + question joined by a single '\\n', e.g.:
        "There were 8 people on the bus. At the next stop 12 more got on.\\nHow many people are there now?"

    Strategy: rsplit on the last '\\n' — everything before is the context body,
    everything after is the question sentence.
    """
    parts = combined.rsplit('\n', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # No newline — treat the whole string as the question
    return '', combined.strip()


def strip_stop_sequences(answer: str) -> str:
    """Strip stop sequences from the answer, matching HuggingfaceModel.predict()."""
    sliced = answer
    for stop in STOP_SEQUENCES:
        if sliced.endswith(stop):
            sliced = sliced[: len(sliced) - len(stop)]
            break
    return sliced.strip()


def clean_answer(text: str) -> str:
    x = str(text or "").strip()
    x = re.sub(
        r"^\s*(?:<\|im_start\|>\s*)?(?:system|user|assistant)\s*(?:<\|im_end\|>)?\s*",
        "",
        x,
        flags=re.IGNORECASE,
    )
    lines = [line.strip() for line in x.splitlines() if line.strip()]
    if lines and lines[0].lower() in {"system", "user", "assistant"}:
        lines = lines[1:]
    x = " ".join(lines).strip()
    x = re.sub(r"^\s*(the\s+answer\s+is|answer\s*:?)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^\s*[:\-\"'`]+\s*", "", x)
    x = x.strip(" \t\n\r\"'`[](){}")
    x = re.sub(r"[.\s]+$", "", x)
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_answer(text: str) -> str:
    return clean_answer(text).lower()


def answers_match(predicted: str, true_answer: str) -> bool:
    p = normalize_answer(predicted)
    t = normalize_answer(true_answer)
    if not p or not t:
        return False
    return p == t or p in t or t in p


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_records(path: Path, max_examples: int, seed: int) -> List[Dict[str, Any]]:
    """Load CSV and return list of dicts with 'question' and 'true_answer' keys.

    Expects at minimum columns: question, ground_truth.
    Optional columns: id, context (used for tracing back to the original dataset).
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            question = row.get("question", "").strip()
            if not question:
                continue
            records.append(
                {
                    "row_id": row.get("id", ""),
                    "question": question,
                    "true_answer": parse_ground_truth(row.get("ground_truth", "")),
                    "context": row.get("context", ""),
                }
            )

    if max_examples > 0 and len(records) > max_examples:
        rng = random.Random(seed)
        records = rng.sample(records, max_examples)

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
# Generation (matching HuggingfaceModel.predict() -- raw text tokenization)
# ---------------------------------------------------------------------------

def generate_answer_matched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Greedy decode matching HuggingfaceModel.predict().

    Uses StoppingCriteriaSub (same as HuggingfaceModel) so generation halts
    as soon as any stop sequence appears in the decoded output, rather than
    running to max_new_tokens and stripping post-hoc.
    """
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    n_input_tokens = inputs['input_ids'].shape[1]
    pad_token_id = getattr(tokenizer, 'pad_token_id', None) or tokenizer.eos_token_id

    # Match HuggingfaceModel: stop sequences + eos_token
    stop_sequences = STOP_SEQUENCES + [tokenizer.eos_token]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        stops=stop_sequences,
        tokenizer=tokenizer,
        initial_length=n_input_tokens,
    )])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            stopping_criteria=stopping_criteria,
        )

    answer_tokens = outputs[0][n_input_tokens:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Strip trailing stop sequences, matching HuggingfaceModel.predict()
    answer = strip_stop_sequences(answer)
    return answer


# ---------------------------------------------------------------------------
# Hidden state extraction (raw text tokenization when using experiment_pkl)
# ---------------------------------------------------------------------------

def extract_hidden_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_prompt: str,
    response: str,
    layer: LayerSpec,
    device: torch.device,
    token_position: str = "last",
) -> Tuple[np.ndarray, List[int]]:
    """Return hidden-state vector(s) for prompt+response at the requested token position.

    Uses raw text tokenization (no chat template) to match HuggingfaceModel.predict().

    token_position choices:
      'last'         : last token of the full (prompt+response) sequence
      'last_prompt'  : last token of the prompt only
      'first_answer' : first token of the generated response
    """
    # Tokenize full prompt + response as raw text
    combined_text = full_prompt + response
    input_ids = tokenizer(combined_text, return_tensors="pt").input_ids.to(device)

    # Tokenize prompt alone to find the boundary
    prompt_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    layer_indices = select_layer_indices(hidden_states, layer)

    seq_len = input_ids.shape[1]
    if token_position == "last":
        target_idx = -1
    elif token_position == "last_prompt":
        target_idx = prompt_len - 1
        if target_idx < 0 or target_idx >= seq_len:
            target_idx = -1
    elif token_position == "first_answer":
        target_idx = prompt_len
        if target_idx < 0 or target_idx >= seq_len:
            target_idx = -1
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
# Layer helpers
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
        if idx < 0 or idx >= n:
            raise IndexError(f"Layer index {idx} out of range for {n} layers")
        return [idx]
    raise ValueError(f"Unsupported layer spec: {layer}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract open-text hidden states from a CSV dataset using greedy decoding."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to the input CSV file (must have 'question' and 'ground_truth' columns)",
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face causal LM name or local path",
    )
    parser.add_argument(
        "--experiment_pkl", type=str, default=None,
        help="Path to experiment_details.pkl from a generate_answers_combined.py run. "
             "When provided, the few-shot prompt, use_context, and max_new_tokens "
             "are loaded from the original run to ensure identical greedy predictions.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="hidden_states_open_only",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--layer", type=parse_layer_spec, default="all",
        help="Layer index, 'last', or 'all'",
    )
    parser.add_argument(
        "--token_position", type=str, default="last",
        choices=["last", "last_prompt", "first_answer"],
        help="Which token position's hidden state to save",
    )
    parser.add_argument(
        "--max_examples", type=int, default=0,
        help="Maximum number of examples to process (0 = all)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=10,
        help="Maximum new tokens for greedy generation (overridden by experiment_pkl if provided)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (used when --max_examples < dataset size)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None,
        help="GPU device ID to use (0, 1, 2, etc.). Only applies when --device is cuda or auto",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def extract_dataset_name(csv_path: Path) -> str:
    return csv_path.stem


def extract_model_short(model_name: str) -> str:
    return model_name.split("/")[-1]


def build_summary(
    metadata: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    csv_path: Path,
) -> Dict[str, Any]:
    n_correct = sum(1 for r in metadata if r.get("is_correct"))
    n_total = len(metadata)
    return {
        "input_csv": str(csv_path),
        "model_name": args.model_name,
        "n_examples": n_total,
        "layer": str(args.layer),
        "token_position": args.token_position,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_total, 4) if n_total > 0 else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment details if provided
    fewshot_prefix = None
    use_context = False
    brief = ""
    brief_always = False

    if args.experiment_pkl:
        print(f"Loading experiment details from: {args.experiment_pkl}")
        exp_details = load_experiment_details(args.experiment_pkl)
        fewshot_prefix = exp_details['prompt']
        brief = exp_details.get('BRIEF', '')
        exp_args = exp_details['args']
        use_context = getattr(exp_args, 'use_context', False)
        brief_always = getattr(exp_args, 'brief_always', False) and getattr(exp_args, 'enable_brief', True)
        # Override max_new_tokens from original run
        orig_max_new_tokens = getattr(exp_args, 'model_max_new_tokens', None)
        if orig_max_new_tokens is not None:
            args.max_new_tokens = orig_max_new_tokens
            print(f"  Using max_new_tokens={args.max_new_tokens} from experiment_details")
        print(f"  use_context={use_context}")
        print(f"  brief_always={brief_always}")
        print(f"  Few-shot prefix length: {len(fewshot_prefix)} chars")
        print(f"  Few-shot prefix preview: {fewshot_prefix[:200]!r}...")
    else:
        print("WARNING: No --experiment_pkl provided. Using zero-shot raw-text prompts.")
        print("         Predictions will NOT match generate_answers_combined.py.")

    print(f"\n{'='*80}")
    print(f"Loading records from: {csv_path}")
    records = load_csv_records(csv_path, args.max_examples, args.seed)
    if not records:
        raise ValueError(f"No usable records found in {csv_path}")
    print(f"Loaded {len(records)} records")
    print(f"{'='*80}\n")

    device = resolve_device(args.device, args.gpu_id)
    print(f"Loading model: {args.model_name} on {device}")
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)
    print("Model loaded.\n")

    open_vectors: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    layer_indices: Optional[List[int]] = None

    print(f"{'='*80}")
    print(f"Processing {len(records)} examples  |  token_position={args.token_position}  |  layer={args.layer}")
    if fewshot_prefix:
        print(f"Prompt mode: few-shot (from experiment_pkl)  |  max_new_tokens={args.max_new_tokens}")
    else:
        print(f"Prompt mode: zero-shot (no experiment_pkl)  |  max_new_tokens={args.max_new_tokens}")
    print(f"{'='*80}\n")

    for idx, row in enumerate(records):
        question = row["question"]
        true_answer = row["true_answer"]
        context = row.get("context", "") or None

        # Build the full prompt matching generate_answers_combined.py.
        # For svamp (use_context=True) the CSV has no context column — the
        # question field contains "context body\nquestion sentence" so we
        # split on the last newline to recover both parts.
        if use_context and not context:
            context, question = split_context_question(question)

        if fewshot_prefix is not None:
            question_input = build_question_input(
                question, context, use_context, brief, brief_always)
            full_prompt = fewshot_prefix + question_input
        else:
            # Legacy zero-shot fallback
            full_prompt = f"Question: {question}\nAnswer:"

        print(
            f"[{idx+1}/{len(records)}] Generating answer...",
            end=" ",
            flush=True,
        )

        try:
            predicted = generate_answer_matched(
                model, tokenizer, full_prompt, device, args.max_new_tokens
            )
            vec, curr_layer_indices = extract_hidden_state(
                model, tokenizer,
                full_prompt, predicted,
                args.layer, device,
                token_position=args.token_position,
            )
        except Exception as exc:
            print(f"\n  SKIPPED — {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        if layer_indices is None:
            layer_indices = curr_layer_indices

        is_correct = answers_match(predicted, true_answer)
        status = "OK" if is_correct else "X"
        print(f"[{status}] pred={predicted[:40]!r}  gt={true_answer[:30]!r}")

        hidden_row_index = len(open_vectors)
        open_vectors.append(vec.astype(np.float32))
        metadata.append(
            {
                "hidden_row_index": hidden_row_index,
                "row_id": row.get("row_id", ""),
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted,
                "is_correct": bool(is_correct),
            }
        )

    # ------------------------------------------------------------------
    # Assemble and save
    # ------------------------------------------------------------------
    if not open_vectors:
        raise ValueError("No hidden states extracted. Check model access and input.")
    if layer_indices is None:
        raise RuntimeError("No valid layer indices collected.")

    processed = len(open_vectors)
    skipped = len(records) - processed
    print(f"\n{'='*80}")
    print(f"Done: {processed}/{len(records)} processed, {skipped} skipped")
    print(f"{'='*80}\n")

    dataset_name = extract_dataset_name(csv_path)
    model_short = extract_model_short(args.model_name)
    suffix = f"{dataset_name}_{model_short}"

    open_matrix = np.stack(open_vectors, axis=0)  # [N, hidden_dim] or [N, n_layers, hidden_dim]

    arrays_path = output_dir / f"open_hidden_states_arrays_{suffix}.npz"
    if args.layer == "all":
        # open_matrix shape: [N, n_layers, hidden_dim]
        open_last = open_matrix[:, -1, :]  # [N, hidden_dim]
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

    import json
    metadata_path = output_dir / f"open_hidden_states_{suffix}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = build_summary(metadata, args, csv_path)
    summary["layer_indices"] = layer_indices
    summary["experiment_pkl"] = args.experiment_pkl
    summary_path = output_dir / f"open_hidden_states_summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Output files saved:")
    print(f"  Arrays   : {arrays_path}")
    print(f"  Metadata : {metadata_path}")
    print(f"  Summary  : {summary_path}")
    print(f"  Shape    : {open_matrix.shape}")
    print(f"  Accuracy : {summary['accuracy']}")


if __name__ == "__main__":
    main()
