#!/usr/bin/env python3
"""Run paired open-text and MCQ greedy inference and dump hidden states.

This script pairs rows from open-text category files (JSON or CSV) and MCQ category JSON files,
runs greedy decoding for both prompts, and saves:
- open hidden state
- mcq hidden state
- delta hidden state (mcq - open)
- delta norm

Layer behavior:
- --layer last: save last-layer vectors (default)
- --layer <int>: save a specific layer index
- --layer all: save vectors for all layers from first to last

When --layer all is used, the NPZ includes both:
- *_all_layers arrays
- compatibility last-layer arrays under the original key names

Supports:
- JSON files (legacy format)
- CSV files (modern format) with columns: question, ground_truth, etc.
- Dynamic category discovery from filenames

================================================================================
USAGE EXAMPLES
================================================================================

1. For CSV + JSON paired files (modern format):
   python plot_hidden_state_tsne_three_categories_mcq_neg_dual_with_hidden_dump.py \\
     --input_open_dir /path/to/csv/files \\
     --input_mcq_dir /path/to/mcq/json/files \\
     --model_name meta-llama/Llama-3.1-8B-Instruct \\
     --output_dir ./output_hidden_states \\
     --layer all \\
     --device auto

2. Specific example with your data structure:
   Directory structure expected:
   /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/
   ├── sciq__llama__0.5/              (open-text CSV files)
   │   ├── detail_sciq__llama_entropy_0.5_Correct.csv
   │   ├── detail_sciq__llama_entropy_0.5_AH.csv
   │   └── detail_sciq__llama_entropy_0.5_UH.csv
   └── sciq__llama__0.5__mcq/         (MCQ JSON files)
       ├── detail_sciq__llama_entropy_0.5_Correct_mcq.json
       ├── detail_sciq__llama_entropy_0.5_AH_mcq.json
       └── detail_sciq__llama_entropy_0.5_UH_mcq.json

   Command:
    sciq: python plot_hidden_state_tsne_three_categories_mcq_neg_dual_with_hidden_dump.py --input_open_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5 --input_mcq_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5__mcq --model_name meta-llama/Llama-3.1-8B-Instruct --output_dir /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output --layer all --max_examples_per_category 100 --device auto --seed 42
    svamp: python plot_hidden_state_tsne_three_categories_mcq_neg_dual_with_hidden_dump.py --input_open_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/svamp__llama__0.5 --input_mcq_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/svamp__llama__0.5__mcq --model_name meta-llama/Llama-3.1-8B-Instruct --output_dir /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output --layer all --max_examples_per_category 100 --device auto --seed 42

3. For small test run (2 examples per category):
   python plot_hidden_state_tsne_three_categories_mcq_neg_dual_with_hidden_dump.py --input_open_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5 --input_mcq_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5__mcq --model_name meta-llama/Llama-2-7b-hf --output_dir /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output --layer all --max_examples_per_category 2 --device auto --seed 42

4. nohup commands:
    sciq:
    nohup python plot_hidden_state_tsne_three_categories_mcq_neg_dual_with_hidden_dump.py --input_open_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5 --input_mcq_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/sciq__llama__0.5__mcq --model_name meta-llama/Llama-3.1-8B-Instruct --output_dir /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output --layer all --max_examples_per_category 100 --device auto --seed 42 > sciq_llama_3.1_8b_instruct_hidden_states.out 2>&1 &
    
    svamp:
    nohup python plot_hidden_state_tsne_three_categories_mcq_neg_dual_with_hidden_dump.py --input_open_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/svamp__llama__0.5 --input_mcq_dir /home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/svamp__llama__0.5__mcq --model_name meta-llama/Llama-3.1-8B-Instruct --output_dir /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output --layer all --max_examples_per_category 100 --device auto --seed 42 > svamp_llama_3.1_8b_instruct_hidden_states.out 2>&1 &


FILE DISCOVERY:
- CSV files in --input_open_dir are automatically paired with JSON files in --input_mcq_dir
- Pairing is done by matching base filenames:
  - "detail_sciq__llama_entropy_0.5_Correct.csv" ↔ "detail_sciq__llama_entropy_0.5_Correct_mcq.json"
- Category labels are auto-detected from filenames (Correct/AH/UH)

OUTPUT FILES:
- paired_hidden_states_arrays_{dataset}_{model}.npz    : Numpy compressed arrays of hidden states
- paired_hidden_states_{dataset}_{model}.json          : Per-example metadata and predictions
- paired_hidden_states_summary_{dataset}_{model}.json  : Aggregate statistics
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LayerSpec = Union[str, int]

CHAT_MODEL_KEYWORDS = ("llama", "qwen", "mistral")

BRIEF_PROMPTS = {
    'default': 'Answer the following question as briefly as possible.\n',
    'chat': 'Answer the following question in a single brief but complete sentence.\n',
}

# Stricter prompts that prevent reasoning output
MCQ_PROMPTS = {
    'default': 'Respond with ONLY the option letter and answer value in format: LETTER. VALUE (e.g., A. 9). No reasoning.\n',
    'chat': 'Respond with ONLY the option letter and answer value in format: LETTER. VALUE (e.g., A. 9). No reasoning.\n',
}

OPEN_TEXT_PROMPTS = {
    'default': 'Answer with ONLY the final answer. Do not show reasoning, steps, or calculations.\n',
    'chat': 'Answer with ONLY the final answer. Do not show reasoning, steps, or calculations.\n',
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ground_truth(text: str) -> str:
    """Parse ground_truth from list string format like "['answer']" to "answer"."""
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


def load_csv_records(path: Path) -> List[Dict[str, Any]]:
    """Load CSV file and convert to standard format with 'prompt' and 'true_answer'."""
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            record = dict(row)
            record["prompt"] = record.get("question", "")
            record["true_answer"] = parse_ground_truth(record.get("ground_truth", ""))
            records.append(record)
    return records


def discover_category_files(open_dir: Path, mcq_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Dynamically discover matching open-text and MCQ file pairs.

    Returns a dict mapping category name to (open_file, mcq_file) tuples.
    Supports both JSON and CSV formats.
    """
    file_pairs: Dict[str, Tuple[Path, Path]] = {}

    if not open_dir.exists() or not mcq_dir.exists():
        return file_pairs

    open_files = {}
    for f in open_dir.iterdir():
        if f.is_file() and (f.suffix == ".json" or f.suffix == ".csv"):
            stem = f.stem
            open_files[stem] = f

    for mcq_file in mcq_dir.iterdir():
        if not mcq_file.is_file() or mcq_file.suffix != ".json":
            continue

        stem = mcq_file.stem
        base_stem = stem.replace("_mcq", "")

        if base_stem in open_files:
            file_pairs[base_stem] = (open_files[base_stem], mcq_file)

    return file_pairs


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSON file and convert to standard format."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    obj = json.loads(text)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list in {path}")
    out: List[Dict[str, Any]] = []
    for row in obj:
        if isinstance(row, dict):
            record = dict(row)
            if "prompt" not in record and "question" in record:
                record["prompt"] = record["question"]
            if "true_answer" not in record and "ground_truth" in record:
                record["true_answer"] = parse_ground_truth(record["ground_truth"])
            out.append(record)
    return out


def load_records_auto(path: Path) -> List[Dict[str, Any]]:
    """Load records from either CSV or JSON file based on extension."""
    if path.suffix == ".csv":
        return load_csv_records(path)
    else:
        return load_json_records(path)


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


def extract_option_letter(text: str) -> Optional[str]:
    cleaned = clean_answer(text)
    match = re.match(r"^\s*([A-Da-d])(?:\b|[\).:\-])", cleaned)
    if not match:
        return None
    return match.group(1).upper()


def get_mcq_option_map(row: Dict[str, Any]) -> Dict[str, str]:
    meta = row.get("mcq_meta")
    if not isinstance(meta, dict):
        return {}
    options = meta.get("options")
    if not isinstance(options, list):
        return {}

    mapped: Dict[str, str] = {}
    for opt in options:
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label", "")).strip().upper()
        text = str(opt.get("text", "")).strip()
        if label in {"A", "B", "C", "D"} and text:
            mapped[label] = text
    return mapped


def is_correct_for_mcq_row(row: Dict[str, Any], predicted: str, true_answer: str) -> bool:
    option_map = get_mcq_option_map(row)
    meta = row.get("mcq_meta")
    correct_option: Optional[str] = None
    if isinstance(meta, dict):
        candidate = str(meta.get("correct_option", "")).strip().upper()
        if candidate in {"A", "B", "C", "D"}:
            correct_option = candidate

    predicted_option = extract_option_letter(predicted)
    if correct_option is not None and predicted_option is not None:
        return predicted_option == correct_option

    if correct_option is not None and correct_option in option_map:
        if answers_match(predicted, option_map[correct_option]):
            return True

    return answers_match(predicted, true_answer)


def label_from_source_category(source_category: Any) -> str:
    """Map source category name to label.

    Supports both hardcoded names (legacy) and dynamic names from filenames.
    """
    category_str = str(source_category).lower()
    mapping = {
        "certain_prediction": "correct",
        "certain_misprediction": "AH_candidate",
        "uncertain_misprediction": "UH_candidate",
    }

    if category_str in mapping:
        return mapping[category_str]

    if "correct" in category_str:
        return "correct"
    if "ah" in category_str or "misprediction" in category_str:
        return "AH_candidate"
    if "uh" in category_str or "uncertain" in category_str:
        return "UH_candidate"

    return "UH_candidate"


def parse_layer_spec(layer_arg: str) -> LayerSpec:
    if layer_arg in {"last", "all"}:
        return layer_arg
    try:
        return int(layer_arg)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid layer value: {layer_arg}") from exc


def select_layer_indices(hidden_states: Tuple[torch.Tensor, ...], layer: LayerSpec) -> List[int]:
    n_layers = len(hidden_states)
    if layer == "all":
        return list(range(n_layers))
    if layer == "last":
        return [n_layers - 1]
    if isinstance(layer, int):
        idx = layer if layer >= 0 else n_layers + layer
        if idx < 0 or idx >= n_layers:
            raise IndexError(f"Layer index {idx} out of range for {n_layers} hidden-state tensors")
        return [idx]
    raise ValueError(f"Unsupported layer spec: {layer}")


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_chat_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(keyword in lowered for keyword in CHAT_MODEL_KEYWORDS)


def get_trailing_token_ids(tokenizer: AutoTokenizer, token_texts: Sequence[str]) -> set[int]:
    token_ids: set[int] = set()
    tokenized = tokenizer(token_texts, add_special_tokens=False)["input_ids"]
    for sequence in tokenized:
        token_ids.update(sequence)

    vocab = tokenizer.get_vocab()
    for token_text in token_texts:
        token_id = vocab.get(token_text)
        if token_id is not None:
            token_ids.add(token_id)
    return token_ids


def build_chat_messages(prompt: str) -> List[Dict[str, str]]:
    if prompt.count("question:") >= 4:
        split_prompt = [x.strip() for x in prompt.split("\n") if x.strip()]
        split_prompt = split_prompt[:-1]
        return [
            {"role": "assistant", "content": x.replace("answer: ", "") + "\n"}
            if i % 2 == 1
            else {"role": "user", "content": x.replace("question: ", "") + "\n"}
            for i, x in enumerate(split_prompt)
        ]
    return [{"role": "user", "content": prompt}]


def encode_prompt_for_model(prompt: str, model_name: str, tokenizer: AutoTokenizer, device: torch.device) -> torch.Tensor:
    if is_chat_model_name(model_name) and getattr(tokenizer, "chat_template", None):
        messages = build_chat_messages(prompt)
        messages += [{"role": "assistant", "content": " The answer is "}]

        # Use continue_final_message=True (transformers ≥4.44) so the template
        # continues the in-progress assistant turn instead of starting a new one.
        # Fall back to add_generation_prompt=False for older transformers.
        # Older transformers returns a BatchEncoding from apply_chat_template;
        # newer versions return a plain tensor. Unwrap either case.
        try:
            raw = tokenizer.apply_chat_template(
                messages,
                continue_final_message=True,
                return_tensors="pt",
            )
        except TypeError:
            raw = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                return_tensors="pt",
            )
        input_ids = (raw["input_ids"] if hasattr(raw, "keys") else raw).to(device)

        unwanted_tokens_embedded = get_trailing_token_ids(
            tokenizer,
            [
                "<|eot_id|>",
                "<|start_header_id|>",
                "assistant",
                "<|end_header_id|>",
                "\n",
                "<end_of_turn>",
                "<start_of_turn>",
                "model",
                " ",
                "\n\n",
                "</s>",
                "<|im_start|>",
                "<|im_end|>",
            ],
        )

        while input_ids.shape[1] > 1 and int(input_ids[0, -1]) in unwanted_tokens_embedded:
            input_ids = input_ids[:, :-1]
        return input_ids

    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def load_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_natural_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    input_ids = encode_prompt_for_model(prompt, model_name, tokenizer, device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return clean_answer(text)


def extract_response_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    response: str,
    layer: LayerSpec,
    device: torch.device,
) -> Tuple[np.ndarray, List[int]]:
    if is_chat_model_name(model_name) and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        raw = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        input_ids = (raw["input_ids"] if hasattr(raw, "keys") else raw).to(device)
    else:
        input_ids = tokenizer(f"{prompt} {response}", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    layer_indices = select_layer_indices(hidden_states, layer)
    selected = [hidden_states[i][0, -1, :].detach().cpu().float().numpy() for i in layer_indices]

    if len(selected) == 1:
        return selected[0], layer_indices
    return np.stack(selected, axis=0), layer_indices


def load_paired_records(open_dir: Path, mcq_dir: Path, max_examples_per_category: int, seed: int) -> List[Dict[str, Any]]:
    """Load paired records from matching open-text and MCQ files.

    Files are discovered dynamically and can be either JSON or CSV format.
    """
    all_pairs: List[Dict[str, Any]] = []
    file_pairs = discover_category_files(open_dir, mcq_dir)

    if not file_pairs:
        raise ValueError(f"No matching file pairs found in {open_dir} and {mcq_dir}")

    for category, (open_file, mcq_file) in sorted(file_pairs.items()):
        open_rows = load_records_auto(open_file)
        mcq_rows = load_json_records(mcq_file)

        n = min(len(open_rows), len(mcq_rows))
        if n == 0:
            print(f"Warning: No valid pairs for category {category}")
            continue

        indices = list(range(n))
        if max_examples_per_category > 0 and n > max_examples_per_category:
            rng = random.Random(seed)
            indices = rng.sample(indices, max_examples_per_category)

        for i in indices:
            open_row = dict(open_rows[i])
            mcq_row = dict(mcq_rows[i])

            open_prompt = open_row.get("prompt")
            mcq_prompt = mcq_row.get("mcq_question_full")
            true_answer = mcq_row.get("true_answer", open_row.get("true_answer"))
            if not isinstance(open_prompt, str) or not isinstance(mcq_prompt, str) or not isinstance(true_answer, str):
                continue

            all_pairs.append(
                {
                    "id": len(all_pairs),
                    "source_category": category,
                    "source_label": label_from_source_category(category),
                    "source_index": i,
                    "open_prompt": open_prompt,
                    "mcq_prompt": mcq_prompt,
                    "open_row": open_row,
                    "mcq_row": mcq_row,
                    "true_answer": true_answer,
                }
            )
    return all_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired open-text and MCQ greedy inference and dump hidden states.")
    parser.add_argument("--input_open_dir", type=str, default="three_category_jsons", help="Directory containing open-text category JSON files")
    parser.add_argument("--input_mcq_dir", type=str, default="mcq_neg", help="Directory containing MCQ category JSON files")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face causal LM name or local path")
    parser.add_argument("--output_dir", type=str, default="hidden_states_open_mcq_paired_all", help="Output directory")
    parser.add_argument("--layer", type=parse_layer_spec, default="all", help="Layer index, 'last', or 'all'")
    parser.add_argument("--max_examples_per_category", type=int, default=0, help="0 means all usable examples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens for greedy generation")
    parser.add_argument("--enable_brief", action="store_true", default=False,
                        help="Prepend a brief instruction to all prompts")
    parser.add_argument("--brief_prompt", type=str, default="default",
                        choices=["default", "chat"],
                        help="Which brief instruction style to use (default or chat)")
    return parser.parse_args()


def extract_dataset_name(open_dir: Path) -> str:
    """Extract dataset name from the open directory path."""
    return open_dir.name


def extract_model_name(model_name: str) -> str:
    """Extract model name from Hugging Face model identifier."""
    return model_name.split("/")[-1]


def build_summary(metadata: Sequence[Dict[str, Any]], args: argparse.Namespace, open_dir: Path, mcq_dir: Path) -> Dict[str, Any]:
    counts_source = {"correct": 0, "AH_candidate": 0, "UH_candidate": 0}
    open_correct = 0
    mcq_correct = 0
    delta_norms: List[float] = []

    for row in metadata:
        src_label = row.get("source_label", "UH_candidate")
        if src_label in counts_source:
            counts_source[src_label] += 1
        open_correct += int(bool(row.get("open_is_correct")))
        mcq_correct += int(bool(row.get("mcq_is_correct")))

        dn = row.get("delta_hidden_norm_last")
        if isinstance(dn, (int, float)):
            delta_norms.append(float(dn))

    return {
        "input_open_dir": str(open_dir),
        "input_mcq_dir": str(mcq_dir),
        "model_name": args.model_name,
        "n_examples": len(metadata),
        "layer": args.layer,
        "counts_by_source_label": counts_source,
        "open_correct_count": open_correct,
        "mcq_correct_count": mcq_correct,
        "mean_delta_hidden_norm_last": float(np.mean(delta_norms)) if delta_norms else None,
        "var_delta_hidden_norm_last": float(np.var(delta_norms)) if delta_norms else None,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    open_dir = Path(args.input_open_dir)
    mcq_dir = Path(args.input_mcq_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_paired_records(open_dir, mcq_dir, args.max_examples_per_category, args.seed)
    if not records:
        raise ValueError("No usable paired examples found in the provided directories")

    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)

    open_vectors: List[np.ndarray] = []
    mcq_vectors: List[np.ndarray] = []
    delta_vectors: List[np.ndarray] = []
    delta_norms_list: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    layer_indices: Optional[List[int]] = None

    print(f"\n{'='*80}")
    print(f"Starting processing of {len(records)} paired examples")
    print(f"{'='*80}\n")

    for idx, row in enumerate(records):
        open_prompt = row["open_prompt"]
        mcq_prompt = row["mcq_prompt"]
        true_answer = row["true_answer"]
        source_category = row.get("source_category", "unknown")
        source_label = row.get("source_label", "unknown")

        # Apply stricter, type-specific prompts if brief instruction enabled
        if args.enable_brief:
            mcq_prefix = MCQ_PROMPTS[args.brief_prompt]
            open_prefix = OPEN_TEXT_PROMPTS[args.brief_prompt]
            open_prompt = open_prefix + open_prompt
            mcq_prompt = mcq_prefix + mcq_prompt

        print(f"[{idx+1}/{len(records)}] Processing example from category '{source_category}' ({source_label})...", end=" ", flush=True)

        try:
            open_predicted = generate_natural_answer(
                model,
                tokenizer,
                args.model_name,
                open_prompt,
                device,
                args.max_new_tokens,
            )
            open_vec, open_layer_indices = extract_response_hidden(
                model,
                tokenizer,
                args.model_name,
                open_prompt,
                open_predicted,
                args.layer,
                device,
            )

            mcq_predicted = generate_natural_answer(
                model,
                tokenizer,
                args.model_name,
                mcq_prompt,
                device,
                args.max_new_tokens,
            )
            mcq_vec, mcq_layer_indices = extract_response_hidden(
                model,
                tokenizer,
                args.model_name,
                mcq_prompt,
                mcq_predicted,
                args.layer,
                device,
            )
        except Exception as exc:
            print(f"\n  ✗ SKIPPED due to error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        if open_layer_indices != mcq_layer_indices:
            raise RuntimeError("Layer index mismatch between open and MCQ hidden states")
        if layer_indices is None:
            layer_indices = open_layer_indices

        open_is_correct = answers_match(open_predicted, true_answer)
        mcq_is_correct = is_correct_for_mcq_row(row["mcq_row"], mcq_predicted, true_answer)

        status = "✓" if (open_is_correct and mcq_is_correct) else "~"
        print(f"{status} (open={open_predicted[:30]:30s}... mcq={mcq_predicted[:30]:30s}...)")

        delta_raw = mcq_vec - open_vec
        if delta_raw.ndim == 1:
            delta_norm = float(np.linalg.norm(delta_raw))
            delta_unit = (delta_raw / delta_norm).astype(np.float32) if delta_norm > 0.0 else np.zeros_like(delta_raw, dtype=np.float32)
            delta_norm_arr = np.asarray(delta_norm, dtype=np.float32)
            delta_norm_last = delta_norm
        else:
            delta_norm_arr = np.linalg.norm(delta_raw, axis=1).astype(np.float32)
            safe = np.where(delta_norm_arr > 0.0, delta_norm_arr, 1.0).reshape(-1, 1)
            delta_unit = (delta_raw / safe).astype(np.float32)
            delta_norm_last = float(delta_norm_arr[-1])

        hidden_row_index = len(open_vectors)
        open_vectors.append(open_vec.astype(np.float32))
        mcq_vectors.append(mcq_vec.astype(np.float32))
        delta_vectors.append(delta_unit)
        delta_norms_list.append(delta_norm_arr)

        metadata.append(
            {
                "id": row.get("id", idx),
                "source_category": row.get("source_category"),
                "source_label": row.get("source_label"),
                "source_index": row.get("source_index"),
                "open_prompt": open_prompt,
                "mcq_prompt": mcq_prompt,
                "true_answer": true_answer,
                "open_predicted_answer": open_predicted,
                "open_is_correct": bool(open_is_correct),
                "mcq_predicted_answer": mcq_predicted,
                "mcq_is_correct": bool(mcq_is_correct),
                "hidden_row_index": hidden_row_index,
                "delta_hidden_norm_last": delta_norm_last,
            }
        )

    if not open_vectors:
        raise ValueError("No vectors extracted. Check model access and input format.")

    processed_count = len(open_vectors)
    skipped_count = len(records) - processed_count
    print(f"\n{'='*80}")
    print(f"Processing complete: {processed_count}/{len(records)} examples processed, {skipped_count} skipped")
    print(f"{'='*80}\n")

    if layer_indices is None:
        raise RuntimeError("No valid layer indices were collected")

    open_matrix = np.stack(open_vectors, axis=0)
    mcq_matrix = np.stack(mcq_vectors, axis=0)
    delta_matrix = np.stack(delta_vectors, axis=0)

    if args.layer == "all":
        delta_norm_matrix = np.stack(delta_norms_list, axis=0)
        open_last = open_matrix[:, -1, :]
        mcq_last = mcq_matrix[:, -1, :]
        delta_last = delta_matrix[:, -1, :]
        delta_norm_last = delta_norm_matrix[:, -1]
    else:
        delta_norm_matrix = np.asarray(delta_norms_list, dtype=np.float32).reshape(-1)
        open_last = open_matrix
        mcq_last = mcq_matrix
        delta_last = delta_matrix
        delta_norm_last = delta_norm_matrix

    dataset_name = extract_dataset_name(open_dir)
    model_name_short = extract_model_name(args.model_name)
    file_suffix = f"{dataset_name}_{model_name_short}"

    arrays_path = output_dir / f"paired_hidden_states_arrays_{file_suffix}.npz"
    if args.layer == "all":
        np.savez_compressed(
            arrays_path,
            open_hidden_state=open_last,
            mcq_hidden_state=mcq_last,
            delta_hidden_state=delta_last,
            delta_hidden_norm=delta_norm_last,
            open_hidden_state_all_layers=open_matrix,
            mcq_hidden_state_all_layers=mcq_matrix,
            delta_hidden_state_all_layers=delta_matrix,
            delta_hidden_norm_all_layers=delta_norm_matrix,
            layer_indices=np.asarray(layer_indices, dtype=np.int32),
        )
    else:
        np.savez_compressed(
            arrays_path,
            open_hidden_state=open_matrix,
            mcq_hidden_state=mcq_matrix,
            delta_hidden_state=delta_matrix,
            delta_hidden_norm=delta_norm_matrix,
            layer_indices=np.asarray(layer_indices, dtype=np.int32),
        )

    points_path = output_dir / f"paired_hidden_states_{file_suffix}.json"
    points_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = build_summary(metadata, args, open_dir, mcq_dir)
    summary["layer_indices"] = layer_indices
    summary_path = output_dir / f"paired_hidden_states_summary_{file_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*80}")
    print("Output files saved:")
    print(f"  Metadata: {points_path}")
    print(f"  Summary:  {summary_path}")
    print(f"  Arrays:   {arrays_path}")
    print(f"  Total examples processed: {len(metadata)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
