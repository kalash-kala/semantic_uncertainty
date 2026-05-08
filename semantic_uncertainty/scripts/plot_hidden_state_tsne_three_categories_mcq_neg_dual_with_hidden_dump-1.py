#!/usr/bin/env python3
"""Run paired open-text and MCQ greedy inference and dump hidden states.

This script pairs rows from open-text category JSON files and MCQ category JSON files,
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
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LayerSpec = Union[str, int]

CHAT_MODEL_KEYWORDS = ("llama", "qwen", "mistral")

OPEN_CATEGORY_FILES = {
    "certain_prediction": "certain_predictions_pruned_100.json",
    "certain_misprediction": "certain_mispredictions_pruned_100.json",
    "uncertain_misprediction": "uncertain_mispredictions_pruned_100.json",
}

MCQ_CATEGORY_FILES = {
    "certain_prediction": "certain_predictions_pruned_100_negated_mcq.json",
    "certain_misprediction": "certain_mispredictions_pruned_100_negated_mcq.json",
    "uncertain_misprediction": "uncertain_mispredictions_pruned_100_negated_mcq.json",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    obj = json.loads(text)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list in {path}")
    out: List[Dict[str, Any]] = []
    for row in obj:
        if isinstance(row, dict):
            out.append(row)
    return out


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
    mapping = {
        "certain_prediction": "correct",
        "certain_misprediction": "AH_candidate",
        "uncertain_misprediction": "UH_candidate",
    }
    return mapping.get(str(source_category), "UH_candidate")


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

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

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
    token_position: str = "last",
) -> Tuple[np.ndarray, List[int]]:
    if is_chat_model_name(model_name) and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(device)
        
        prompt_messages = [
            {"role": "user", "content": prompt},
        ]
        try:
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            prompt_len = prompt_ids.shape[1]
        except Exception:
            prompt_len = input_ids.shape[1] - 1
    else:
        input_ids = tokenizer(f"{prompt} {response}", return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    layer_indices = select_layer_indices(hidden_states, layer)

    if token_position == "last":
        target_idx = -1
    elif token_position == "last_prompt":
        target_idx = prompt_len - 1
        if target_idx < 0 or target_idx >= input_ids.shape[1]:
            target_idx = -1
    elif token_position == "first_answer":
        target_idx = prompt_len
        if target_idx < 0 or target_idx >= input_ids.shape[1]:
            target_idx = -1
    else:
        raise ValueError(f"Unknown token_position: {token_position}")

    selected = [hidden_states[i][0, target_idx, :].detach().cpu().numpy().astype(np.float32) for i in layer_indices]

    if len(selected) == 1:
        return selected[0], layer_indices
    return np.stack(selected, axis=0), layer_indices


def load_paired_records(open_dir: Path, mcq_dir: Path, max_examples_per_category: int, seed: int) -> List[Dict[str, Any]]:
    all_pairs: List[Dict[str, Any]] = []
    for category, open_filename in OPEN_CATEGORY_FILES.items():
        mcq_filename = MCQ_CATEGORY_FILES[category]
        open_rows = load_json_records(open_dir / open_filename)
        mcq_rows = load_json_records(mcq_dir / mcq_filename)

        n = min(len(open_rows), len(mcq_rows))
        if n == 0:
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
    parser.add_argument("--output_dir", type=str, default="hidden_states_open_mcq_paired_all_first_answer", help="Output directory")
    parser.add_argument("--layer", type=parse_layer_spec, default="all", help="Layer index, 'last', or 'all'")
    parser.add_argument("--max_examples_per_category", type=int, default=0, help="0 means all usable examples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens for greedy generation")
    parser.add_argument("--token_position", type=str, default="first_answer", choices=["last", "last_prompt", "first_answer"], help="Which token's hidden state to save")
    return parser.parse_args()


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
        "token_position": args.token_position,
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
    delta_raw_vectors: List[np.ndarray] = []
    delta_norms_list: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    layer_indices: Optional[List[int]] = None

    for idx, row in enumerate(records):
        open_prompt = row["open_prompt"]
        mcq_prompt = row["mcq_prompt"]
        true_answer = row["true_answer"]

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
                token_position=args.token_position,
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
                token_position=args.token_position,
            )
        except Exception as exc:
            print(f"Skipping example {idx} due to model error: {exc}")
            continue

        if open_layer_indices != mcq_layer_indices:
            raise RuntimeError("Layer index mismatch between open and MCQ hidden states")
        if layer_indices is None:
            layer_indices = open_layer_indices

        open_is_correct = answers_match(open_predicted, true_answer)
        mcq_is_correct = is_correct_for_mcq_row(row["mcq_row"], mcq_predicted, true_answer)

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
        delta_raw_vectors.append(delta_raw.astype(np.float32))
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

    if layer_indices is None:
        raise RuntimeError("No valid layer indices were collected")

    open_matrix = np.stack(open_vectors, axis=0)
    mcq_matrix = np.stack(mcq_vectors, axis=0)
    delta_matrix = np.stack(delta_vectors, axis=0)
    delta_raw_matrix = np.stack(delta_raw_vectors, axis=0)

    if args.layer == "all":
        delta_norm_matrix = np.stack(delta_norms_list, axis=0)
        open_last = open_matrix[:, -1, :]
        mcq_last = mcq_matrix[:, -1, :]
        delta_last = delta_matrix[:, -1, :]
        delta_raw_last = delta_raw_matrix[:, -1, :]
        delta_norm_last = delta_norm_matrix[:, -1]
    else:
        delta_norm_matrix = np.asarray(delta_norms_list, dtype=np.float32).reshape(-1)
        open_last = open_matrix
        mcq_last = mcq_matrix
        delta_last = delta_matrix
        delta_raw_last = delta_raw_matrix
        delta_norm_last = delta_norm_matrix

    arrays_path = output_dir / "paired_hidden_states_arrays.npz"
    if args.layer == "all":
        np.savez_compressed(
            arrays_path,
            open_hidden_state=open_last,
            mcq_hidden_state=mcq_last,
            delta_hidden_state=delta_last,
            delta_hidden_state_raw=delta_raw_last,
            delta_hidden_norm=delta_norm_last,
            open_hidden_state_all_layers=open_matrix,
            mcq_hidden_state_all_layers=mcq_matrix,
            delta_hidden_state_all_layers=delta_matrix,
            delta_hidden_state_raw_all_layers=delta_raw_matrix,
            delta_hidden_norm_all_layers=delta_norm_matrix,
            layer_indices=np.asarray(layer_indices, dtype=np.int32),
        )
    else:
        np.savez_compressed(
            arrays_path,
            open_hidden_state=open_matrix,
            mcq_hidden_state=mcq_matrix,
            delta_hidden_state=delta_matrix,
            delta_hidden_state_raw=delta_raw_matrix,
            delta_hidden_norm=delta_norm_matrix,
            layer_indices=np.asarray(layer_indices, dtype=np.int32),
        )

    points_path = output_dir / "paired_hidden_states.json"
    points_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = build_summary(metadata, args, open_dir, mcq_dir)
    summary["layer_indices"] = layer_indices
    summary_path = output_dir / "paired_hidden_states_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved paired metadata to {points_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved hidden-state arrays to {arrays_path}")


if __name__ == "__main__":
    main()
