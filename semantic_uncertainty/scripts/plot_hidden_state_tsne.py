#!/usr/bin/env python3
"""Plot tSNE of final prompt-token hidden states.

Expected input per example:
- prompt (required)
- true_answer (required)
- subject (required by the user workflow, optional in script output)
- semantic_entropy (optional, configurable field name)

Workflow:
1) Build chat input with user prompt + assistant response and run with output_hidden_states=True.
2) Extract hidden state of the final token from selected layer.
3) Generate natural answer from the same prompt.
4) Compare prediction to true_answer and assign one label:
   - correct
   - AH_candidate (wrong and semantic_entropy < threshold)
   - UH_candidate (all other wrong predictions)
5) Stack one vector per example.
6) tSNE -> 2 dims.
7) Save plot, coordinates, and summary statistics.

ablations:
more models
different se thresholds
different layers
better correctness criteria 
different hidden state extraction (e.g. final prompt token instead of final response token, or average over last N tokens, etc.)

both mcq and open text 
1. 3 class knn classifier on hidden states - last layer and 16th layer, 70-30 split for each class with 100 samples
2. logistic regression probes (LLMKnows style) on hidden states - last layer and 16th layer, 70-30 split for each class with 100 samples
3. avg and variance cosine distances between unit vector(delta H(hidden state for mcq - hidden state for open text))(to focus on direction) of each sample within same class
3a. keep storing magitude of delta H

steering vector
2. learable steering vector - learn log reg from individual hidden state and corresponding delta H and apply this log reg to test set hidden states and 
2a. 100 correct and AH hidden states from layer 16(70-30 split)
2b. learn logreg from the 140 samples hidden states to predict delta H
2c. test set - logreg on layer 16 hidden state gives delta H and added for greedy and sampling decoding and then see if it improves accuracy and semantic entropy
save everything

1. take avg delta H (mag and direction both) from train set for layer 16 and add to test set decoded hidden states and then take it forward
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate import load
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_uncertainty.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)


LayerSpec = Union[str, int]

CHAT_MODEL_KEYWORDS = ("llama", "qwen", "mistral")

GOOD_SHOTS = [
    "question: What is the capital of France?\nanswer: Paris\n",
    "question: How many continents are there?\nanswer: 7\n",
    "question: Who wrote 'Romeo and Juliet'?\nanswer: William Shakespeare\n",
    "question: What is the square root of 64?\nanswer: 8\n",
    "question: Which element has the chemical symbol 'H'?\nanswer: Hydrogen\n",
    "question: Who was the first President of the United States?\nanswer: George Washington\n",
    "question: What is the powerhouse of the cell?\nanswer: Mitochondria\n",
    "question: In what year did World War II end?\nanswer: 1945\n",
    "question: What is the currency of Japan?\nanswer: Japanese Yen\n",
    "question: Who painted the Mona Lisa?\nanswer: Leonardo da Vinci\n",
    "question: What is the speed of light?\nanswer: 299,792 kilometers per second\n",
    "question: How many sides does a hexagon have?\nanswer: 6\n",
    "question: What is the boiling point of water in Celsius?\nanswer: 100 degrees\n",
    "question: Who wrote 'To Kill a Mockingbird'?\nanswer: Harper Lee\n",
    "question: What is the capital of Australia?\nanswer: Canberra\n",
    "question: What is the largest ocean on Earth?\nanswer: Pacific Ocean\n",
    "question: Who discovered penicillin?\nanswer: Alexander Fleming\n",
    "question: What is the chemical symbol for gold?\nanswer: Au\n",
    "question: What is the smallest prime number?\nanswer: 2\n",
    "question: How many planets are there in our solar system?\nanswer: 8\n",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("data", "examples", "items", "records", "failures"):
                value = obj.get(key)
                if isinstance(value, list):
                    return value
            return [obj]
    except json.JSONDecodeError:
        pass

    rows: List[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    return rows


def extract_last_question_prompt(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = re.findall(r"question:\s*(.*?)\nanswer:", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    q = matches[-1].strip()
    if not q:
        return None
    return f"question: {q}\nanswer:"


def build_prompt_from_method(question_prompt: str, method: str, rng: random.Random) -> str:
    shot = GOOD_SHOTS[rng.randrange(len(GOOD_SHOTS))]
    if method == "prompt_8":
        return "Answer the following question as briefly as possible.\n" + question_prompt
    if method == "prompt_4":
        prefix = (
            "I'm working on a big school project, and there's a lot of information I'm supposed to know. "
            "But some of it is kind of hard to understand, and I'm not sure where to start. "
            "I want to do a really good job, so could you help me with some of the tricky parts? "
            "It would mean a lot!"
        )
        return prefix + shot + question_prompt
    if method == "none":
        return question_prompt
    # Fallback: conservative prompt_8-style if unknown method name.
    return shot + "Answer the following question as briefly as possible.\n" + question_prompt


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


def encode_prompt_for_model(
    prompt: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.Tensor:
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


def normalize_record(raw: Any, idx: int) -> Optional[Dict[str, Any]]:
    """Normalize supported input schemas into a common dict format.

    Supported schemas:
    1) Dict records with prompt/true_answer/subject.
    2) Formatted list records:
       [prompt, true_answer, token_ids, subject_code]
    """
    if isinstance(raw, dict):
        prompt = raw.get("prompt")
        true_answer = raw.get("true_answer")
        subject = raw.get("subject")
        if not isinstance(true_answer, str):
            return None
        question_prompt = None
        if isinstance(prompt, str):
            question_prompt = extract_last_question_prompt(prompt)
        if question_prompt is None and isinstance(raw.get("question"), str):
            question_prompt = f"question: {raw['question'].strip()}\nanswer:"
        if question_prompt is None:
            return None
        out = dict(raw)
        out.setdefault("id", idx)
        out.setdefault("subject", subject if subject is not None else "unknown")
        out["question_prompt"] = question_prompt
        return out

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        prompt = raw[0]
        true_answer = raw[1]
        if not isinstance(prompt, str) or not isinstance(true_answer, str):
            return None

        subject = raw[3] if len(raw) >= 4 else "unknown"
        token_ids = raw[2] if len(raw) >= 3 else None
        question_prompt = extract_last_question_prompt(prompt)
        if question_prompt is None:
            return None
        return {
            "id": idx,
            "prompt": prompt,
            "question_prompt": question_prompt,
            "true_answer": true_answer,
            "subject": subject,
            "answer_token_ids": token_ids,
        }

    return None


def clean_answer(text: str) -> str:
    x = str(text or "").strip()
    x = re.sub(r"^\s*(?:<\|im_start\|>\s*)?(?:system|user|assistant)\s*(?:<\|im_end\|>)?\s*", "", x, flags=re.IGNORECASE)
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
    """Use relaxed matching similar to the main uncertainty pipeline."""
    p = normalize_answer(predicted)
    t = normalize_answer(true_answer)
    if not p or not t:
        return False
    return p == t or p in t or t in p


def is_correct_squad(squad_metric: Any, predicted: str, true_answer: str, exid: Union[str, int]) -> bool:
    """Match correctness logic used in utils.py: SQuAD F1 >= 50."""
    prediction = {
        "prediction_text": str(predicted),
        "no_answer_probability": 0.0,
        "id": str(exid),
    }
    reference = {
        "answers": {
            "answer_start": [0],
            "text": [str(true_answer)],
        },
        "id": str(exid),
    }
    results = squad_metric.compute(predictions=[prediction], references=[reference])
    return float(results["f1"]) >= 50.0


def parse_layer_spec(layer_arg: str) -> LayerSpec:
    if layer_arg in {"last", "all"}:
        return layer_arg
    try:
        return int(layer_arg)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid layer value: {layer_arg}") from exc


def select_layer_index(hidden_states: Tuple[torch.Tensor, ...], layer: LayerSpec) -> int:
    if isinstance(layer, int):
        return layer if layer >= 0 else len(hidden_states) + layer
    if layer == "last":
        return len(hidden_states) - 1
    raise ValueError(f"Unsupported layer spec: {layer}")


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    kwargs: Dict[str, Any] = {}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def get_input_ids_from_prompt(
    prompt: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Build model input ids similar to uncertainty_calculation.get_input_ids_from_prompt."""
    if is_chat_model_name(model_name) and getattr(tokenizer, "chat_template", None):
        return encode_prompt_for_model(prompt, model_name, tokenizer, device)

    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def compute_single_response_stats(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Compute single-response stats aligned with uncertainty_calculation outputs."""
    input_ids = get_input_ids_from_prompt(prompt, model_name, tokenizer, device)

    special_tokens = {
        "<|assistant|>", "<|user|>", "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "<|start|>",
        "<|end|>", "<|sep|>", "<|sep_id|>", "assistant", "user", "\n", "answer", "The", "Answer",
        '"', "'", " answer", "is", "it", "it's", ":", " ", " is", " correct", "correct", "*", "**",
        " **",
    }

    probs_list: List[float] = []
    entropy_list: List[float] = []
    prob_delta_list: List[float] = []
    next_tokens: List[str] = []
    most_likely_tokens: List[Tuple[str, float]] = []
    answer_index = 0

    with torch.no_grad():
        work_ids = input_ids
        for _ in range(max_new_tokens):
            outputs = model(work_ids, return_dict=True)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)
            next_token_prob = float(probs[0, next_token_id].item())
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            top_k_probs = sorted_probs[0, :5]
            top_k_indices = sorted_indices[0, :5]
            most_likely_tokens = [
                (tokenizer.decode(top_k_indices[i].item()), float(top_k_probs[i].item()))
                for i in range(5)
            ]

            token_text = tokenizer.decode(next_token_id)
            prob_diff = float(probs[0, next_token_id].item() - sorted_probs[0, 1].item())
            entropy = float((-torch.sum(probs * torch.log(probs), dim=-1)).item())

            probs_list.append(next_token_prob)
            entropy_list.append(entropy)
            prob_delta_list.append(prob_diff)
            next_tokens.append(token_text)
            work_ids = torch.cat([work_ids, next_token_id.unsqueeze(0)], dim=-1)

            if token_text.strip() and token_text.strip() not in special_tokens:
                answer_index = len(next_tokens) - 1
                break

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                top_p=None,
                temperature=None,
                attention_mask=torch.ones_like(input_ids),
            )

    generated_full = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    generated_only = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    if not probs_list:
        return {
            "prob": None,
            "prob_diff": None,
            "mean_entropy": None,
            "most_likely_tokens": [],
            "generated": generated_full,
            "generated_answer_only": clean_answer(generated_only),
        }

    return {
        "prob": probs_list[answer_index],
        "prob_diff": prob_delta_list[answer_index],
        "mean_entropy": entropy_list[answer_index],
        "most_likely_tokens": most_likely_tokens,
        "generated": generated_full,
        "generated_answer_only": clean_answer(generated_only),
    }


def sample_answer_with_loglik(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Optional[str], Optional[List[float]]]:
    """Sample one answer and return (clean_text, token_log_likelihoods)."""
    input_ids = get_input_ids_from_prompt(prompt, model_name, tokenizer, device)
    do_sample = True

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=temperature,
            do_sample=do_sample,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(input_ids),
        )

    generated_ids = outputs.sequences[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    generated_text = clean_answer(generated_text)

    if len(outputs.scores) == 0:
        return generated_text, None

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True,
    )
    token_log_likelihoods = [float(x) for x in transition_scores[0].tolist()]
    return generated_text, token_log_likelihoods


def compute_semantic_entropy_with_loaded_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    device: torch.device,
    entailment_model: EntailmentDeberta,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Optional[float], Optional[float], List[str]]:
    """Compute semantic entropy using the already-loaded generation model.

    Mirrors the pipeline structure: 1 low-temp sample + 10 high-temp samples,
    then computes entropy over high-temp samples.
    """
    responses: List[str] = []
    log_liks: List[List[float]] = []

    # First generation at low temperature (pipeline behavior), discarded for SE.
    _ = sample_answer_with_loglik(
        model,
        tokenizer,
        model_name,
        prompt,
        device,
        max_new_tokens,
        temperature=0.1,
    )

    for _ in range(10):
        ans, lls = sample_answer_with_loglik(
            model,
            tokenizer,
            model_name,
            prompt,
            device,
            max_new_tokens,
            temperature=temperature,
        )
        if ans is None or lls is None or len(lls) == 0:
            continue
        responses.append(ans)
        log_liks.append(lls)

    if not responses or not log_liks:
        return None, None, responses

    # Match calc_semantic_entropy behavior by prefixing question/context text.
    prefixed = [f"{prompt} {r}" for r in responses]
    semantic_ids = get_semantic_ids(prefixed, model=entailment_model, strict_entailment=False, example={"question": prompt})
    avg_token_log_liks = [float(np.mean(x)) for x in log_liks]
    regular_entropy = float(predictive_entropy(avg_token_log_liks))
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, avg_token_log_liks, agg="sum_normalized")
    semantic_entropy = float(predictive_entropy_rao(log_likelihood_per_semantic_id))
    return semantic_entropy, regular_entropy, responses


def prepare_examples(records: Sequence[Dict[str, Any]], max_examples: int, seed: int) -> List[Dict[str, Any]]:
    usable: List[Dict[str, Any]] = []
    for i, raw_row in enumerate(records):
        row = normalize_record(raw_row, i)
        if row is None:
            continue
        usable.append(row)

    if max_examples > 0 and len(usable) > max_examples:
        rng = random.Random(seed)
        usable = rng.sample(usable, max_examples)
    return usable


def extract_final_response_token_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    response: str,
    layer: LayerSpec,
    device: torch.device,
) -> np.ndarray:
    # Match probe pipeline: hidden states from chat(user=prompt, assistant=response).
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
    else:
        input_ids = tokenizer(f"{prompt} {response}", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    layer_idx = select_layer_index(hidden_states, layer)
    if layer_idx < 0 or layer_idx >= len(hidden_states):
        raise IndexError(f"Layer index {layer_idx} out of range for {len(hidden_states)} hidden-state tensors")

    # Final sequence token hidden state, aligned with probe token=-1 behavior.
    vec = hidden_states[layer_idx][0, -1, :].detach().cpu().numpy().astype(np.float32)
    return vec


def generate_natural_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    model_name: str = "",
) -> str:
    inputs = get_input_ids_from_prompt(prompt, model_name, tokenizer, device)
    prompt_len = inputs.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return clean_answer(text)


def assign_label(is_correct: bool, semantic_entropy: Optional[float], threshold: float) -> str:
    if is_correct:
        return "correct"
    if semantic_entropy is not None and semantic_entropy < threshold:
        return "AH_candidate"
    return "UH_candidate"


def reduce_to_2d(vectors: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    if vectors.shape[0] < 2:
        raise ValueError("Need at least 2 vectors for tSNE plotting")

    n_samples = vectors.shape[0]
    max_valid_perplexity = max(1.0, float(n_samples) - 1e-3)
    effective_perplexity = min(perplexity, max_valid_perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="random",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(vectors)


def plot_tsne(
    points: np.ndarray,
    metadata: Sequence[Dict[str, Any]],
    output_path: Path,
    *,
    label_key: str = "label",
    title: str = "tSNE of final response-token hidden states",
) -> None:
    colors = {
        "correct": "#1f77b4",
        "AH_candidate": "#d62728",
        "UH_candidate": "#2ca02c",
    }
    markers = {
        "correct": "o",
        "AH_candidate": "^",
        "UH_candidate": "s",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    seen: set[str] = set()

    for i, row in enumerate(metadata):
        label = str(row.get(label_key, "UH_candidate"))
        ax.scatter(
            points[i, 0],
            points[i, 1],
            s=44,
            c=colors[label],
            marker=markers[label],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            label=label if label not in seen else None,
            zorder=2,
        )
        seen.add(label)

    ax.set_title(title)
    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_paired_hidden_state_artifacts(
    run_dir: Path,
    arrays_file: str,
    metadata_file: str,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    arrays_path = run_dir / arrays_file
    metadata_path = run_dir / metadata_file
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing arrays file: {arrays_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    loaded = np.load(arrays_path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError(f"Expected metadata list in {metadata_path}")
    return arrays, metadata


def select_layer_indices_for_array(layer: LayerSpec, n_layers: int) -> List[int]:
    if layer == "all":
        return list(range(n_layers))
    if isinstance(layer, int):
        idx = layer if layer >= 0 else n_layers + layer
        if idx < 0 or idx >= n_layers:
            raise IndexError(f"Layer index {idx} out of range for {n_layers} layers")
        return [idx]
    if layer == "last":
        return [n_layers - 1]
    raise ValueError(f"Unsupported layer spec: {layer}")


def get_layer_stack(arrays: Dict[str, np.ndarray], key_candidates: Sequence[str]) -> np.ndarray:
    for key in key_candidates:
        if key in arrays:
            return np.asarray(arrays[key], dtype=np.float32)
    raise ValueError(f"Missing any of arrays: {list(key_candidates)}")


def run_paired_hidden_state_tsne(
    run_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    arrays, metadata = load_paired_hidden_state_artifacts(run_dir, args.paired_arrays_file, args.paired_metadata_file)

    open_stack = get_layer_stack(arrays, ["open_hidden_state_all_layers", "open_hidden_state"])
    mcq_stack = get_layer_stack(arrays, ["mcq_hidden_state_all_layers", "mcq_hidden_state"])

    if open_stack.shape[0] != mcq_stack.shape[0]:
        raise ValueError(f"Open/MCQ row mismatch: {open_stack.shape[0]} vs {mcq_stack.shape[0]}")
    if len(metadata) != open_stack.shape[0]:
        raise ValueError(f"Metadata row mismatch: {len(metadata)} vs {open_stack.shape[0]}")

    if open_stack.ndim == 2:
        open_stack = open_stack[:, None, :]
    if mcq_stack.ndim == 2:
        mcq_stack = mcq_stack[:, None, :]

    if open_stack.ndim != 3 or mcq_stack.ndim != 3:
        raise ValueError("Expected open and mcq hidden-state arrays to be rank-2 or rank-3")
    if open_stack.shape[1] != mcq_stack.shape[1]:
        raise ValueError(f"Open/MCQ layer mismatch: {open_stack.shape[1]} vs {mcq_stack.shape[1]}")

    layer_indices = select_layer_indices_for_array(args.paired_layer, open_stack.shape[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "arrays_file": args.paired_arrays_file,
        "metadata_file": args.paired_metadata_file,
        "layer": args.paired_layer,
        "layer_indices": layer_indices,
        "n_examples": len(metadata),
        "plots": [],
    }

    for layer_idx in layer_indices:
        layer_dir = output_dir / f"layer_{layer_idx:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        open_vectors = open_stack[:, layer_idx, :]
        mcq_vectors = mcq_stack[:, layer_idx, :]

        open_points = reduce_to_2d(open_vectors, args.seed, args.perplexity)
        mcq_points = reduce_to_2d(mcq_vectors, args.seed, args.perplexity)

        open_plot_path = layer_dir / "open_hidden_state_tsne.png"
        mcq_plot_path = layer_dir / "mcq_hidden_state_tsne.png"

        open_metadata = [{"label": str(row.get("source_label", "UH_candidate"))} for row in metadata]
        mcq_metadata = [{"label": str(row.get("source_label", "UH_candidate"))} for row in metadata]

        plot_tsne(
            open_points,
            open_metadata,
            open_plot_path,
            label_key="label",
            title=f"Open hidden states tSNE (layer {layer_idx})",
        )
        plot_tsne(
            mcq_points,
            mcq_metadata,
            mcq_plot_path,
            label_key="label",
            title=f"MCQ hidden states tSNE (layer {layer_idx})",
        )

        open_points_path = layer_dir / "open_hidden_state_tsne_points.json"
        mcq_points_path = layer_dir / "mcq_hidden_state_tsne_points.json"
        open_point_rows: List[Dict[str, Any]] = []
        mcq_point_rows: List[Dict[str, Any]] = []
        for point, row in zip(open_points, metadata):
            open_point_rows.append({**row, "tsne_x": float(point[0]), "tsne_y": float(point[1]), "layer_index": layer_idx, "view": "open"})
        for point, row in zip(mcq_points, metadata):
            mcq_point_rows.append({**row, "tsne_x": float(point[0]), "tsne_y": float(point[1]), "layer_index": layer_idx, "view": "mcq"})

        open_points_path.write_text(json.dumps(open_point_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        mcq_points_path.write_text(json.dumps(mcq_point_rows, indent=2, ensure_ascii=False), encoding="utf-8")

        layer_summary = {
            "layer_index": layer_idx,
            "open_plot": str(open_plot_path),
            "mcq_plot": str(mcq_plot_path),
            "open_points": str(open_points_path),
            "mcq_points": str(mcq_points_path),
            "n_examples": len(metadata),
        }
        (layer_dir / "layer_summary.json").write_text(json.dumps(layer_summary, indent=2), encoding="utf-8")
        summary["plots"].append(layer_summary)

    (output_dir / "paired_hidden_state_tsne_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved paired tSNE plots to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tSNE of final prompt-token hidden states.")
    parser.add_argument("--input_file", type=str, help="JSON or JSONL file for the original single-prompt mode")
    parser.add_argument(
        "--paired_run_dir",
        type=str,
        help="Run directory containing paired_hidden_states_arrays.npz and paired_hidden_states.json",
    )
    parser.add_argument(
        "--paired_arrays_file",
        type=str,
        default="paired_hidden_states_arrays.npz",
        help="NumPy archive written by the paired hidden-state dump script",
    )
    parser.add_argument(
        "--paired_metadata_file",
        type=str,
        default="paired_hidden_states.json",
        help="Metadata JSON written by the paired hidden-state dump script",
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", required=False, help="Hugging Face causal LM name or local path")
    parser.add_argument("--output_dir", type=str, default="hidden_states_tsne_layer16_0.3", help="Output directory")
    parser.add_argument("--layer", type=parse_layer_spec, default="16", help="Hidden-state layer index or 'last'")
    parser.add_argument("--paired_layer", type=parse_layer_spec, default="all", help="Layer index, 'last', or 'all' for paired plots")
    parser.add_argument("--max_examples", type=int, default=1000, help="0 means all usable examples")
    parser.add_argument("--perplexity", type=float, default=50.0, help="tSNE perplexity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens for natural answer generation")
    parser.add_argument("--semantic_entropy_field", type=str, default="semantic_entropy", help="Field name for semantic entropy")
    parser.add_argument("--ah_entropy_threshold", type=float, default=0.3, help="Threshold for AH_candidate label")
    parser.add_argument(
        "--compute_semantic_entropy_if_missing",
        # action="store_true",
        default=True,
        help="If semantic entropy is missing in input, compute it via sampling (same module used by uncertainty pipeline)",
    )
    parser.add_argument(
        "--semantic_entropy_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used for semantic entropy computation",
    )
    parser.add_argument(
        "--method_k_positive",
        type=str,
        default="prompt_8",
        choices=["prompt_8", "prompt_4", "none"],
        help="Prompt construction method for inference when input has only question-style rows",
    )
    return parser.parse_args()


def build_summary(metadata: Sequence[Dict[str, Any]], args: argparse.Namespace, input_path: Path) -> Dict[str, Any]:
    counts = {"correct": 0, "AH_candidate": 0, "UH_candidate": 0}
    for row in metadata:
        counts[row["label"]] += 1

    entropy_values = [row["semantic_entropy"] for row in metadata if row["semantic_entropy"] is not None]

    return {
        "input_file": str(input_path),
        "model_name": args.model_name,
        "n_examples": len(metadata),
        "layer": args.layer,
        "semantic_entropy_field": args.semantic_entropy_field,
        "ah_entropy_threshold": args.ah_entropy_threshold,
        "counts": counts,
        "mean_semantic_entropy": float(np.mean(entropy_values)) if entropy_values else None,
        "median_semantic_entropy": float(np.median(entropy_values)) if entropy_values else None,
    }


def save_metadata_checkpoint(rows: Sequence[Dict[str, Any]], checkpoint_path: Path) -> None:
    """Persist per-example metadata after each processed example."""
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.paired_run_dir:
        run_paired_hidden_state_tsne(Path(args.paired_run_dir), Path(args.output_dir), args)
        return

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / f"{input_path.stem}_prompt_tsne"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_records(input_path)
    records = prepare_examples(raw_records, args.max_examples, args.seed)
    if not records:
        raise ValueError(f"No usable examples found in {input_path}")

    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)
    squad_metric = load("squad_v2")

    entailment_model: Optional[EntailmentDeberta] = None
    if args.compute_semantic_entropy_if_missing:
        # Load only entailment model; reuse already-loaded generation model for sampling.
        entailment_model = EntailmentDeberta()

    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    prompt_rng = random.Random(args.seed)
    checkpoint_path = output_dir / "prompt_hidden_state_tsne_points_partial.json"

    for idx, row in enumerate(records):
        question_prompt = row["question_prompt"]
        inference_prompt = build_prompt_from_method(question_prompt, args.method_k_positive, prompt_rng)
        true_answer = row["true_answer"]
        subject = row.get("subject")

        entropy_raw = row.get(args.semantic_entropy_field)
        semantic_entropy: Optional[float]
        try:
            semantic_entropy = float(entropy_raw) if entropy_raw is not None else None
        except (TypeError, ValueError):
            semantic_entropy = None

        mean_entropy_temp: Optional[float] = None
        temp_generations: List[str] = []
        if semantic_entropy is None and entailment_model is not None:
            try:
                semantic_entropy, mean_entropy_temp, temp_generations = compute_semantic_entropy_with_loaded_model(
                    model,
                    tokenizer,
                    args.model_name,
                    inference_prompt,
                    device,
                    entailment_model,
                    args.max_new_tokens,
                    args.semantic_entropy_temperature,
                )
            except Exception as exc:
                print(f"Warning: semantic entropy sampling failed for example {idx}: {exc}")

        try:
            single_stats = compute_single_response_stats(
                model,
                tokenizer,
                args.model_name,
                inference_prompt,
                device,
                args.max_new_tokens,
            )
            predicted = single_stats["generated_answer_only"]
            vec = extract_final_response_token_hidden(
                model,
                tokenizer,
                args.model_name,
                inference_prompt,
                predicted,
                args.layer,
                device,
            )
        except Exception as exc:
            print(f"Skipping example {idx} due to model error: {exc}")
            continue

        try:
            is_correct = is_correct_squad(squad_metric, predicted, true_answer, row.get("id", idx))
        except Exception as exc:
            print(f"Warning: SQuAD metric failed for example {idx}, using string-match fallback: {exc}")
            is_correct = answers_match(predicted, true_answer)
        label = assign_label(is_correct, semantic_entropy, args.ah_entropy_threshold)

        vectors.append(vec)
        metadata.append(
            {
                "id": row.get("id", idx),
                "subject": subject,
                "prompt": inference_prompt,
                "question_prompt": question_prompt,
                "true_answer": true_answer,
                "generated": single_stats["generated"],
                "predicted_answer": predicted,
                "prob": single_stats["prob"],
                "prob_diff": single_stats["prob_diff"],
                "mean_entropy": single_stats["mean_entropy"],
                "most_likely_tokens": single_stats["most_likely_tokens"],
                "semantic_entropy": semantic_entropy,
                "mean_entropy_temp_sampling": mean_entropy_temp,
                "temp_generations": temp_generations,
                "label": label,
            }
        )

        # Save progress after each successful example so partial work is never lost.
        save_metadata_checkpoint(metadata, checkpoint_path)

    if not vectors:
        raise ValueError("No vectors extracted. Check model access and input format.")

    matrix = np.vstack(vectors)
    points = reduce_to_2d(matrix, args.seed, args.perplexity)

    plot_path = output_dir / "prompt_hidden_state_tsne.png"
    plot_tsne(points, metadata, plot_path)

    point_rows: List[Dict[str, Any]] = []
    for point, row in zip(points, metadata):
        point_rows.append(
            {
                **row,
                "tsne_x": float(point[0]),
                "tsne_y": float(point[1]),
            }
        )

    points_path = output_dir / "prompt_hidden_state_tsne_points.json"
    with points_path.open("w", encoding="utf-8") as f:
        json.dump(point_rows, f, indent=2, ensure_ascii=False)

    summary = build_summary(metadata, args, input_path)
    summary_path = output_dir / "prompt_hidden_state_tsne_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved plot to {plot_path}")
    print(f"Saved point data to {points_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()