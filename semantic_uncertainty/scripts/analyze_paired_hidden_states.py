#!/usr/bin/env python3
"""Analyze paired open-text and MCQ hidden-state dumps.

This script consumes the artifacts written by plot_hidden_state_tsne_with_hidden_dump.py:
- paired_hidden_states_arrays.npz
- paired_hidden_states.json

For each provided run directory, it:
1. Selects exactly N samples per source class (default 100).
2. Builds a per-class 70/30 train/test split.
3. Trains/evaluates KNN and logistic-regression probes separately on:
   - open hidden states
   - MCQ hidden states
4. Computes within-class cosine-distance statistics on unit delta vectors.
5. Computes delta-magnitude statistics from delta norms.

Class labels are fixed to source_label:
- correct
- AH_candidate
- UH_candidate

Usage:
python semantic_uncertainty/semantic_uncertainty/scripts/analyze_paired_hidden_states.py --run_dirs /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output --arrays_file paired_hidden_states_arrays_svamp__llama__0.5_Llama-3.1-8B-Instruct.npz --metadata_file paired_hidden_states_svamp__llama__0.5_Llama-3.1-8B-Instruct.json --output_dir /home/kalashkala/paired_hidden_states_analysis_svamp__llama__0.5_Llama-3.1-8B-Instruct

nohup code:

nohup python semantic_uncertainty/semantic_uncertainty/scripts/analyze_paired_hidden_states.py --run_dirs /home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output --arrays_file paired_hidden_states_arrays_svamp__llama__0.5_Llama-3.1-8B-Instruct.npz --metadata_file paired_hidden_states_svamp__llama__0.5_Llama-3.1-8B-Instruct.json --output_dir /home/kalashkala/paired_hidden_states_analysis_svamp__llama__0.5_Llama-3.1-8B-Instruct > /home/kalashkala/paired_hidden_states_analysis_svamp__llama__0.5_Llama-3.1-8B-Instruct/output.log 2>&1 &


"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_LABELS = ["correct", "AH_candidate", "UH_candidate"]


@dataclass
class RunArtifacts:
    run_dir: Path
    arrays_path: Path
    metadata_path: Path
    arrays: Dict[str, np.ndarray]
    metadata: List[Dict[str, Any]]


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired open-text and MCQ hidden states.")
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        # default=["hidden_states_open_mcq_paired_last_layer", "hidden_states_open_mcq_paired_layer16"],
        default=["hidden_states_open_binary_mcq_paired_layer16"],
        help="One or more run directories created by the hidden-state dump script.",
    )
    parser.add_argument(
        "--arrays_file",
        type=str,
        default="paired_hidden_states_arrays.npz",
        help="Compressed NumPy archive saved by the dump script.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="paired_hidden_states.json",
        help="Metadata JSON saved by the dump script.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paired_hidden_states_analysis",
        help="Directory where analysis results will be written.",
    )
    parser.add_argument("--samples_per_class", type=int, default=100, help="Samples per source class to use.")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Per-class training fraction.")
    parser.add_argument("--knn_k", type=int, default=5, help="Number of neighbors for KNN.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_run_artifacts(run_dir: Path, arrays_file: str, metadata_file: str) -> RunArtifacts:
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

    return RunArtifacts(run_dir=run_dir, arrays_path=arrays_path, metadata_path=metadata_path, arrays=arrays, metadata=metadata)

def validate_artifacts(artifacts: RunArtifacts) -> None:
    required_arrays = {"open_hidden_state", "mcq_hidden_state", "delta_hidden_state", "delta_hidden_norm"}
    missing = required_arrays.difference(artifacts.arrays)
    if missing:
        raise ValueError(f"{artifacts.arrays_path} is missing arrays: {sorted(missing)}")

    n_rows = len(artifacts.metadata)
    for key in required_arrays:
        if artifacts.arrays[key].shape[0] != n_rows:
            raise ValueError(
                f"Row mismatch for {artifacts.run_dir}: metadata has {n_rows} rows but {key} has {artifacts.arrays[key].shape[0]}"
            )


def extract_labels(metadata: Sequence[Dict[str, Any]]) -> np.ndarray:
    labels = []
    for row in metadata:
        label = row.get("source_label")
        if label not in SOURCE_LABELS:
            raise ValueError(f"Unexpected source_label: {label}")
        labels.append(label)
    return np.asarray(labels, dtype=object)


def choose_balanced_subset(labels: np.ndarray, samples_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    label_indices = {}
    min_available = samples_per_class
    for label in SOURCE_LABELS:
        indices = np.where(labels == label)[0]
        label_indices[label] = indices
        if len(indices) < min_available:
            min_available = len(indices)
            
    actual_samples_per_class = min_available
    print(f"  [Info] Balancing classes to {actual_samples_per_class} samples each (requested max {samples_per_class})")

    chosen: List[int] = []
    for label in SOURCE_LABELS:
        indices = label_indices[label]
        picked = rng.choice(indices, size=actual_samples_per_class, replace=False)
        chosen.extend(picked.tolist())
        
    chosen = np.asarray(chosen, dtype=int)
    rng.shuffle(chosen)
    return chosen


def stratified_train_test_split(labels: np.ndarray, train_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for label in SOURCE_LABELS:
        indices = np.where(labels == label)[0]
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        train_count = int(math.floor(len(shuffled) * train_fraction))
        train_indices.extend(shuffled[:train_count].tolist())
        test_indices.extend(shuffled[train_count:].tolist())

    train_indices = np.asarray(sorted(train_indices), dtype=int)
    test_indices = np.asarray(sorted(test_indices), dtype=int)
    return train_indices, test_indices


def build_classifier_knn(k: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]
    )


def build_classifier_logreg(multiclass_mode: str) -> Pipeline:
    if multiclass_mode not in {"multinomial", "ovr"}:
        raise ValueError(f"Unsupported multiclass_mode: {multiclass_mode}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    multi_class=multiclass_mode,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = SOURCE_LABELS
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def run_probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model_kind: str, knn_k: int) -> Dict[str, Any]:
    if model_kind == "knn":
        model = build_classifier_knn(knn_k)
    elif model_kind == "logreg_multinomial":
        model = build_classifier_logreg("multinomial")
    elif model_kind == "logreg_ovr":
        model = build_classifier_logreg("ovr")
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "metrics": compute_metrics(y_test, y_pred),
        "predictions": y_pred.tolist(),
    }


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return vectors / safe


def cosine_distance_stats_by_class(delta_vectors: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    delta_norms = np.linalg.norm(delta_vectors, axis=1)
    unit_vectors = normalize_rows(delta_vectors)

    for label in SOURCE_LABELS:
        class_mask = labels == label
        class_indices = np.where(class_mask)[0]
        class_delta_norms = delta_norms[class_indices]
        class_unit_vectors = unit_vectors[class_indices]

        nonzero_mask = class_delta_norms > 0
        nonzero_vectors = class_unit_vectors[nonzero_mask]

        pairwise_values: List[float] = []
        pairwise_angles_deg: List[float] = []
        if nonzero_vectors.shape[0] >= 2:
            dist_matrix = cosine_distances(nonzero_vectors)
            upper = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
            pairwise_values = [float(x) for x in upper.tolist()]
            cosine_similarities = np.clip(1.0 - upper, -1.0, 1.0)
            pairwise_angles_deg = [float(x) for x in np.degrees(np.arccos(cosine_similarities)).tolist()]

        stats[label] = {
            "n_samples": int(class_indices.size),
            "n_zero_delta_vectors": int((~nonzero_mask).sum()),
            "mean_pairwise_cosine_distance": float(np.mean(pairwise_values)) if pairwise_values else None,
            "var_pairwise_cosine_distance": float(np.var(pairwise_values)) if pairwise_values else None,
            "mean_pairwise_angle_degrees": float(np.mean(pairwise_angles_deg)) if pairwise_angles_deg else None,
            "var_pairwise_angle_degrees": float(np.var(pairwise_angles_deg)) if pairwise_angles_deg else None,
            "min_pairwise_angle_degrees": float(np.min(pairwise_angles_deg)) if pairwise_angles_deg else None,
            "max_pairwise_angle_degrees": float(np.max(pairwise_angles_deg)) if pairwise_angles_deg else None,
            "mean_delta_norm": float(np.mean(class_delta_norms)) if class_delta_norms.size else None,
            "var_delta_norm": float(np.var(class_delta_norms)) if class_delta_norms.size else None,
            "min_delta_norm": float(np.min(class_delta_norms)) if class_delta_norms.size else None,
            "max_delta_norm": float(np.max(class_delta_norms)) if class_delta_norms.size else None,
        }

    return stats


def interclass_cosine_distance_stats(delta_vectors: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute pairwise inter-class cosine-distance and angle statistics."""
    stats: Dict[str, Any] = {}
    delta_norms = np.linalg.norm(delta_vectors, axis=1)
    unit_vectors = normalize_rows(delta_vectors)

    for i, label_a in enumerate(SOURCE_LABELS):
        for label_b in SOURCE_LABELS[i + 1 :]:
            key = f"{label_a}__vs__{label_b}"
            idx_a = np.where(labels == label_a)[0]
            idx_b = np.where(labels == label_b)[0]

            vecs_a = unit_vectors[idx_a]
            vecs_b = unit_vectors[idx_b]
            norms_a = delta_norms[idx_a]
            norms_b = delta_norms[idx_b]

            nonzero_a = vecs_a[norms_a > 0]
            nonzero_b = vecs_b[norms_b > 0]

            pairwise_values: List[float] = []
            pairwise_angles_deg: List[float] = []
            if nonzero_a.shape[0] >= 1 and nonzero_b.shape[0] >= 1:
                dist_matrix = cosine_distances(nonzero_a, nonzero_b)
                flat = dist_matrix.reshape(-1)
                pairwise_values = [float(x) for x in flat.tolist()]
                cosine_similarities = np.clip(1.0 - flat, -1.0, 1.0)
                pairwise_angles_deg = [float(x) for x in np.degrees(np.arccos(cosine_similarities)).tolist()]

            stats[key] = {
                "class_a": label_a,
                "class_b": label_b,
                "n_samples_a": int(idx_a.size),
                "n_samples_b": int(idx_b.size),
                "n_zero_delta_vectors_a": int((norms_a <= 0).sum()),
                "n_zero_delta_vectors_b": int((norms_b <= 0).sum()),
                "n_cross_pairs": int(len(pairwise_values)),
                "mean_pairwise_cosine_distance": float(np.mean(pairwise_values)) if pairwise_values else None,
                "var_pairwise_cosine_distance": float(np.var(pairwise_values)) if pairwise_values else None,
                "min_pairwise_cosine_distance": float(np.min(pairwise_values)) if pairwise_values else None,
                "max_pairwise_cosine_distance": float(np.max(pairwise_values)) if pairwise_values else None,
                "mean_pairwise_angle_degrees": float(np.mean(pairwise_angles_deg)) if pairwise_angles_deg else None,
                "var_pairwise_angle_degrees": float(np.var(pairwise_angles_deg)) if pairwise_angles_deg else None,
                "min_pairwise_angle_degrees": float(np.min(pairwise_angles_deg)) if pairwise_angles_deg else None,
                "max_pairwise_angle_degrees": float(np.max(pairwise_angles_deg)) if pairwise_angles_deg else None,
            }

    return stats


def extract_per_class_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract per-class precision, recall, F1 from classification_report."""
    per_class = {}
    report = metrics_dict.get("classification_report", {})
    for label in SOURCE_LABELS:
        if label in report:
            per_class[label] = {
                "precision": float(report[label].get("precision", 0.0)),
                "recall": float(report[label].get("recall", 0.0)),
                "f1_score": float(report[label].get("f1-score", 0.0)),
            }
    return per_class


def extract_per_class_accuracy(metrics_dict: Dict[str, Any]) -> Dict[str, float]:
    """Compute class-wise accuracy as TP/support from confusion matrix."""
    cm = np.asarray(metrics_dict.get("confusion_matrix", []), dtype=np.float64)
    if cm.shape != (len(SOURCE_LABELS), len(SOURCE_LABELS)):
        return {label: 0.0 for label in SOURCE_LABELS}

    per_class_accuracy: Dict[str, float] = {}
    for i, label in enumerate(SOURCE_LABELS):
        row_sum = float(cm[i, :].sum())
        per_class_accuracy[label] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0
    return per_class_accuracy


def save_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_run(artifacts: RunArtifacts, args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    validate_artifacts(artifacts)

    labels_all = extract_labels(artifacts.metadata)
    chosen_indices = choose_balanced_subset(labels_all, args.samples_per_class, args.seed)

    open_all = np.asarray(artifacts.arrays["open_hidden_state"], dtype=np.float32)
    mcq_all = np.asarray(artifacts.arrays["mcq_hidden_state"], dtype=np.float32)
    delta_all = np.asarray(artifacts.arrays["delta_hidden_state"], dtype=np.float32)
    delta_norm_all = np.asarray(artifacts.arrays["delta_hidden_norm"], dtype=np.float32)

    open_vectors = open_all[chosen_indices]
    mcq_vectors = mcq_all[chosen_indices]
    delta_vectors = delta_all[chosen_indices]
    delta_norms = delta_norm_all[chosen_indices]
    labels = labels_all[chosen_indices]
    metadata = [artifacts.metadata[int(i)] for i in chosen_indices]

    train_indices_rel, test_indices_rel = stratified_train_test_split(labels, args.train_fraction, args.seed)

    split_mask = np.full(labels.shape[0], "train", dtype=object)
    split_mask[test_indices_rel] = "test"

    run_out_dir = output_root / artifacts.run_dir.name
    run_out_dir.mkdir(parents=True, exist_ok=True)

    run_summary: Dict[str, Any] = {
        "run_dir": str(artifacts.run_dir),
        "arrays_path": str(artifacts.arrays_path),
        "metadata_path": str(artifacts.metadata_path),
        "samples_per_class": args.samples_per_class,
        "train_fraction": args.train_fraction,
        "knn_k": args.knn_k,
        "n_selected": int(labels.shape[0]),
        "class_counts": {label: int((labels == label).sum()) for label in SOURCE_LABELS},
        "train_counts": {label: int(((labels == label) & np.isin(np.arange(len(labels)), train_indices_rel)).sum()) for label in SOURCE_LABELS},
        "test_counts": {label: int(((labels == label) & np.isin(np.arange(len(labels)), test_indices_rel)).sum()) for label in SOURCE_LABELS},
    }

    results: Dict[str, Any] = {
        "open": {},
        "mcq": {},
        "delta": {},
        "split": {
            "train_indices": train_indices_rel.tolist(),
            "test_indices": test_indices_rel.tolist(),
        },
    }

    feature_sets = {
        "open": open_vectors,
        "mcq": mcq_vectors,
    }

    prediction_rows: List[Dict[str, Any]] = []

    for feature_name, feature_vectors in feature_sets.items():
        X_train = feature_vectors[train_indices_rel]
        X_test = feature_vectors[test_indices_rel]
        y_train = labels[train_indices_rel]
        y_test = labels[test_indices_rel]

        knn_result = run_probe(X_train, y_train, X_test, y_test, "knn", args.knn_k)
        logreg_multinomial_result = run_probe(X_train, y_train, X_test, y_test, "logreg_multinomial", args.knn_k)
        logreg_ovr_result = run_probe(X_train, y_train, X_test, y_test, "logreg_ovr", args.knn_k)

        results[feature_name] = {
            "knn": knn_result["metrics"],
            "logreg": logreg_multinomial_result["metrics"],
            "logreg_multinomial": logreg_multinomial_result["metrics"],
            "logreg_ovr": logreg_ovr_result["metrics"],
        }

        for pos, rel_idx in enumerate(test_indices_rel):
            prediction_rows.append(
                {
                    "run_dir": artifacts.run_dir.name,
                    "feature_set": feature_name,
                    "row_index": int(chosen_indices[rel_idx]),
                    "source_label": str(labels[rel_idx]),
                    "split": "test",
                    "knn_pred": knn_result["predictions"][pos],
                    "logreg_pred": logreg_multinomial_result["predictions"][pos],
                    "logreg_multinomial_pred": logreg_multinomial_result["predictions"][pos],
                    "logreg_ovr_pred": logreg_ovr_result["predictions"][pos],
                }
            )

    results["delta"] = {
        "within_class_cosine_distance": cosine_distance_stats_by_class(delta_vectors, labels),
        "inter_class_cosine_distance": interclass_cosine_distance_stats(delta_vectors, labels),
        "delta_norm_summary": {
            label: {
                "mean": float(np.mean(delta_norms[labels == label])),
                "var": float(np.var(delta_norms[labels == label])),
                "min": float(np.min(delta_norms[labels == label])),
                "max": float(np.max(delta_norms[labels == label])),
            }
            for label in SOURCE_LABELS
        },
    }

    example_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(metadata):
        example_rows.append(
            {
                "row_index": int(chosen_indices[i]),
                "source_label": row.get("source_label"),
                "source_category": row.get("source_category"),
                "split": str(split_mask[i]),
                "delta_hidden_norm": float(delta_norms[i]),
            }
        )

    examples_path = run_out_dir / "selected_examples.csv"
    save_csv(
        examples_path,
        example_rows,
        ["row_index", "source_label", "source_category", "split", "delta_hidden_norm"],
    )

    predictions_path = run_out_dir / "test_predictions.csv"
    save_csv(
        predictions_path,
        prediction_rows,
        [
            "run_dir",
            "feature_set",
            "row_index",
            "source_label",
            "split",
            "knn_pred",
            "logreg_pred",
            "logreg_multinomial_pred",
            "logreg_ovr_pred",
        ],
    )

    split_path = run_out_dir / "split_indices.json"
    split_payload = {
        "train_indices": train_indices_rel.tolist(),
        "test_indices": test_indices_rel.tolist(),
        "source_labels": labels.tolist(),
    }
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    metrics_path = run_out_dir / "analysis_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    run_summary.update(
        {
            "metrics_path": str(metrics_path),
            "split_path": str(split_path),
            "examples_path": str(examples_path),
            "predictions_path": str(predictions_path),
            "open_knn_accuracy": results["open"]["knn"]["accuracy"],
            "open_logreg_accuracy": results["open"]["logreg"]["accuracy"],
            "open_logreg_multinomial_accuracy": results["open"]["logreg_multinomial"]["accuracy"],
            "open_logreg_ovr_accuracy": results["open"]["logreg_ovr"]["accuracy"],
            "mcq_knn_accuracy": results["mcq"]["knn"]["accuracy"],
            "mcq_logreg_accuracy": results["mcq"]["logreg"]["accuracy"],
            "mcq_logreg_multinomial_accuracy": results["mcq"]["logreg_multinomial"]["accuracy"],
            "mcq_logreg_ovr_accuracy": results["mcq"]["logreg_ovr"]["accuracy"],
            "open_knn_per_class": extract_per_class_metrics(results["open"]["knn"]),
            "open_logreg_per_class": extract_per_class_metrics(results["open"]["logreg"]),
            "open_logreg_multinomial_per_class": extract_per_class_metrics(results["open"]["logreg_multinomial"]),
            "open_logreg_ovr_per_class": extract_per_class_metrics(results["open"]["logreg_ovr"]),
            "mcq_knn_per_class": extract_per_class_metrics(results["mcq"]["knn"]),
            "mcq_logreg_per_class": extract_per_class_metrics(results["mcq"]["logreg"]),
            "mcq_logreg_multinomial_per_class": extract_per_class_metrics(results["mcq"]["logreg_multinomial"]),
            "mcq_logreg_ovr_per_class": extract_per_class_metrics(results["mcq"]["logreg_ovr"]),
            "open_knn_class_accuracy": extract_per_class_accuracy(results["open"]["knn"]),
            "open_logreg_class_accuracy": extract_per_class_accuracy(results["open"]["logreg"]),
            "open_logreg_multinomial_class_accuracy": extract_per_class_accuracy(results["open"]["logreg_multinomial"]),
            "open_logreg_ovr_class_accuracy": extract_per_class_accuracy(results["open"]["logreg_ovr"]),
            "mcq_knn_class_accuracy": extract_per_class_accuracy(results["mcq"]["knn"]),
            "mcq_logreg_class_accuracy": extract_per_class_accuracy(results["mcq"]["logreg"]),
            "mcq_logreg_multinomial_class_accuracy": extract_per_class_accuracy(results["mcq"]["logreg_multinomial"]),
            "mcq_logreg_ovr_class_accuracy": extract_per_class_accuracy(results["mcq"]["logreg_ovr"]),
            "delta_within_class_cosine_distance": results["delta"]["within_class_cosine_distance"],
            "delta_inter_class_cosine_distance": results["delta"]["inter_class_cosine_distance"],
            "delta_norm_summary": results["delta"]["delta_norm_summary"],
        }
    )

    summary_path = run_out_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    run_summary["summary_path"] = str(summary_path)

    return {
        "summary": run_summary,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, Any] = {
        "run_dirs": [str(Path(p)) for p in args.run_dirs],
        "samples_per_class": args.samples_per_class,
        "train_fraction": args.train_fraction,
        "knn_k": args.knn_k,
        "seed": args.seed,
        "runs": [],
    }

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        artifacts = load_run_artifacts(run_dir, args.arrays_file, args.metadata_file)
        analysis = analyze_run(artifacts, args, output_root)
        combined["runs"].append(analysis["summary"])
        print(f"Analyzed {run_dir}")
        print(f"  Overall Accuracies:")
        print(f"    open knn: {analysis['summary']['open_knn_accuracy']:.4f}")
        print(f"    open logreg: {analysis['summary']['open_logreg_accuracy']:.4f}")
        print(f"    open logreg multinomial (direct 3-class): {analysis['summary']['open_logreg_multinomial_accuracy']:.4f}")
        print(f"    open logreg ovr: {analysis['summary']['open_logreg_ovr_accuracy']:.4f}")
        print(f"    mcq knn: {analysis['summary']['mcq_knn_accuracy']:.4f}")
        print(f"    mcq logreg: {analysis['summary']['mcq_logreg_accuracy']:.4f}")
        print(f"    mcq logreg multinomial (direct 3-class): {analysis['summary']['mcq_logreg_multinomial_accuracy']:.4f}")
        print(f"    mcq logreg ovr: {analysis['summary']['mcq_logreg_ovr_accuracy']:.4f}")
        print(f"  Per-Class Accuracy (TP/support):")
        for model_key in [
            "open_knn_class_accuracy",
            "open_logreg_class_accuracy",
            "open_logreg_multinomial_class_accuracy",
            "open_logreg_ovr_class_accuracy",
            "mcq_knn_class_accuracy",
            "mcq_logreg_class_accuracy",
            "mcq_logreg_multinomial_class_accuracy",
            "mcq_logreg_ovr_class_accuracy",
        ]:
            print(f"    {model_key}:")
            for label in SOURCE_LABELS:
                print(f"      {label}: {analysis['summary'][model_key][label]:.4f}")
        print(f"  Per-Class Metrics (Precision / Recall / F1):")
        print(f"    open_knn:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['open_knn_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"    open_logreg:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['open_logreg_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"    open_logreg_multinomial:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['open_logreg_multinomial_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"    open_logreg_ovr:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['open_logreg_ovr_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"    mcq_knn:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['mcq_knn_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"    mcq_logreg:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['mcq_logreg_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"    mcq_logreg_multinomial:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['mcq_logreg_multinomial_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"    mcq_logreg_ovr:")
        for label in SOURCE_LABELS:
            pc = analysis['summary']['mcq_logreg_ovr_per_class'][label]
            print(f"      {label}: {pc['precision']:.4f} / {pc['recall']:.4f} / {pc['f1_score']:.4f}")
        print(f"  Delta Vector Analysis:")
        for label in SOURCE_LABELS:
            delta_stats = analysis["results"]["delta"]["within_class_cosine_distance"][label]
            norm_stats = analysis["results"]["delta"]["delta_norm_summary"][label]
            print(
                f"    {label} delta cosine distance mean/var: "
                f"{delta_stats['mean_pairwise_cosine_distance']:.4f} / {delta_stats['var_pairwise_cosine_distance']:.4f}"
            )
            print(
                f"    {label} delta angle (deg) mean/var [min,max]: "
                f"{delta_stats['mean_pairwise_angle_degrees']:.4f} / {delta_stats['var_pairwise_angle_degrees']:.4f} "
                f"[{delta_stats['min_pairwise_angle_degrees']:.4f}, {delta_stats['max_pairwise_angle_degrees']:.4f}]"
            )
            print(
                f"    {label} delta norm mean/var: "
                f"{norm_stats['mean']:.4f} / {norm_stats['var']:.4f}"
            )
        print(f"  Delta Inter-Class Geometry:")
        for key, pair_stats in analysis["results"]["delta"]["inter_class_cosine_distance"].items():
            print(
                f"    {key} cosine distance mean/var: "
                f"{pair_stats['mean_pairwise_cosine_distance']:.4f} / {pair_stats['var_pairwise_cosine_distance']:.4f}"
            )
            print(
                f"    {key} angle (deg) mean/var [min,max]: "
                f"{pair_stats['mean_pairwise_angle_degrees']:.4f} / {pair_stats['var_pairwise_angle_degrees']:.4f} "
                f"[{pair_stats['min_pairwise_angle_degrees']:.4f}, {pair_stats['max_pairwise_angle_degrees']:.4f}]"
            )

    combined_path = output_root / "combined_analysis_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Saved combined summary to {combined_path}")


if __name__ == "__main__":
    main()