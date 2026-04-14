#!/usr/bin/env python3
"""Extract and process hidden states from npz file using tSNE visualization."""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Sequence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def reduce_to_2d(vectors: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    """Reduce high-dimensional vectors to 2D using tSNE."""
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
    """Create and save a tSNE plot."""
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
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and process hidden states from npz file using tSNE visualization."
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="",
        help="Custom file name extension/suffix to distinguish between runs (e.g., '_run1', '_v2'). Default is empty string.",
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default="/home/kalashkala/Datasets/Semantic-Uncertainty/hidden_states_output/paired_hidden_states_arrays_svamp__llama__0.5_Llama-3.1-8B-Instruct.npz",
        help="Path to the npz file containing hidden states.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/kalashkala/tsne_output",
        help="Output directory for saving plots and coordinates.",
    )

    args = parser.parse_args()
    npz_path = Path(args.npz_path)
    output_dir = Path(args.output_dir)
    file_extension = args.extension

    print(f"Loading {npz_path.name}...")
    loaded = np.load(npz_path, allow_pickle=False)

    print("\nAvailable arrays:")
    for key in loaded.files:
        print(f"  {key}: shape={loaded[key].shape}, dtype={loaded[key].dtype}")

    # Create output directory
    output_dir = Path("/home/kalashkala/tsne_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract hidden states
    open_vectors = np.asarray(loaded["open_hidden_state"], dtype=np.float32)
    mcq_vectors = np.asarray(loaded["mcq_hidden_state"], dtype=np.float32)
    delta_vectors = np.asarray(loaded["delta_hidden_state"], dtype=np.float32)

    print(f"\nProcessing vectors...")
    print(f"  Open vectors: {open_vectors.shape}")
    print(f"  MCQ vectors: {mcq_vectors.shape}")
    print(f"  Delta vectors: {delta_vectors.shape}")

    # Load actual labels from the paired metadata JSON
    metadata_path = npz_path.parent / npz_path.name.replace("paired_hidden_states_arrays_", "paired_hidden_states_").replace(".npz", ".json")
    print(f"\nLoading metadata from {metadata_path.name}...")
    metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_samples = open_vectors.shape[0]
    if len(metadata_raw) != n_samples:
        raise ValueError(f"Metadata length {len(metadata_raw)} doesn't match array rows {n_samples}")
    metadata = [{"label": str(row.get("source_label", "UH_candidate")), "index": i} for i, row in enumerate(metadata_raw)]

    # Reduce to 2D and plot
    seed = 42
    perplexity = 30.0

    print(f"\nComputing tSNE (seed={seed}, perplexity={perplexity})...")

    open_points = reduce_to_2d(open_vectors, seed, perplexity)
    print(f"Open tSNE points: {open_points.shape}")
    plot_tsne(open_points, metadata, output_dir / f"open_hidden_state_tsne_{file_extension}.png",
              title="tSNE of Open-ended Hidden States")

    mcq_points = reduce_to_2d(mcq_vectors, seed, perplexity)
    print(f"MCQ tSNE points: {mcq_points.shape}")
    plot_tsne(mcq_points, metadata, output_dir / f"mcq_hidden_state_tsne_{file_extension}.png",
              title="tSNE of MCQ Hidden States")

    delta_points = reduce_to_2d(delta_vectors, seed, perplexity)
    print(f"Delta tSNE points: {delta_points.shape}")
    plot_tsne(delta_points, metadata, output_dir / f"delta_hidden_state_tsne_{file_extension}.png",
              title="tSNE of Delta Hidden States")

    # Save point coordinates as JSON
    print(f"\nSaving point coordinates...")
    for name, points in [("open", open_points), ("mcq", mcq_points), ("delta", delta_points)]:
        points_data = []
        for i, point in enumerate(points):
            points_data.append({
                "index": i,
                "tsne_x": float(point[0]),
                "tsne_y": float(point[1]),
            })
        points_path = output_dir / f"{name}_hidden_state_tsne_points_{file_extension}.json"
        points_path.write_text(json.dumps(points_data, indent=2), encoding="utf-8")
        print(f"  Saved {points_path.name}")

    print(f"\nDone! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
