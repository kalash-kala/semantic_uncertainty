"""
Cross-model label analysis for semantic uncertainty CSVs.
"""

import os
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuration: modify these lists as needed
FILES = [
    "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_gemma_gsm8k_combined_llm_verdict.csv",
    "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_llama_gsm8k_combined_llm_verdict.csv",
    "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_mistral_gsm8k_combined_llm_verdict.csv",
    "/home/kalashkala/Datasets/Semantic-Uncertainty/gsm8k/uncertainty_run_qwen_gsm8k_combined_llm_verdict.csv",
]
MODEL_NAMES = [
    "gemma",
    "llama",
    "mistral",
    "qwen",
]
THRESHOLD = 0.5
DATASET_NAME = "gsm8k"
OUTPUT_DIR = "/home/kalashkala/semantic_uncertainty/semantic_uncertainty/cross_model_analysis"

LABELS = ["Correct", "AH", "UH"]


def assign_labels(df: pd.DataFrame, threshold: float) -> pd.Series:
    def _label(row):
        if row["LLM_verdict"] is True or str(row["LLM_verdict"]).strip().lower() == "true":
            return "Correct"
        return "AH" if row["cluster_assignment_entropy"] < threshold else "UH"

    return df.apply(_label, axis=1)


def build_confusion_matrix(labels_a: pd.Series, labels_b: pd.Series) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for la, lb in zip(labels_a, labels_b):
        matrix.loc[la, lb] += 1
    return matrix


def plot_confusion_matrix_subplot(matrix: pd.DataFrame, model_a: str, model_b: str, ax):
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel(f"{model_b} label", fontsize=10)
    ax.set_ylabel(f"{model_a} label", fontsize=10)
    ax.set_title(f"{model_a} vs {model_b}", fontsize=11, fontweight="bold")


def print_summary(matrix: pd.DataFrame, model_a: str, model_b: str):
    total = matrix.values.sum()
    same = sum(matrix.loc[l, l] for l in LABELS)
    print(f"\n{model_a} vs {model_b}")
    print(f"  Total questions : {total}")
    print(f"  Same label      : {same}  ({100 * same / total:.1f}%)")
    print(f"  Different label : {total - same}  ({100 * (total - same) / total:.1f}%)")
    print("  Agreement breakdown:")
    for l in LABELS:
        print(f"    {l:8s}: {matrix.loc[l, l]}")


def main():
    if len(FILES) != len(MODEL_NAMES):
        raise ValueError("FILES and MODEL_NAMES must have the same number of entries")

    if len(FILES) == 0:
        raise ValueError("FILES list is empty. Please populate FILES and MODEL_NAMES lists.")

    out_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # Load and label all files
    labelled = {}
    for path, name in zip(FILES, MODEL_NAMES):
        df = pd.read_csv(path)
        required = {"LLM_verdict", "cluster_assignment_entropy"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"File {path} is missing columns: {missing}")
        df["label"] = assign_labels(df, THRESHOLD)
        labelled[name] = df
        print(f"Labelled {name}: {df['label'].value_counts().to_dict()}")

    # Pairwise cross-model analysis
    pairs = list(combinations(MODEL_NAMES, 2))
    n_pairs = len(pairs)

    # Create grid: 2 rows x 3 cols for up to 6 pairs
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (model_a, model_b) in enumerate(pairs):
        df_a = labelled[model_a]
        df_b = labelled[model_b]

        if len(df_a) != len(df_b):
            print(f"WARNING: {model_a} ({len(df_a)} rows) and {model_b} ({len(df_b)} rows) differ in length — truncating to shorter")
            n = min(len(df_a), len(df_b))
            df_a = df_a.iloc[:n]
            df_b = df_b.iloc[:n]

        matrix = build_confusion_matrix(df_a["label"], df_b["label"])
        print_summary(matrix, model_a, model_b)

        plot_confusion_matrix_subplot(matrix, model_a, model_b, axes[idx])

    # Hide any unused subplots
    for idx in range(n_pairs, 6):
        axes[idx].set_visible(False)

    plt.suptitle(f"Cross-model Label Comparison ({DATASET_NAME.upper()})", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"cross_model_confusion_matrices_{DATASET_NAME}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGrid plot saved to: {out_path}")


if __name__ == "__main__":
    main()
