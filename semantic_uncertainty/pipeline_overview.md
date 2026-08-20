# Semantic Uncertainty Pipeline: End-to-End Overview

This document outlines the execution flow and core components of the `semantic_uncertainty` pipeline. The process transitions from raw answer generation to sophisticated uncertainty estimation and statistical analysis.

---

## 1. Phase 1: Answer Generation
**Script:** `generate_answers.py`

This phase interacts with the LLM to gather a diverse set of responses for each question in the dataset.

*   **Model Initialization**: Loads the targeted Large Language Model (e.g., Qwen2.5, Llama, Mistral) based on the provided HuggingFace path.
*   **Dual-Temperature Sampling**: 
    *   **Low-Temperature (T=0.1)**: Generates the "most likely" answer, used to determine the baseline accuracy of the model.
    *   **High-Temperature (T=1.0)**: Generates multiple stochastic samples (e.g., 10 responses). These samples form the distribution used to measure uncertainty.
*   **Self-Evaluation baseline (`p_true`)**: If enabled, the script asks the model itself to verify its output (e.g., "Is the answer provided above correct?"). This is a baseline for truthfulness.
*   **Output**: Saves raw text, token log-likelihoods, and internal embeddings to `validation_generations.jsonl` and `uncertainty_measures.pkl`.

---

## 2. Phase 2: Uncertainty Computation
**Script:** `compute_uncertainty_measures.py`

This phase processes the saved generations to calculate various metrics that capture how "uncertain" the model is about its answer.

### A. Semantic Clustering
The pipeline uses an **Entailment Model** (typically a DeBERTa-v3 variant) to group equivalent answers. 
*   **Process**: It checks if Answer A entails Answer B and vice versa. 
*   **Goal**: Distinguish between **syntactic** variety ("Paris", "the capital of France") and **semantic** variety ("Paris", "London"). The former should not increase uncertainty.

### B. Entropy Calculations
*   **Regular Entropy**: Measures the distribution of specific text strings generated at high temperature.
*   **Semantic Entropy (The Core Metric)**: Calculated using the `predictive_entropy_rao` method. It sums the probabilities of answers within the same semantic cluster before calculating entropy.
*   **Cluster Assignment Entropy**: measures the distribution of frequencies across the clusters identified.

### C. Embedding-based baseline (`p_ik`)
*   **Linear Classifier**: Fits a `LogisticRegression` model on the LLM's internal hidden-state embeddings.
*   **Mechanism**: It learns to predict if an answer is "False" just by looking at the model's internal "state of mind," providing a powerful baseline compared to purely text-based entropy.

---

## 3. Phase 3: Performance Analysis
**Script:** `analyze_results.py`

The final phase evaluates how well the uncertainty metrics actually correlate with model errors.

*   **AUROC (Area Under ROC Curve)**: The primary metric for uncertainty. It measures how well the uncertainty score separates Correct answers from Incorrect ones.
*   **Accuracy at Quantile**: Measures model accuracy if we only allow it to answer for the top X% most "certain" questions.
*   **Bootstrapping**: Computes 90% confidence intervals for every metric. This uses resampling to ensure results aren't due to a lucky subset of data.

---

## 4. Understanding Log Warnings

When reviewing your `uncertainty_run_qwen.log`, you may see the following:

| Warning | Meaning |
| :--- | :--- |
| `UndefinedMetricWarning` | Triggered when a split has **zero** unanswerable (or zero answerable) questions. AUROC becomes mathematically `nan` because there is no class to compare against. |
| `RuntimeWarning (scalar divide)` | Occurs during bootstrapping when a result is **perfectly constant** (e.g., 100% accuracy). There is zero variance for the algorithm to measure, causing a "divide by zero" in the normalization step. |

---

## 5. Summary of Sequential Execution

1.  **`generate_answers.py`** starts...
2.  $\downarrow$ **`model.predict()`** (Generates text)
3.  $\downarrow$ **`FINISHED generate_answers!`**
4.  $\downarrow$ **`compute_uncertainty_measures.py`** starts...
5.  $\downarrow$ **`get_semantic_ids()`** (Clusters answers using DeBERTa)
6.  $\downarrow$ **`get_p_ik()`** (Trains linear classifier on embeddings)
7.  $\downarrow$ **`analyze_run()`** (Calculates final AUROC and Bootstrap CIs)
8.  $\downarrow$ **`FINISHED compute_uncertainty_measures!`**
