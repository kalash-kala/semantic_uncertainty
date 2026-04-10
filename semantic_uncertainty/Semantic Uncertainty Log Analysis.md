# Semantic Uncertainty Log Analysis: `uncertainty_run_qwen.log`

This document summarizes the different types of information being captured in the pipeline logs to help navigate the execution flow.

---

## 1. Run Configuration & Metadata
At the very top, the log records the exact state of the environment and the parameters used:
*   **Namespace (Args)**: Captures every command-line flag (e.g., `num_samples=150`, `model_name='Qwen/Qwen2.5-7B-Instruct'`, `dataset='trivia_qa'`).
*   **System Info**: Timestamps and library initialization messages.

## 2. Model & Dataset Loading
Records the heavy lifting of preparing the environment:
*   **HTTP Requests**: Logs interactions with HuggingFace Hub to verify dataset revisions and model configs.
*   **Dataset Schema**: Prints the features (`question`, `context`, `answers`) and the number of rows in the loaded splits.
*   **Weight Loading**: Real-time progress bars indicating the status of loading the LLM into memory.

## 3. P(True) Few-Shot Construction
The pipeline uses a "self-evaluation" baseline that requires high-quality examples. The log captures:
*   **Brainstorming**: Shows the model generating 10 alternatives for "seed" questions to build the few-shot prompt.
*   **Accuracy Check**: Logs if the model actually knows the answer to these seed questions before including them.
*   **Final Prompt**: Dumps the complete formatted `p_true_few_shot_prompt` that will be sent to the model.

## 4. Execution Flow Indicators
Key markers that help identify which phase of the pipeline is active:
*   `STARTING generate_answers!`
*   `Starting with dataset_split train`
*   `Starting with dataset_split validation`
*   `Overall train split accuracy` summary.

## 5. Iteration-Level Details (Per Question)
For every question in the sample set, the log provides a detailed block:
*   **Current Input**: The raw question being passed to the model.
*   **Low-T Prediction**: The model's answer generated at temperature 0.1 (greedy-like).
*   **Ground Truth**: The reference answer(s) from the dataset.
*   **Accuracy**: A local score (0.0 or 1.0) for that specific response.
*   **Model Probabilities**: Token log-likelihoods (if enabled).

## 6. Warnings & Errors
Crucial for debugging and health monitoring:
*   **Max Token Limit**: Warning `Generation interrupted by max_token limit` appears if the model's response is too wordy for the `model_max_new_tokens` setting.
*   **Runtime Warnings**: Statistical warnings like SciPy's `scalar divide by zero` during bootstrapping or Scikit-learn's `UndefinedMetricWarning` when data splits are skewed.

## 7. Performance Counters
*   **TQDM Bars**: High-level progress indicators for generation loops and uncertainty loops.
*   **Aggregate Stats**: Final accuracy and mean uncertainty metrics printed at the end of each stage.
