#!/bin/bash
# AdVQA UQ runs: Qwen2.5-VL-7B then gemma-3-12b-it, sequentially on one GPU.
# Non-reasoning, no hidden states, p_true disabled.
set -u
cd /home/kalashkala/semantic_uncertainty/semantic_uncertainty

PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
export HF_HOME=/data/.cache/huggingface
export CUDA_VISIBLE_DEVICES=${GPU:-1}
mkdir -p logs/vlm

run_one () {
  local model="$1" bsz="$2" tag="$3"
  local log="logs/vlm/${tag}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== $(date) starting $model (bsz=$bsz) -> $log ==="
  $PY generate_answers_combined.py \
    --model_name "$model" \
    --dataset advqa \
    --model_max_new_tokens 50 \
    --num_generations 10 \
    --num_samples 3000 \
    --num_few_shot 0 \
    --generation_batch_size "$bsz" \
    --metric squad \
    --no-compute_p_true \
    --no-save_hidden_states \
    --no-use_context \
    --no-get_training_set_generations \
    > "$log" 2>&1
  echo "=== $(date) finished $model rc=$? ==="
}

run_one "Qwen/Qwen2.5-VL-7B-Instruct" 8 qwen25vl
run_one "google/gemma-3-12b-it"       8 gemma3_12b
echo "=== ALL VLM RUNS DONE $(date) ==="
