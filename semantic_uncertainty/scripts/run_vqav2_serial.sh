#!/bin/bash
# ============================================================================
# VQAv2 generation — serial driver, single GPU
# ============================================================================
# Runs each VLM in sequence on one GPU (they don't fit two at once alongside
# other jobs). qwen25vl already completed successfully at bs=16
# (logs/vqav2_qwen25vl.log, finished 2026-08-20). gemma3_12b completed
# generation + uncertainty_measures.pkl at bs=12 (accuracy 0.4715,
# logs/vqav2_gemma3_12b.log) -- its analyze_run bootstrap-CI pass was killed
# partway through since it's a re-runnable post-hoc step, not a dependency
# (see --no-analyze_run note below). Only pixtral12b remains to run.
#
# --no-analyze_run: skip the bootstrap-CI AUROC reporting pass at the end.
# uncertainty_measures.pkl (raw per-item cluster_assignment_entropy etc.) is
# written before analyze_run runs, so nothing is lost -- analyze_run('local')
# can be re-run standalone against that pkl later, whenever the AUROC numbers
# are actually needed, without re-running generation.
#
# Usage: setsid nohup ./run_vqav2_serial.sh < /dev/null &
# ============================================================================
set -e
cd /home/kalashkala/semantic_uncertainty/semantic_uncertainty
export HF_HOME=/data/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1
PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
LOGDIR=logs

run_model () {
  name=$1
  model=$2
  bs=$3
  echo "=== starting $name $(date) ==="
  $PY generate_answers_combined.py \
    --model_name "$model" \
    --dataset vqav2 \
    --model_max_new_tokens 50 \
    --num_generations 10 \
    --num_samples 40000 \
    --num_few_shot 0 \
    --generation_batch_size "$bs" \
    --metric squad \
    --no-compute_p_true \
    --no-save_hidden_states \
    --no-use_context \
    --no-analyze_run \
    --entity vqav2_run \
    >> "$LOGDIR/vqav2_${name}.log" 2>&1
  echo "=== finished $name $(date) ==="
}

run_model pixtral12b "mistral-community/pixtral-12b" 8

echo "ALL DONE $(date)"
