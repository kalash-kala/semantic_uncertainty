#!/bin/bash
# ============================================================================
# NQ generation — submit all 6 models, both partitions
# ============================================================================
# A100 batch sizes are the confirmed results from the NQ probe
# (logs/probe_nq/results.txt, GPU 1, 3000-sample slice, real-run flags).
# H200 batch sizes are derived per-model from measured VRAM-scaling multipliers
# ( (140-W)/(79-W), discounted for safety ), rounded down:
#   1.40x  qwen25_7b, llama31_8b, mistral7b   (7-8B)
#   1.45x  gemma3_12b
#   1.55x  qwen3_14b
#   2.30x  gemma3_27b
#
# Each `sbatch` call below is an INDEPENDENT job with its own GPU and its own
# 24h wall clock -- comment out any lines you don't want to submit right now
# and run this script again later for the rest. If a job hits the wall,
# resubmit the same line with RESUME_DIR=<run dir> added to --export.
#
# Usage:
#   ./submit_nq_all.sh a100   # submit the 6 A100 jobs
#   ./submit_nq_all.sh h200   # submit the 6 H200 jobs
#   ./submit_nq_all.sh all    # submit all 12 (only if both partitions are free)
# ============================================================================
set -euo pipefail

PARTITION=${1:?"usage: $0 {a100|h200|all}"}

declare -A MODEL_NAME=(
  [qwen25_7b]="Qwen/Qwen2.5-7B-Instruct"
  [llama31_8b]="meta-llama/Llama-3.1-8B-Instruct"
  [mistral7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [gemma3_12b]="google/gemma-3-12b-it"
  [qwen3_14b]="Qwen/Qwen3-14B"
  [gemma3_27b]="google/gemma-3-27b-it"
)

# Confirmed via logs/probe_nq/results.txt
declare -A BS_A100=(
  [qwen25_7b]=128
  [llama31_8b]=128
  [mistral7b]=128
  [gemma3_12b]=32
  [qwen3_14b]=64
  [gemma3_27b]=16
)

# A100 bs * multiplier, rounded down
declare -A BS_H200=(
  [qwen25_7b]=179
  [llama31_8b]=179
  [mistral7b]=179
  [gemma3_12b]=46
  [qwen3_14b]=99
  [gemma3_27b]=36
)

MODELS=(qwen25_7b llama31_8b mistral7b gemma3_12b qwen3_14b gemma3_27b)

submit_a100() {
  for key in "${MODELS[@]}"; do
    echo "sbatch [a100] $key  bs=${BS_A100[$key]}  ${MODEL_NAME[$key]}"
    sbatch --export=MODEL_NAME="${MODEL_NAME[$key]}",GENERATION_BATCH_SIZE="${BS_A100[$key]}" \
      run_nq_a100.sh
  done
}

submit_h200() {
  for key in "${MODELS[@]}"; do
    echo "sbatch [h200] $key  bs=${BS_H200[$key]}  ${MODEL_NAME[$key]}"
    sbatch --export=MODEL_NAME="${MODEL_NAME[$key]}",GENERATION_BATCH_SIZE="${BS_H200[$key]}" \
      run_nq_h200.sh
  done
}

case "$PARTITION" in
  a100) submit_a100 ;;
  h200) submit_h200 ;;
  all)  submit_a100; submit_h200 ;;
  *) echo "unknown partition '$PARTITION', expected a100|h200|all" >&2; exit 1 ;;
esac
