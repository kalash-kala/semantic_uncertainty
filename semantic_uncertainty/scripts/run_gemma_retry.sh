#!/bin/bash
# Runs one gemma model to completion on one GPU, retrying indefinitely from
# its own last flushed sample. Both gemma-12b and gemma-27b OOM'd inside
# compute_transition_scores (log_softmax on the stacked per-step scores
# tensor, huggingface_models.py:776) even after one resume at their original
# batch size, so this halves the batch size again and retries as many times
# as it takes -- no time bound, just get to 3000/3000.
#
#   ./run_gemma_retry.sh <gpu_id> <model> <batch_size> <resume_dir> [max_attempts]

set -u

GPU=$1; MODEL=$2; BSZ=$3; RESUME=$4; MAX_ATTEMPTS=${5:-20}

PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
SU=/home/kalashkala/semantic_uncertainty/semantic_uncertainty
LOGDIR=/home/kalashkala/semantic_uncertainty/semantic_uncertainty/logs/math_cot
mkdir -p "$LOGDIR"
SLUG=${MODEL//\//__}

run_dir_from_log() {
  grep -oP '(?<=Using run-specific output directory: ).*' "$1" | tail -1
}

ARGS=(
  --model_name "$MODEL"
  --dataset answerable_math
  --model_max_new_tokens 256
  --num_generations 10
  --num_samples 3000
  --generation_batch_size "$BSZ"
  --reasoning
  --metric math
)

cd "$SU" || exit 1

for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS"); do
  LOG="$LOGDIR/${SLUG}__retry${ATTEMPT}__$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%T)] GPU$GPU: attempt $ATTEMPT for $MODEL (bsz=$BSZ) resuming $RESUME" | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/.cache/huggingface \
    SU_TEACHER_FORCED_SCORES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" generate_answers_combined.py "${ARGS[@]}" --resume_dir "$RESUME" >> "$LOG" 2>&1

  if [ $? -eq 0 ]; then
    echo "[$(date +%T)] GPU$GPU: $MODEL COMPLETED on attempt $ATTEMPT" | tee -a "$LOG"
    exit 0
  fi

  NEW_RESUME=$(run_dir_from_log "$LOG")
  if [ -n "$NEW_RESUME" ] && [ -s "$NEW_RESUME/combined_generations.jsonl" ]; then
    RESUME="$NEW_RESUME"
  fi
  # Otherwise keep the previous RESUME -- this attempt died before flushing
  # anything new (e.g. OOM on the very first batch), so there is nothing to
  # gain from switching to its empty output dir.
  echo "[$(date +%T)] GPU$GPU: $MODEL failed attempt $ATTEMPT, will resume from $RESUME" | tee -a "$LOG"
  sleep 10
done

echo "[$(date +%T)] GPU$GPU: $MODEL FAILED after $MAX_ATTEMPTS attempts" | tee -a "$LOGDIR/${SLUG}__retry_giveup.log"
exit 1
