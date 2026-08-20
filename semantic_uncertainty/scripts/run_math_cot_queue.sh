#!/bin/bash
# Runs the answerable_math CoT regeneration for one GPU's queue of models.
#
#   ./run_math_cot_queue.sh <gpu_id> <model> [<model> ...]
#
# Each model is retried once with --resume_dir pointing at its own partial run
# directory, so a mid-run failure picks up from the last flushed sample instead
# of restarting. The flags are held in a single array and reused verbatim for
# the retry: resuming a CoT run without --reasoning would silently append
# short-form answers to a chain-of-thought JSONL.

set -u

GPU=$1; shift
MODELS=("$@")

PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
SU=/home/kalashkala/semantic_uncertainty/semantic_uncertainty
DATA=/data/kalashkala/semantic_uncertainty_data/uncertainty
LOGDIR=/home/kalashkala/semantic_uncertainty/semantic_uncertainty/logs/math_cot
mkdir -p "$LOGDIR"

# Sized from: weights + 2*(max_new_tokens * B*N * vocab * 4) + KV + activations,
# against an 80GB card with headroom, then derated ~35% because measured usage
# ran above the analytic model. The doubled scores term is generate()'s per-step
# scores tuple plus the copy compute_transition_scores stacks out of it; at
# max_new_tokens=256 and num_generations=10 that term dominates everything else,
# which is what OOMed the first attempt at these batch sizes.
batch_size_for() {
  case "$1" in
    *27b*)       echo 2 ;;   # 51G of weights leaves almost nothing
    *12b*)       echo 4 ;;   # 262k vocab makes the scores term expensive
    *14B*)       echo 6 ;;
    *Mistral*)   echo 16 ;;  # 32k vocab -- scores term is 8x cheaper
    *)           echo 8 ;;   # 7-8B, ~150k vocab
  esac
}

# The run directory this attempt actually created, read back from its own log.
# Globbing $DATA for the newest matching directory would be wrong: unrelated
# runs (old baselines, small probes) share the naming scheme and could be
# resumed into by mistake.
run_dir_from_log() {
  grep -oP '(?<=Using run-specific output directory: ).*' "$1" | tail -1
}

for MODEL in "${MODELS[@]}"; do
  BSZ=$(batch_size_for "$MODEL")
  SLUG=${MODEL//\//__}
  # Timestamped per invocation: run_dir_from_log greps this file for the
  # directory to resume into, so it must not contain a previous run's path.
  LOG="$LOGDIR/${SLUG}__$(date +%Y%m%d_%H%M%S).log"

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

  echo "[$(date +%T)] GPU$GPU: starting $MODEL (bsz=$BSZ)" | tee -a "$LOG"
  cd "$SU" || exit 1
  # SU_TEACHER_FORCED_SCORES=0 keeps the original compute_transition_scores path
  # bit-for-bit, so these runs stay comparable with every previous generation.
  # The teacher-forced scorer matches it exactly for greedy but deviates ~5% on
  # temperature-sampled sequences, which is what feeds the entropy measures.
  CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/.cache/huggingface \
    SU_TEACHER_FORCED_SCORES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" generate_answers_combined.py "${ARGS[@]}" >> "$LOG" 2>&1

  if [ $? -ne 0 ]; then
    RESUME=$(run_dir_from_log "$LOG")
    # A run that died before flushing its first sample has nothing to resume;
    # retrying it just reproduces the same failure (an OOM on batch 1 did
    # exactly this last time, doubling the wasted time).
    if [ -n "$RESUME" ] && [ -s "$RESUME/combined_generations.jsonl" ]; then
      echo "[$(date +%T)] GPU$GPU: $MODEL failed, resuming from $RESUME" | tee -a "$LOG"
      CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/data/.cache/huggingface \
        SU_TEACHER_FORCED_SCORES=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" generate_answers_combined.py "${ARGS[@]}" \
          --resume_dir "$RESUME" >> "$LOG" 2>&1
      [ $? -ne 0 ] && echo "[$(date +%T)] GPU$GPU: $MODEL FAILED after resume" | tee -a "$LOG"
    else
      echo "[$(date +%T)] GPU$GPU: $MODEL failed with no run dir to resume" | tee -a "$LOG"
    fi
  fi

  echo "[$(date +%T)] GPU$GPU: finished $MODEL" | tee -a "$LOG"
done

echo "[$(date +%T)] GPU$GPU: queue complete"
