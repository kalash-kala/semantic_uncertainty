#!/bin/bash
# ============================================================================
# OK-VQA generation — Qwen2.5-VL-7B-Instruct, single GPU
# ============================================================================
# Combined train2014+val2014 pool (14,055 questions, see uncertainty/data/
# data_utils.py "okvqa" branch). Batch size 16 reused from the VQAv2 run,
# where qwen25vl completed cleanly at bs=16 (logs/vqav2_qwen25vl.log).
#
# --no-analyze_run: skip the bootstrap-CI AUROC reporting pass at the end.
# uncertainty_measures.pkl (raw per-item cluster_assignment_entropy etc.) is
# written before analyze_run runs, so nothing is lost -- analyze_run('local')
# can be re-run standalone against that pkl later, whenever the AUROC numbers
# are actually needed, without re-running generation.
#
# Usage: setsid nohup ./scripts/run_okvqa_qwen25vl.sh < /dev/null &
# To resume an interrupted run: RESUME_DIR=<out_dir> ./scripts/run_okvqa_qwen25vl.sh
# ============================================================================
set -e
cd /home/kalashkala/semantic_uncertainty/semantic_uncertainty
export HF_HOME=/data/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
LOGDIR=logs
RESUME_DIR=${RESUME_DIR:-""}

RESUME_ARG=""
if [ -n "$RESUME_DIR" ]; then
  RESUME_ARG="--resume_dir $RESUME_DIR"
fi

echo "=== starting okvqa_qwen25vl $(date) ==="
$PY generate_answers_combined.py \
  --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
  --dataset okvqa \
  --model_max_new_tokens 50 \
  --num_generations 10 \
  --num_samples 14055 \
  --num_few_shot 0 \
  --generation_batch_size 16 \
  --metric squad \
  --no-compute_p_true \
  --no-save_hidden_states \
  --no-use_context \
  --no-analyze_run \
  --entity okvqa_run \
  $RESUME_ARG \
  >> "$LOGDIR/okvqa_qwen25vl.log" 2>&1
echo "=== finished okvqa_qwen25vl $(date) ==="
