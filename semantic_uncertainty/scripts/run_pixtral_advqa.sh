#!/bin/bash
# Pixtral-12B on AdVQA. Waits for run_vlm_advqa.sh (qwen -> gemma) to finish
# first: Pixtral peaks ~42GB at bsz=4 and gemma ~40GB, which together would
# exceed the 80GB card.
set -u
cd /home/kalashkala/semantic_uncertainty/semantic_uncertainty

PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
export HF_HOME=/data/.cache/huggingface
export CUDA_VISIBLE_DEVICES=${GPU:-1}
mkdir -p logs/vlm

until grep -q "ALL VLM RUNS DONE" logs/vlm/driver.log 2>/dev/null; do
  sleep 60
done
echo "=== $(date) predecessor queue finished, starting Pixtral ==="

log="logs/vlm/pixtral_$(date +%Y%m%d_%H%M%S).log"
$PY generate_answers_combined.py \
  --model_name mistral-community/pixtral-12b \
  --dataset advqa \
  --model_max_new_tokens 50 \
  --num_generations 10 \
  --num_samples 3000 \
  --num_few_shot 0 \
  --generation_batch_size 4 \
  --metric squad \
  --no-compute_p_true \
  --no-save_hidden_states \
  --no-use_context \
  --no-get_training_set_generations \
  > "$log" 2>&1
echo "=== $(date) Pixtral finished rc=$? log=$log ==="
