#!/bin/bash
# ============================================================================
# SLURM Wrapper — OK-VQA generation, gemma-3-12b-it, A100 (kalashabhayk account)
# ============================================================================
# Runs the remaining OK-VQA model (Qwen2.5-VL already ran on the other box).
# Hardcoded to /home/sriramg/kalashabhayk since this script targets one
# specific account/server, unlike the generic wrappers at the repo root.
#
# Self-contained: downloads the OK-VQA questions/annotations + COCO
# train2014/val2014 images (idempotent -- skips anything already present)
# and pre-warms the HF model cache before generation. --no-analyze_run skips
# the bootstrap-CI AUROC pass; uncertainty_measures.pkl is written first, so
# analyze_run can be re-run standalone later without regenerating.
#
#   sbatch run_okvqa_gemma3_a100_kalashabhayk.sh
# ============================================================================
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=okvqa_gemma3_12b
#SBATCH --output=slurm_logs/logs/%x_%j.out
#SBATCH --error=slurm_logs/errors/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

# ============================================================================
# Configuration — hardcoded for /home/sriramg/kalashabhayk
# ============================================================================
REPO_ROOT="/home/sriramg/kalashabhayk/semantic_uncertainty"
SCRATCH_DIR="/storage/users/sriramg/kalashabhayk"
OKVQA_DATA_ROOT="$SCRATCH_DIR/vqa_data/okvqa"
OKVQA_IMG_ROOT="$SCRATCH_DIR/vqa_data/coco"

MODEL_NAME="google/gemma-3-12b-it"
# Confirmed on this GPU class (79.15 GiB capacity): bs=16 OOM'd during
# compute_transition_scores (logs/vqav2_gemma3_12b.log), bs=12 completed
# the full VQAv2 run cleanly. Reusing that confirmed value here.
GENERATION_BATCH_SIZE=12
NUM_GENERATIONS=10
NUM_SAMPLES=14055
MODEL_MAX_NEW_TOKENS=50
METRIC="squad"
RESUME_DIR=${RESUME_DIR:-""}

cd "$REPO_ROOT"
mkdir -p ./slurm_logs/logs ./slurm_logs/errors

source ~/miniconda3/etc/profile.d/conda.sh
conda activate semantic_uncertainty

# ============================================================================
# Download OK-VQA questions + annotations (idempotent)
# ============================================================================
mkdir -p "$OKVQA_DATA_ROOT"
cd "$OKVQA_DATA_ROOT"
for f in OpenEnded_mscoco_train2014_questions OpenEnded_mscoco_val2014_questions \
         mscoco_train2014_annotations mscoco_val2014_annotations; do
  if [ ! -f "${f}.json" ]; then
    echo "Downloading ${f}.json ..."
    curl -fL -o "${f}.json.zip" "https://okvqa.allenai.org/static/data/${f}.json.zip"
    unzip -q -o "${f}.json.zip"
    rm -f "${f}.json.zip"
  fi
done

# ============================================================================
# Download COCO train2014 + val2014 images (idempotent, ~20GB total)
# ============================================================================
mkdir -p "$OKVQA_IMG_ROOT"
cd "$OKVQA_IMG_ROOT"
for split in train2014 val2014; do
  if [ ! -d "$split" ]; then
    echo "Downloading COCO $split ..."
    curl -fL -o "${split}.zip" "http://images.cocodataset.org/zips/${split}.zip"
    unzip -q -o "${split}.zip"
    rm -f "${split}.zip"
  fi
done

cd "$REPO_ROOT"

# ============================================================================
# Pre-warm the HF model cache
# ============================================================================
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_NAME')
"

echo "=========================================="
echo "OK-VQA generation — gemma-3-12b-it — A100"
echo "=========================================="
echo "Model:              $MODEL_NAME"
echo "Num Samples:        $NUM_SAMPLES"
echo "Num Generations:    $NUM_GENERATIONS"
echo "Max New Tokens:     $MODEL_MAX_NEW_TOKENS"
echo "Batch Size:         $GENERATION_BATCH_SIZE"
echo "OKVQA_DATA_ROOT:    $OKVQA_DATA_ROOT"
echo "OKVQA_IMG_ROOT:     $OKVQA_IMG_ROOT"
[ -n "$RESUME_DIR" ] && echo "Resume Dir:         $RESUME_DIR"
echo "Node:               $(hostname)"
echo "Started:            $(date)"
echo "=========================================="

RESUME_ARG=""
if [ -n "$RESUME_DIR" ]; then
    RESUME_ARG="--resume_dir $RESUME_DIR"
fi

# ============================================================================
# Run
# ============================================================================
env OKVQA_DATA_ROOT="$OKVQA_DATA_ROOT" \
    OKVQA_IMG_ROOT="$OKVQA_IMG_ROOT" \
    SCRATCH_DIR="$SCRATCH_DIR" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    python semantic_uncertainty/generate_answers_combined.py \
    --model_name "$MODEL_NAME" \
    --dataset okvqa \
    --model_max_new_tokens "$MODEL_MAX_NEW_TOKENS" \
    --num_generations "$NUM_GENERATIONS" \
    --num_samples "$NUM_SAMPLES" \
    --num_few_shot 0 \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --metric "$METRIC" \
    --no-compute_p_true \
    --no-save_hidden_states \
    --no-use_context \
    --no-analyze_run \
    --entity okvqa_run \
    $RESUME_ARG

echo "=========================================="
echo "Run complete! $(date)"
echo "=========================================="
