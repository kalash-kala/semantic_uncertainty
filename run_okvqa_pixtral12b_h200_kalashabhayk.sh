#!/bin/bash
# ============================================================================
# SLURM Wrapper — OK-VQA generation, pixtral-12b, H200 (kalashabhayk account)
# ============================================================================
# Identical to run_okvqa_pixtral12b_a100_kalashabhayk.sh except for the
# SBATCH resource block (H200 partition/qos). Runs the remaining OK-VQA
# model (Qwen2.5-VL already ran on the other box). Hardcoded to
# /home/sriramg/kalashabhayk since this script targets one specific
# account/server.
#
# Downloads the tiny OK-VQA questions/annotations JSON itself, but expects
# COCO train2014/val2014 images to already be rsync'd in (the public COCO
# host was far too slow from this cluster) and the model to already be in
# the HF cache (download with `hf download <model>` beforehand).
# --no-analyze_run skips the bootstrap-CI AUROC pass; uncertainty_measures.pkl
# is written first, so analyze_run can be re-run standalone later without
# regenerating.
#
# Pixtral's prompts run long (~1200 visual tokens/image) and generate 10
# sequences per example, so it is the OOM-prone one of the three VLMs --
# batch size is kept conservative (4, the value confirmed stable on the
# other server after an OOM at 8). Raise only after watching nvidia-smi.
#
#   sbatch run_okvqa_pixtral12b_h200_kalashabhayk.sh
# ============================================================================
#SBATCH --partition=h200
#SBATCH --account=sriramg
#SBATCH --qos=h200_qos
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=okvqa_pixtral12b_h200
#SBATCH --output=slurm_logs/logs/%x_%j.out
#SBATCH --error=slurm_logs/errors/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
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

MODEL_NAME="mistral-community/pixtral-12b-240910"
# Pixtral wasn't part of the NQ multiplier study (text-only models), so
# there's no measured H200 factor for it specifically. Applying the same
# 1.45x used for gemma3-12b as a same-parameter-class proxy on the
# A100-confirmed bs=4: 4*1.45=5.8 -> 5. Pixtral's ~1200 visual
# tokens/image make it the most OOM-prone of the three VLMs (it OOM'd at
# 8 on the 79 GiB card), so this is an extrapolation, not a confirmed
# value -- watch nvidia-smi and back off if it OOMs.
GENERATION_BATCH_SIZE=5
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
# OK-VQA questions + annotations (idempotent, flock-serialized -- tiny, ~1s
# each, safe to fetch on every job)
# ============================================================================
mkdir -p "$SCRATCH_DIR/vqa_data" "$OKVQA_DATA_ROOT"
(
flock -x 200
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
) 200>"$SCRATCH_DIR/vqa_data/.download.lock"

# ============================================================================
# COCO images -- NOT downloaded by this script. The public COCO host was
# far too slow from this cluster (~108KB/s, 33hr ETA for train2014.zip
# alone), so images are rsync'd in ahead of time from the box that already
# has them extracted. Fail fast with a clear message if they're missing
# rather than silently trying (and racing another job) to fetch them.
for split in train2014 val2014; do
  if [ ! -d "$OKVQA_IMG_ROOT/$split" ]; then
    echo "ERROR: $OKVQA_IMG_ROOT/$split not found." >&2
    echo "Expected COCO $split to be rsync'd in ahead of time, e.g.:" >&2
    echo "  rsync -avP <source>:/data/kalashkala/vqa_data/coco/$split $OKVQA_IMG_ROOT/" >&2
    exit 1
  fi
done

cd "$REPO_ROOT"

# ============================================================================
# HF model cache -- NOT pre-warmed by this script. Download the model
# yourself ahead of time, e.g.:
#   hf download $MODEL_NAME
# generate_answers_combined.py will use whatever's already in HF_HOME's
# cache; if it's missing it falls back to downloading it inline.
# ============================================================================

echo "=========================================="
echo "OK-VQA generation — pixtral-12b — H200"
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
