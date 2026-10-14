#!/bin/bash
# ============================================================================
# Single-GPU Runner for Generate Answers Combined — Mistral-7B-v0.3
# ============================================================================
# This script runs mistralai/Mistral-7B-v0.3 on GPU 0:
#   - GPU 0: mistralai/Mistral-7B-v0.3
#
# Usage:
#   bash run_generate_answers_combined_mistral.sh
#   # Or run in background with nohup:
#   # nohup bash run_generate_answers_combined_mistral.sh > ./logs/mistral_run.log 2>&1 &
# ============================================================================

# ============================================================================
# Configuration Variables (modify as needed)
# ============================================================================

MODEL_NAME=${MODEL_NAME:-"mistralai/Mistral-7B-v0.3"}
GPU_DEVICE=${GPU_DEVICE:-"0"}

DATASET=${DATASET:-"trivia_qa_nocontext"}
NUM_GENERATIONS=${NUM_GENERATIONS:-10}
NUM_SAMPLES=${NUM_SAMPLES:-50000}
MODEL_MAX_NEW_TOKENS=${MODEL_MAX_NEW_TOKENS:-15}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-16}
COMPUTE_UNCERTAINTIES=${COMPUTE_UNCERTAINTIES:-"--compute_uncertainties"}
RANDOM_SEED=${RANDOM_SEED:-10}
HF_HOME=${HF_HOME:-"/data/.cache/huggingface"}
SCRATCH_DIR=${SCRATCH_DIR:-"/data/malay/semantic_uncertainty"}

# ============================================================================
# Setup
# ============================================================================

# Create log directory
mkdir -p ./logs

# Setup conda environment (adjust environment name as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate semantic_uncertainty

# ============================================================================
# Run generate_answers_combined.py on GPU 0
# ============================================================================

echo "=========================================="
echo "Starting Execution"
echo "=========================================="
echo "Model: $MODEL_NAME -> GPU $GPU_DEVICE"
echo "Dataset: $DATASET"
echo "Num Generations: $NUM_GENERATIONS"
echo "Num Samples: $NUM_SAMPLES"
echo "Generation Batch Size: $GENERATION_BATCH_SIZE"
echo "Random Seed: $RANDOM_SEED"
echo "HF_HOME: $HF_HOME"
echo "SCRATCH_DIR: $SCRATCH_DIR"
echo "=========================================="

LOG_FILE="./logs/mistral_7b_v0.3_${DATASET}.log"
echo "[GPU $GPU_DEVICE] Launching $MODEL_NAME ... (Logs: $LOG_FILE)"

env HF_HOME="$HF_HOME" \
    SCRATCH_DIR="$SCRATCH_DIR" \
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    python semantic_uncertainty/generate_answers_combined.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --model_max_new_tokens "$MODEL_MAX_NEW_TOKENS" \
    --num_generations "$NUM_GENERATIONS" \
    --num_samples "$NUM_SAMPLES" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --random_seed "$RANDOM_SEED" \
    $COMPUTE_UNCERTAINTIES > "$LOG_FILE" 2>&1 &
PID=$!
echo "[GPU $GPU_DEVICE] Started PID: $PID"

echo "=========================================="
echo "Model launched! Waiting for job to complete..."
echo "Monitor progress with: tail -f $LOG_FILE"
echo "=========================================="

# Wait for background process to finish
wait $PID

echo "=========================================="
echo "Run completed!"
echo "=========================================="
