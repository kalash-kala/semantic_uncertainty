#!/bin/bash
# ============================================================================
# Multi-GPU Runner for Generate Answers Combined — 2x A100 Server
# ============================================================================
# This script runs inference across 2x A100 GPUs.
# Currently configured to run Qwen3-14B with Thinking Mode on GPU 0.
# (GPU 1 is currently occupied by ongoing Qwen-2.5-7B / Mistral runs).
#
# Usage:
#   bash run_generate_answers_combined_malay.sh
#   # Or run in background with nohup:
#   # nohup bash run_generate_answers_combined_malay.sh > ./logs/master_run_qwen3.log 2>&1 &
# ============================================================================

# ============================================================================
# Configuration Variables (modify as needed)
# ============================================================================

MODEL_1=${MODEL_1:-"Qwen/Qwen3-14B"}
GPU_1=${GPU_1:-"0"}

# MODEL_2=${MODEL_2:-"Qwen/Qwen2.5-7B"}
# GPU_2=${GPU_2:-"1"}

DATASET=${DATASET:-"trivia_qa_nocontext"}
NUM_GENERATIONS=${NUM_GENERATIONS:-10}
NUM_SAMPLES=${NUM_SAMPLES:-50000}
MODEL_MAX_NEW_TOKENS=${MODEL_MAX_NEW_TOKENS:-1024}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-16}
COMPUTE_UNCERTAINTIES=${COMPUTE_UNCERTAINTIES:-"--compute_uncertainties"}
RANDOM_SEED=${RANDOM_SEED:-10}
HF_HOME=${HF_HOME:-"/data/.cache/huggingface"}
SCRATCH_DIR=${SCRATCH_DIR:-"/data/malay/semantic_uncertainty"}
ENABLE_THINKING=${ENABLE_THINKING:-"--enable_thinking"}

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
echo "Model 1 ($MODEL_1) -> GPU $GPU_1 (Thinking Mode: Enabled)"
echo "Dataset: $DATASET"
echo "Num Generations: $NUM_GENERATIONS"
echo "Num Samples: $NUM_SAMPLES"
echo "Max New Tokens: $MODEL_MAX_NEW_TOKENS"
echo "Generation Batch Size: $GENERATION_BATCH_SIZE"
echo "Random Seed: $RANDOM_SEED"
echo "HF_HOME: $HF_HOME"
echo "SCRATCH_DIR: $SCRATCH_DIR"
echo "=========================================="

# Launch Model 1 on GPU 0
LOG_FILE_1="./logs/qwen3_14b_thinking_${DATASET}.log"
echo "[GPU $GPU_1] Launching $MODEL_1 with Thinking Mode ... (Logs: $LOG_FILE_1)"
env HF_HOME="$HF_HOME" \
    SCRATCH_DIR="$SCRATCH_DIR" \
    CUDA_VISIBLE_DEVICES="$GPU_1" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    python semantic_uncertainty/generate_answers_combined.py \
    --model_name "$MODEL_1" \
    --dataset "$DATASET" \
    --model_max_new_tokens "$MODEL_MAX_NEW_TOKENS" \
    --num_generations "$NUM_GENERATIONS" \
    --num_samples "$NUM_SAMPLES" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --random_seed "$RANDOM_SEED" \
    $ENABLE_THINKING \
    $COMPUTE_UNCERTAINTIES > "$LOG_FILE_1" 2>&1 &
PID_1=$!
echo "[GPU $GPU_1] Started PID: $PID_1"

# To run a second model on GPU 1 when free, uncomment below:
# LOG_FILE_2="./logs/qwen_2.5_7b_${DATASET}.log"
# echo "[GPU $GPU_2] Launching $MODEL_2 ... (Logs: $LOG_FILE_2)"
# env HF_HOME="$HF_HOME" \
#     SCRATCH_DIR="$SCRATCH_DIR" \
#     CUDA_VISIBLE_DEVICES="$GPU_2" \
#     PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
#     python semantic_uncertainty/generate_answers_combined.py \
#     --model_name "$MODEL_2" \
#     --dataset "$DATASET" \
#     --model_max_new_tokens "$MODEL_MAX_NEW_TOKENS" \
#     --num_generations "$NUM_GENERATIONS" \
#     --num_samples "$NUM_SAMPLES" \
#     --generation_batch_size "$GENERATION_BATCH_SIZE" \
#     --random_seed "$RANDOM_SEED" \
#     $COMPUTE_UNCERTAINTIES > "$LOG_FILE_2" 2>&1 &
# PID_2=$!
# echo "[GPU $GPU_2] Started PID: $PID_2"

echo "=========================================="
echo "Model launched! Waiting for job to complete..."
echo "Monitor progress with: tail -f $LOG_FILE_1"
echo "=========================================="

# Wait for background process to finish
wait $PID_1

echo "=========================================="
echo "All runs completed!"
echo "=========================================="
