#!/bin/bash
# ============================================================================
# SLURM Wrapper for Generate Answers Combined — A100 Server
# ============================================================================
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=generate_answers_combined
#SBATCH --output=/home/sriramg/kalashabhayk/semantic_uncertainty/slurm_logs/logs/%x_%j.out
#SBATCH --error=/home/sriramg/kalashabhayk/semantic_uncertainty/slurm_logs/errors/%x_%j.err
#SBATCH --chdir=/home/sriramg/kalashabhayk/semantic_uncertainty
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# ============================================================================
# Example Commands (uncomment and modify as needed)
# ============================================================================
# sbatch --export=MODEL_NAME="Qwen/Qwen2.5-7B-Instruct",DATASET="trivia_qa",NUM_GENERATIONS=10,NUM_SAMPLES=10000,GENERATION_BATCH_SIZE=64 run_generate_answers_combined.sh
# sbatch --export=MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct",DATASET="trivia_qa",NUM_GENERATIONS=10,NUM_SAMPLES=10000,GENERATION_BATCH_SIZE=64 run_generate_answers_combined.sh
# sbatch --export=MODEL_NAME="google/gemma-3-12b-it",DATASET="trivia_qa",NUM_GENERATIONS=10,NUM_SAMPLES=10000,GENERATION_BATCH_SIZE=64 run_generate_answers_combined.sh
# sbatch --export=MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3",DATASET="trivia_qa",NUM_GENERATIONS=10,NUM_SAMPLES=10000,GENERATION_BATCH_SIZE=64 run_generate_answers_combined.sh
# sbatch --export=MODEL_NAME="google/gemma-3-12b-it",DATASET="trivia_qa",NUM_GENERATIONS=10,NUM_SAMPLES=10000,GENERATION_BATCH_SIZE=64,GPU_DEVICE=1 run_generate_answers_combined.sh
# Resume a previous incomplete run (generation already done, redo embedding extraction):
# sbatch --export=MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3",DATASET="trivia_qa_nocontext",NUM_GENERATIONS=10,NUM_SAMPLES=50000,GENERATION_BATCH_SIZE=64,GPU_DEVICE=0,RESUME_DIR="/storage/users/sriramg/kalashabhayk/semantic_uncertainty_data/uncertainty/trivia_qa_nocontext__mistralai__Mistral-7B-Instruct-v0.3__seed10__pid881004__20260512_114055" run_generate_answers_combined.sh

# ============================================================================
# Configuration Variables (modify as needed)
# ============================================================================

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
DATASET=${DATASET:-"trivia_qa"}
NUM_GENERATIONS=${NUM_GENERATIONS:-10}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
MODEL_MAX_NEW_TOKENS=${MODEL_MAX_NEW_TOKENS:-15}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-64}
COMPUTE_UNCERTAINTIES=${COMPUTE_UNCERTAINTIES:-"--compute_uncertainties"}
RANDOM_SEED=${RANDOM_SEED:-10}
HF_HOME=${HF_HOME:-"/home/sriramg/kalashabhayk/.cache/huggingface"}
SCRATCH_DIR=${SCRATCH_DIR:-"/storage/users/sriramg/kalashabhayk"}
GPU_DEVICE=${GPU_DEVICE:-""}  # Optional: set to "0", "1", etc. for CUDA_VISIBLE_DEVICES
RESUME_DIR=${RESUME_DIR:-""}  # Optional: path to a previous incomplete run directory to resume from

# ============================================================================
# Setup
# ============================================================================

# Create log directories
mkdir -p ./slurm_logs/logs ./slurm_logs/errors

# Setup conda environment (adjust environment name as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate semantic_uncertainty

# ============================================================================
# Run generate_answers_combined.py
# ============================================================================

echo "=========================================="
echo "Starting generate_answers_combined.py"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Num Generations: $NUM_GENERATIONS"
echo "Num Samples: $NUM_SAMPLES"
echo "Generation Batch Size: $GENERATION_BATCH_SIZE"
echo "Random Seed: $RANDOM_SEED"
echo "HF_HOME: $HF_HOME"
echo "SCRATCH_DIR: $SCRATCH_DIR"
if [ -n "$GPU_DEVICE" ]; then
    echo "GPU Device: $GPU_DEVICE"
fi
if [ -n "$RESUME_DIR" ]; then
    echo "Resume Dir: $RESUME_DIR"
fi
echo "=========================================="

# Build optional flags
RESUME_ARG=""
if [ -n "$RESUME_DIR" ]; then
    RESUME_ARG="--resume_dir $RESUME_DIR"
fi

# Build and run with appropriate environment variables
if [ -n "$GPU_DEVICE" ]; then
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
        $COMPUTE_UNCERTAINTIES \
        $RESUME_ARG
else
    env HF_HOME="$HF_HOME" \
        SCRATCH_DIR="$SCRATCH_DIR" \
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        python semantic_uncertainty/generate_answers_combined.py \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET" \
        --model_max_new_tokens "$MODEL_MAX_NEW_TOKENS" \
        --num_generations "$NUM_GENERATIONS" \
        --num_samples "$NUM_SAMPLES" \
        --generation_batch_size "$GENERATION_BATCH_SIZE" \
        --random_seed "$RANDOM_SEED" \
        $COMPUTE_UNCERTAINTIES \
        $RESUME_ARG
fi

echo "=========================================="
echo "Run complete!"
echo "=========================================="
