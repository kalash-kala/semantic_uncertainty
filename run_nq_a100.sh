#!/bin/bash
# ============================================================================
# SLURM Wrapper — Natural Questions (nq) generation, A100 partition
# ============================================================================
# One model per submission. Each `sbatch` below is an INDEPENDENT job with its
# own GPU and its own 24h wall clock -- they do not run sequentially, so the
# six models can be in the queue at once.
#
#   sbatch --export=MODEL_NAME="Qwen/Qwen2.5-7B-Instruct",GENERATION_BATCH_SIZE=64 run_nq_a100.sh
#
# Individual model commands (confirmed A100 batch sizes, see submit_nq_all.sh):
#   sbatch --export=MODEL_NAME="Qwen/Qwen2.5-7B-Instruct",GENERATION_BATCH_SIZE=128 run_nq_a100.sh
#   sbatch --export=MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct",GENERATION_BATCH_SIZE=128 run_nq_a100.sh
#   sbatch --export=MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3",GENERATION_BATCH_SIZE=128 run_nq_a100.sh
#   sbatch --export=MODEL_NAME="google/gemma-3-12b-it",GENERATION_BATCH_SIZE=32 run_nq_a100.sh
#   sbatch --export=MODEL_NAME="Qwen/Qwen3-14B",GENERATION_BATCH_SIZE=64 run_nq_a100.sh
#   sbatch --export=MODEL_NAME="google/gemma-3-27b-it",GENERATION_BATCH_SIZE=16 run_nq_a100.sh
#
# If a model does hit the 24h wall, resume it instead of restarting:
#   sbatch --export=MODEL_NAME="google/gemma-3-27b-it",GENERATION_BATCH_SIZE=16,RESUME_DIR="<run dir>" run_nq_a100.sh
#
# NOTE: #SBATCH directives cannot reference shell variables, so the paths below
# are literal. Adjust them if this repo lives elsewhere on the cluster.
# ============================================================================
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --job-name=nq_generate
#SBATCH --output=slurm_logs/logs/%x_%j.out
#SBATCH --error=slurm_logs/errors/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME=${MODEL_NAME:?"MODEL_NAME must be set, e.g. --export=MODEL_NAME=Qwen/Qwen2.5-7B-Instruct,..."}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:?"GENERATION_BATCH_SIZE must be set (see submit_nq_all.sh for tuned values)"}

DATASET=${DATASET:-"nq"}
NUM_GENERATIONS=${NUM_GENERATIONS:-10}
NUM_SAMPLES=${NUM_SAMPLES:-50000}
MODEL_MAX_NEW_TOKENS=${MODEL_MAX_NEW_TOKENS:-15}
METRIC=${METRIC:-"squad"}
COMPUTE_UNCERTAINTIES=${COMPUTE_UNCERTAINTIES:-"--compute_uncertainties"}
RANDOM_SEED=${RANDOM_SEED:-10}
# Both derived from $HOME so this resolves correctly under whichever account
# runs the job, e.g. $HOME=/home/sriramg/kalashabhayk -> storage/users/sriramg/kalashabhayk
HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
SCRATCH_DIR=${SCRATCH_DIR:-"/storage/users${HOME#/home}"}
RESUME_DIR=${RESUME_DIR:-""}

# Pins the exact 50k rows so every server evaluates an identical set. The file
# is committed in the repo; if it is missing the first run writes it.
SELECTION_MANIFEST=${SELECTION_MANIFEST:-"manifests/nq_50k_seed10.json"}

# ============================================================================
# Setup
# ============================================================================
mkdir -p ./slurm_logs/logs ./slurm_logs/errors

source ~/miniconda3/etc/profile.d/conda.sh
conda activate semantic_uncertainty

echo "=========================================="
echo "NQ generation — A100"
echo "=========================================="
echo "Model:              $MODEL_NAME"
echo "Dataset:            $DATASET"
echo "Num Samples:        $NUM_SAMPLES"
echo "Num Generations:    $NUM_GENERATIONS"
echo "Max New Tokens:     $MODEL_MAX_NEW_TOKENS"
echo "Batch Size:         $GENERATION_BATCH_SIZE"
echo "Random Seed:        $RANDOM_SEED"
echo "Selection Manifest: $SELECTION_MANIFEST"
echo "HF_HOME:            $HF_HOME"
echo "SCRATCH_DIR:        $SCRATCH_DIR"
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
# Non-reasoning arm: --model_max_new_tokens 15 with no --reasoning and no
# --enable_thinking (both default off). The explicit --no-enable_thinking below
# matters for Qwen3-14B, which would otherwise be the one model in the set
# tempted into a thinking trace and blow past the 15-token budget.
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
    --metric "$METRIC" \
    --selection_manifest "$SELECTION_MANIFEST" \
    --no-enable_thinking \
    --no-reasoning \
    --no-save_hidden_states \
    --no-analyze_run \
    $COMPUTE_UNCERTAINTIES \
    $RESUME_ARG

echo "=========================================="
echo "Run complete! $(date)"
echo "=========================================="
