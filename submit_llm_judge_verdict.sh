#!/bin/bash
#SBATCH --partition=h200
#SBATCH --account=sriramg
#SBATCH --qos=h200_qos
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=llm_judge_verdict_hf_llama70b
#SBATCH --output=/home/sriramg/kalashabhayk/semantic_uncertainty/slurm_logs/logs/%x_%j.out
#SBATCH --error=/home/sriramg/kalashabhayk/semantic_uncertainty/slurm_logs/errors/%x_%j.err
#SBATCH --chdir=/home/sriramg/kalashabhayk/semantic_uncertainty
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ─────────────────────────────────────────────────────────────────────────────
# LLM Judge Verdict Script for Semantic Uncertainty
# Uses Llama-3.3-70B-Instruct model via HuggingFace transformers backend
# ─────────────────────────────────────────────────────────────────────────────

set -e

# Configuration
MODEL_PATH="/home/sriramg/sakshamm/models/Llama-3.3-70B-Instruct"
CUDA_DEVICES="0,1"
BATCH_SIZE=8
MAX_NEW_TOKENS=16

# Paths
SCRIPT_PATH="/home/sriramg/kalashabhayk/semantic_uncertainty/semantic_uncertainty/scripts/llm_judge_verdict_hf.py"

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting LLM Judge Verdict at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_DEVICES"
echo ""

# Print environment
echo "Environment:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU Count: {torch.cuda.device_count()}')"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Run LLM Judge Verdict
# ─────────────────────────────────────────────────────────────────────────────

echo "Running LLM Judge Verdict..."
echo "Model: $MODEL_PATH"
echo "Script: $SCRIPT_PATH"
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$SCRIPT_PATH" \
    --model "$MODEL_PATH" \
    --cuda_device "$CUDA_DEVICES" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --skip_existing

echo ""
echo "LLM Judge Verdict completed at $(date)"
