"""Extract hidden states (embeddings) from stored token IDs.

Run this any time after generate_answers_combined.py has saved
combined_generations.jsonl — no need to re-run the full generation loop.

The script reads the JSONL, loads the same HuggingFace model used for
generation, runs a batched forward pass to recover the three hidden-state
positions for every response, and writes two PKL files that the existing
compute_uncertainty_measures.py pipeline consumes directly:

  - combined_generations.pkl
  - validation_generations.pkl  (identical copy, expected by compute script)

The three positions extracted per response:
  - last_prompt  : last token of the input prompt (just before generation)
  - first_answer : first generated token
  - last_token   : last generated token

Example commands
----------------
# Basic usage:
python extract_hidden_states.py \\
    --jsonl_path /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag>/combined_generations.jsonl \\
    --out_dir    /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag> \\
    --model_name Qwen/Qwen2.5-7B-Instruct

# Larger batch size (faster, needs more VRAM; auto-halved on OOM):
python extract_hidden_states.py \\
    --jsonl_path /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag>/combined_generations.jsonl \\
    --out_dir    /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag> \\
    --model_name meta-llama/Llama-3.1-8B-Instruct \\
    --batch_size 256

# With conda env + env vars (mirrors the generate_answers_combined.py style):
nohup conda run --no-capture-output -n semantic_uncertainty \\
    env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python extract_hidden_states.py \\
        --jsonl_path /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag>/combined_generations.jsonl \\
        --out_dir    /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag> \\
        --model_name Qwen/Qwen2.5-7B-Instruct \\
    > extract_hidden_states_<run_tag>.log 2>&1 &

Notes
-----
- <run_tag> is printed at the start of a generate_answers_combined.py run, e.g.
  trivia_qa__Qwen__Qwen2.5-7B-Instruct__seed0__pid12345__20250430_120000
- --model_name must exactly match what was used during generation so that
  tokenizer / model architecture align with the stored token IDs.
- PKL files are written to --out_dir, overwriting any existing files of the
  same name. The JSONL is never modified.
"""

import argparse
import gc
import logging
import os
import sys

import torch

# Make sure the project root is on sys.path when running the script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from uncertainty.utils import utils
from generate_answers_combined import _extract_embeddings_from_token_ids

utils.setup_logger()


def _build_args(parsed):
    """Wrap parsed CLI args in a lightweight namespace that init_model accepts."""
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description='Extract hidden-state embeddings from token IDs stored in a JSONL file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--jsonl_path', type=str, required=True,
        help='Path to combined_generations.jsonl produced by generate_answers_combined.py.',
    )
    parser.add_argument(
        '--out_dir', type=str, required=True,
        help='Directory where combined_generations.pkl and validation_generations.pkl are written. '
             'Usually the same directory that contains the JSONL file.',
    )
    parser.add_argument(
        '--model_name', type=str, required=True,
        help='HuggingFace model identifier used during generation '
             '(e.g. Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct).',
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Number of sequences per GPU forward pass. Auto-halved on OOM.',
    )
    # init_model only needs model_name and model_max_new_tokens from args.
    parser.add_argument(
        '--model_max_new_tokens', type=int, default=15,
        help='Must match the value used during generation (affects tokenizer stop logic).',
    )

    args = parser.parse_args()

    if not os.path.isfile(args.jsonl_path):
        logging.error('JSONL file not found: %s', args.jsonl_path)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    # utils.save() writes to the directory pointed to by SU_LOCAL_RUN_DIR.
    os.environ['SU_LOCAL_RUN_DIR'] = args.out_dir

    logging.info('Loading model: %s', args.model_name)
    model = utils.init_model(args)

    logging.info('Extracting embeddings from: %s  (batch_size=%d)', args.jsonl_path, args.batch_size)
    generations = _extract_embeddings_from_token_ids(model, args.jsonl_path, bsz=args.batch_size)

    logging.info('Unloading model.')
    del model
    gc.collect()
    torch.cuda.empty_cache()

    logging.info('Saving PKL files to: %s', args.out_dir)
    utils.save(generations, 'combined_generations.pkl')
    utils.save(generations, 'validation_generations.pkl')
    logging.info('Done. Written files:')
    logging.info('  %s/combined_generations.pkl', args.out_dir)
    logging.info('  %s/validation_generations.pkl', args.out_dir)


if __name__ == '__main__':
    main()
