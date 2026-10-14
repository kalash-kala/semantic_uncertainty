"""Extract hidden states (embeddings) for a subset of rows from a sampled CSV.

This is a variant of extract_hidden_states.py that filters the JSONL records
to only the question IDs present in a sampled CSV before running the forward
pass. This avoids loading a model and processing thousands of records when only
a few hundred are needed.

The CSV must have an 'id' column whose values match the top-level keys in the
JSONL file (e.g. 'train::10225').

Example command
---------------
nohup conda run --no-capture-output -n semantic_uncertainty \\
    env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python extract_hidden_states_subset.py \\
        --jsonl_path /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag>/combined_generations.jsonl \\
        --csv_path   /home/kalashkala/recovery-gaps-experiment/data/sampled_1400_uncertainty_run_llama_sciq_combined_llm_verdict.csv \\
        --out_dir    /data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag>/subset \\
        --model_name meta-llama/Llama-3.1-8B-Instruct \\
        --batch_size 128 \\
    > extract_hidden_states_subset.log 2>&1 &
"""

import argparse
import gc
import logging
import os
import sys

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.join(_HERE, '..', '..', 'semantic_uncertainty', 'semantic_uncertainty')
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from uncertainty.utils import utils
from generate_answers_combined import _extract_embeddings_from_token_ids

utils.setup_logger()


def _filter_jsonl_to_subset(jsonl_path: str, keep_ids: set) -> str:
    """Write a temporary JSONL containing only records whose key is in keep_ids.

    Returns the path to the filtered JSONL (written alongside the original with
    a '_subset' suffix so the caller can clean it up if desired).
    """
    import json
    import tempfile

    filtered_path = jsonl_path.replace('.jsonl', '_subset_tmp.jsonl')
    kept = 0
    skipped = 0
    with open(jsonl_path) as src, open(filtered_path, 'w') as dst:
        for line in src:
            record = json.loads(line)
            matching = {k: v for k, v in record.items() if k in keep_ids}
            if matching:
                dst.write(json.dumps(matching) + '\n')
                kept += len(matching)
            else:
                skipped += len(record)
    logging.info('Filtered JSONL: kept %d records, skipped %d records → %s', kept, skipped, filtered_path)
    return filtered_path


def main():
    parser = argparse.ArgumentParser(
        description='Extract hidden-state embeddings for a subset of rows defined by a sampled CSV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--jsonl_path', type=str, required=True,
        help='Path to combined_generations.jsonl produced by generate_answers_combined.py.',
    )
    parser.add_argument(
        '--csv_path', type=str, required=True,
        help='Path to sampled CSV. Must have an "id" column matching JSONL record keys.',
    )
    parser.add_argument(
        '--out_dir', type=str, required=True,
        help='Directory where combined_generations.pkl and validation_generations.pkl are written.',
    )
    parser.add_argument(
        '--model_name', type=str, required=True,
        help='HuggingFace model identifier used during generation.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Number of sequences per GPU forward pass. Auto-halved on OOM.',
    )
    parser.add_argument(
        '--model_max_new_tokens', type=int, default=15,
        help='Must match the value used during generation.',
    )

    args = parser.parse_args()

    for path, label in [(args.jsonl_path, 'JSONL'), (args.csv_path, 'CSV')]:
        if not os.path.isfile(path):
            logging.error('%s file not found: %s', label, path)
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ['SU_LOCAL_RUN_DIR'] = args.out_dir

    df = pd.read_csv(args.csv_path)
    if 'id' not in df.columns:
        logging.error("CSV has no 'id' column. Found columns: %s", list(df.columns))
        sys.exit(1)

    keep_ids = set(df['id'].dropna().astype(str).tolist())
    logging.info('Subset size from CSV: %d unique IDs', len(keep_ids))

    filtered_jsonl = _filter_jsonl_to_subset(args.jsonl_path, keep_ids)

    try:
        logging.info('Loading model: %s', args.model_name)
        model = utils.init_model(args)

        logging.info('Extracting embeddings (batch_size=%d)', args.batch_size)
        generations = _extract_embeddings_from_token_ids(model, filtered_jsonl, bsz=args.batch_size)

        logging.info('Unloading model.')
        del model
        gc.collect()
        torch.cuda.empty_cache()

        logging.info('Saving PKL files to: %s', args.out_dir)
        utils.save(generations, 'combined_generations.pkl')
        utils.save(generations, 'validation_generations.pkl')
        logging.info('Done. Written:')
        logging.info('  %s/combined_generations.pkl', args.out_dir)
        logging.info('  %s/validation_generations.pkl', args.out_dir)
    finally:
        if os.path.isfile(filtered_jsonl):
            os.remove(filtered_jsonl)
            logging.info('Removed temporary filtered JSONL: %s', filtered_jsonl)


if __name__ == '__main__':
    main()
