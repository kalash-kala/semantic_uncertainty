"""Sample answers from LLMs on a combined train+validation QA pool.

This variant is based on the attached generate_answers-style script, but it:
1. Builds the few-shot prompt exactly as before.
2. Combines the target dataset's train and validation splits for generation.
3. Runs one low-temperature greedy generation plus N high-temperature samples.
4. Saves the merged pool in a validation-compatible format so the existing
   compute_uncertainty_measures.py pipeline can run without changes.
"""

import gc
import json
import logging
import os
import pickle
import random
import shutil
import time
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils
from compute_uncertainty_measures import main as main_compute


utils.setup_logger()

# Example command qwen + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_qwen_triviaqa_combined.log 2>&1 &
# Example command llama + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_llama_triviaqa_combined.log 2>&1 &
# Example command Gemma + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_gemma_triviaqa_combined.log 2>&1 &
# Example command mistral + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_mistral_triviaqa_combined.log 2>&1 &

# Example command gemma + trivia_qa + A100 GPU: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_gemma_triviaqa_combined.log 2>&1 &
# Example command Llama + trivia_qa + A100 GPU: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_llama_triviaqa_combined.log 2>&1 &
# Example command mistral + trivia_qa + A100 GPU: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_mistral_triviaqa_combined.log 2>&1 &
# Example command Qwen + trivia_qa + A100 GPU: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_qwen_triviaqa_combined.log 2>&1 &

# Example command qwen + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_qwen_svamp_combined.log 2>&1 &
# Example command gemma + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_gemma_svamp_combined.log 2>&1 &
# Example command Llama + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_llama_svamp_combined.log 2>&1 &
# Example command mistral + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_mistral_svamp_combined.log 2>&1 &

# Example command qwen + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_qwen_sciq_combined.log 2>&1 &
# Example command gemma + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_gemma_sciq_combined.log 2>&1 &
# Example command Llama + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_llama_sciq_combined.log 2>&1 &
# Example command mistral + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_mistral_sciq_combined.log 2>&1 &

# Example command qwen + gsm8k: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_qwen_gsm8k_combined.log 2>&1 &
# Example command gemma + gsm8k: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset gsm8k --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_gemma_gsm8k_combined.log 2>&1 &
# Example command Llama + gsm8k: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset gsm8k --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_llama_gsm8k_combined.log 2>&1 &
# Example command mistral + gsm8k: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset gsm8k --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_mistral_gsm8k_combined.log 2>&1 &

# Example command qwen + trivia_qa_nocontext: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset trivia_qa_nocontext --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_qwen_trivia-qa-nocontext_combined.log 2>&1 &
# Example command gemma + trivia_qa_nocontext: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset trivia_qa_nocontext --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_gemma_trivia-qa-nocontext_combined.log 2>&1 &
# Example command Llama + trivia_qa_nocontext: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset trivia_qa_nocontext --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_llama_trivia-qa-nocontext_combined.log 2>&1 &
# Example command mistral + trivia_qa_nocontext: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset trivia_qa_nocontext --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --generation_batch_size 64 --compute_uncertainties > uncertainty_run_mistral_trivia-qa-nocontext_combined.log 2>&1 &

def _extract_embeddings_from_token_ids(model, jsonl_path, bsz=8):
    """Read JSONL (with token IDs), extract embeddings in batches, return full generations dict.

    Each JSONL record stores 'generated_ids' (list[int]) instead of float embedding
    tensors.  This function reconstructs embeddings via a single forward pass per
    batch and returns a dict compatible with the downstream PKL format.
    """
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            records.update(json.loads(line))

    all_ids = list(records.keys())
    logging.info('Extracting embeddings for %d records (bsz=%d)…', len(all_ids), bsz)

    if not all_ids:
        logging.error('JSONL file is empty: %s', jsonl_path)
        raise ValueError(f'JSONL file has no records: {jsonl_path}')

    # Check first sample to see what format we have
    first_rec = records[all_ids[0]]
    first_mla = first_rec.get('most_likely_answer', {})
    has_generated_ids = 'generated_ids' in first_mla
    has_embedding = 'embedding' in first_mla
    logging.info('First sample: has_generated_ids=%s, has_embedding=%s', has_generated_ids, has_embedding)

    # Flatten into (record_id, resp_idx, sequence_info).
    # resp_idx == -1  →  most_likely_answer (greedy)
    # resp_idx >= 0   →  sampled responses
    flat = []
    for eid in all_ids:
        rec = records[eid]
        prompt_ids = rec.get('prompt_token_ids') or []

        mla = rec.get('most_likely_answer', {})
        if 'generated_ids' in mla:
            flat.append((eid, -1, {'prompt_ids': prompt_ids, 'generated_ids': mla['generated_ids']}))
        else:
            flat.append((eid, -1, None))  # old format — embedding already present

        for j, resp in enumerate(rec.get('responses', [])):
            gen_ids = resp[2] if (isinstance(resp, (list, tuple)) and len(resp) > 2
                                  and isinstance(resp[2], list)
                                  and (not resp[2] or isinstance(resp[2][0], int))) else None
            if gen_ids is not None:
                flat.append((eid, j, {'prompt_ids': prompt_ids, 'generated_ids': gen_ids}))
            else:
                flat.append((eid, j, None))  # old format

    # Only extract embeddings for greedy (most_likely_answer) sequences.
    # all_hidden stores 33 layers × 3 positions × 4096 floats ≈ 1.62 MB per greedy sequence.
    # Sampled responses (ridx >= 0) get None embeddings intentionally — not needed for analysis.
    need_greedy = [(eid, ridx, info) for eid, ridx, info in flat if info is not None and ridx == -1]

    SHARD_INTERVAL = 200  # batches between flushes
    shard_dirs = []  # track shard dir for final cleanup

    emb_map = {}

    for group_list, shard_dir_name, do_all_hidden in [
        (need_greedy, 'emb_shards_greedy', True),
    ]:
        shard_dir = os.path.join(os.path.dirname(jsonl_path), shard_dir_name)
        shard_dirs.append(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)

        existing_shards = sorted(f for f in os.listdir(shard_dir) if f.endswith('.pkl'))
        covered_batch_indices = set()
        for shard_name in existing_shards:
            shard_path = os.path.join(shard_dir, shard_name)
            with open(shard_path, 'rb') as fh:
                shard_data = pickle.load(fh)
            covered_batch_indices.update(shard_data.get('batch_indices', []))
        if covered_batch_indices:
            logging.info('Resuming %s extraction: %d batches already done via %d shards.',
                         shard_dir_name, len(covered_batch_indices), len(existing_shards))

        group_emb_map = {}
        shard_idx = len(existing_shards)
        all_batch_indices = list(range(0, len(group_list), bsz))
        pending_batch_indices = []

        label = 'greedy (all_hidden)' if do_all_hidden else 'sampled (scalars only)'
        logging.info('Extracting embeddings for %d sequences [%s] (bsz=%d)…',
                     len(group_list), label, bsz)

        for batch_num, batch_start in enumerate(tqdm(all_batch_indices, desc=f'Extracting embeddings [{label}]')):
            if batch_num in covered_batch_indices:
                continue

            batch = group_list[batch_start: batch_start + bsz]
            keys = [(eid, ridx) for eid, ridx, _ in batch]
            seq_infos = [info for _, _, info in batch]
            try:
                embs = model.extract_embeddings_batch(seq_infos, extract_all_hidden=do_all_hidden)
                for key, emb in zip(keys, embs):
                    group_emb_map[key] = emb
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower():
                    logging.warning('OOM during embedding batch at %d. Reducing batch size and retrying…', batch_start)
                    reduced_bsz = max(2, bsz // 2)
                    for sub_start in range(0, len(batch), reduced_bsz):
                        sub_batch = batch[sub_start: sub_start + reduced_bsz]
                        sub_keys = keys[sub_start: sub_start + reduced_bsz]
                        sub_infos = [info for _, _, info in sub_batch]
                        try:
                            embs = model.extract_embeddings_batch(sub_infos, extract_all_hidden=do_all_hidden)
                            for key, emb in zip(sub_keys, embs):
                                group_emb_map[key] = emb
                        except Exception as sub_exc:
                            logging.error('Sub-batch extraction failed: %s', sub_exc)
                            for key in sub_keys:
                                group_emb_map[key] = {'first_answer': None, 'last_prompt': None, 'last_token': None, 'all_hidden': None}
                else:
                    logging.error('Embedding extraction failed at batch %d: %s', batch_start, exc, exc_info=True)
                    for key in keys:
                        group_emb_map[key] = {'first_answer': None, 'last_prompt': None, 'last_token': None, 'all_hidden': None}
            except Exception as exc:
                logging.error('Embedding extraction failed at batch %d: %s', batch_start, exc, exc_info=True)
                for key in keys:
                    group_emb_map[key] = {'first_answer': None, 'last_prompt': None, 'last_token': None, 'all_hidden': None}

            pending_batch_indices.append(batch_num)

            if len(pending_batch_indices) >= SHARD_INTERVAL:
                shard_path = os.path.join(shard_dir, f'shard_{shard_idx:06d}.pkl')
                with open(shard_path, 'wb') as fh:
                    pickle.dump({'batch_indices': pending_batch_indices, 'emb_map': group_emb_map}, fh)
                logging.info('Flushed shard %d (%d batches) to %s', shard_idx, len(pending_batch_indices), shard_path)
                group_emb_map = {}
                pending_batch_indices = []
                shard_idx += 1

        if group_emb_map or pending_batch_indices:
            shard_path = os.path.join(shard_dir, f'shard_{shard_idx:06d}.pkl')
            with open(shard_path, 'wb') as fh:
                pickle.dump({'batch_indices': pending_batch_indices, 'emb_map': group_emb_map}, fh)
            logging.info('Flushed final shard %d to %s', shard_idx, shard_path)

    # Merge all shards from both greedy and sampled groups into a single emb_map.
    emb_map = {}
    for shard_dir in shard_dirs:
        logging.info('Merging shards from %s…', shard_dir)
        for shard_name in sorted(os.listdir(shard_dir)):
            if not shard_name.endswith('.pkl'):
                continue
            shard_path = os.path.join(shard_dir, shard_name)
            with open(shard_path, 'rb') as fh:
                shard_data = pickle.load(fh)
            emb_map.update(shard_data['emb_map'])
            del shard_data

    logging.info('Extraction complete: %d embeddings extracted, %d records total',
                 len(emb_map), len(all_ids))

    # Reconstruct full records with embedding dicts.
    generations = {}
    missing_embeddings = []

    for eid in all_ids:
        rec = dict(records[eid])

        mla = dict(rec.get('most_likely_answer', {}))
        if (eid, -1) in emb_map:
            mla.pop('generated_ids', None)
            mla['embedding'] = emb_map[(eid, -1)]
        elif 'generated_ids' in mla:
            # Sample has generated_ids but no extracted embedding — extraction incomplete
            missing_embeddings.append((eid, -1))
            # Use None embedding as fallback to avoid downstream KeyError
            mla.pop('generated_ids', None)
            mla['embedding'] = {'first_answer': None, 'last_prompt': None, 'last_token': None, 'all_hidden': None}
        rec['most_likely_answer'] = mla

        new_responses = []
        for j, resp in enumerate(rec.get('responses', [])):
            resp = list(resp)
            if (eid, j) in emb_map:
                resp[2] = emb_map[(eid, j)]  # replace generated_ids with embedding dict
            elif isinstance(resp[2], list) and (not resp[2] or isinstance(resp[2][0], int)):
                # Sampled responses: embeddings intentionally not extracted.
                resp[2] = {'first_answer': None, 'last_prompt': None, 'last_token': None, 'all_hidden': None}
            new_responses.append(tuple(resp))
        rec['responses'] = new_responses

        rec.pop('prompt_token_ids', None)
        generations[eid] = rec

    if missing_embeddings:
        logging.warning('Embedding extraction incomplete: %d samples missing embeddings. '
                       'Check logs for extraction errors.', len(missing_embeddings))

    # Validation: ensure all samples have embedding dicts
    missing_keys = []
    for eid, rec in generations.items():
        mla = rec.get('most_likely_answer', {})
        if 'embedding' not in mla:
            missing_keys.append(eid)
            # Add fallback None embedding to avoid downstream KeyError
            mla['embedding'] = {'first_answer': None, 'last_prompt': None, 'last_token': None, 'all_hidden': None}
            rec['most_likely_answer'] = mla

    if missing_keys:
        logging.warning('Final fallback: added None embeddings to %d samples with missing embeddings. '
                       'This usually indicates extraction errors (OOM, crashes, etc.). '
                       'Downstream analysis will use None for these samples.', len(missing_keys))
    else:
        logging.info('Embeddings successfully extracted for all %d samples.', len(generations))

    # Clean up both shard directories now that the final generations dict is assembled.
    for shard_dir in shard_dirs:
        shutil.rmtree(shard_dir, ignore_errors=True)

    return generations


def _build_eval_pool(
    target_train_dataset,
    validation_dataset,
    args,
    excluded_train_answerable_indices=None,
):
    """Create a single evaluation pool by merging train and validation samples.

    `excluded_train_answerable_indices` should contain prompt/p_true examples that
    must not be evaluated when the prompt is built from the target train split.
    """
    excluded_train_answerable_indices = set(excluded_train_answerable_indices or [])

    train_answerable_indices, train_unanswerable_indices = utils.split_dataset(target_train_dataset)
    val_answerable_indices, val_unanswerable_indices = utils.split_dataset(validation_dataset)

    if args.answerable_only:
        train_indices = [
            idx for idx in train_answerable_indices
            if idx not in excluded_train_answerable_indices
        ]
        val_indices = list(val_answerable_indices)
    else:
        train_indices = [
            idx for idx in train_answerable_indices
            if idx not in excluded_train_answerable_indices
        ] + list(train_unanswerable_indices)
        val_indices = list(val_answerable_indices) + list(val_unanswerable_indices)

    combined_examples: List[Dict[str, Any]] = []
    for idx in train_indices:
        combined_examples.append({
            'source_split': 'train',
            'source_index': idx,
            'example': target_train_dataset[idx],
        })

    for idx in val_indices:
        combined_examples.append({
            'source_split': 'validation',
            'source_index': idx,
            'example': validation_dataset[idx],
        })

    return combined_examples



def main(args):
    # Setup dataset-specific defaults.
    if args.dataset == 'svamp' and not args.use_context:
        logging.info('Forcing `use_context=True` for svamp dataset.')
        args.use_context = True
    elif args.dataset == 'squad' and not args.answerable_only:
        logging.info('Forcing `answerable_only=True` for squad dataset.')
        args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    scratch_dir = os.getenv('SCRATCH_DIR', '/data/kalashkala')
    if not os.path.exists(f"{scratch_dir}/semantic_uncertainty_data/uncertainty"):
        os.makedirs(f"{scratch_dir}/semantic_uncertainty_data/uncertainty")

    safe_model_name = args.model_name.replace('/', '__').replace(' ', '_')
    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    run_tag = f"{args.dataset}__{safe_model_name}__seed{args.random_seed}__pid{os.getpid()}__{run_stamp}"
    out_dir = os.path.join(f"{scratch_dir}/semantic_uncertainty_data/uncertainty", run_tag)
    os.makedirs(out_dir, exist_ok=True)
    os.environ['SU_LOCAL_RUN_DIR'] = out_dir
    logging.info('Using run-specific output directory: %s', out_dir)
    experiment_details['out_dir'] = out_dir
    experiment_details['run_tag'] = run_tag

    metric = utils.get_metric(args.metric)

    # Load the target dataset whose train+validation splits will be merged.
    target_train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)

    # Prompt few-shot examples default to the target train split, unless an OOD
    # prompt dataset is explicitly requested.
    prompt_dataset = target_train_dataset
    prompt_dataset_name = args.dataset
    prompt_from_target_train = True

    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        prompt_dataset, _ = load_ds(
            args.ood_train_dataset,
            add_options=args.use_mc_options,
            seed=args.random_seed,
        )
        prompt_dataset_name = args.ood_train_dataset
        prompt_from_target_train = False

    if not isinstance(prompt_dataset, list):
        logging.info('Prompt dataset (%s): %s', prompt_dataset_name, prompt_dataset)

    prompt_answerable_indices, _ = utils.split_dataset(prompt_dataset)
    if len(prompt_answerable_indices) < args.num_few_shot:
        raise ValueError(
            f'Not enough answerable examples ({len(prompt_answerable_indices)}) '
            f'to sample num_few_shot={args.num_few_shot} from {prompt_dataset_name}.')

    prompt_indices = random.sample(prompt_answerable_indices, args.num_few_shot)
    experiment_details['prompt_dataset'] = prompt_dataset_name
    experiment_details['prompt_indices'] = prompt_indices

    excluded_target_train_answerable_indices = set()
    if prompt_from_target_train:
        excluded_target_train_answerable_indices.update(prompt_indices)

    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    brief_arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        prompt_dataset, prompt_indices, BRIEF, brief_arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    model = utils.init_model(args)

    if args.compute_p_true:
        logging.info(80 * '#')
        logging.info('Constructing few-shot prompt for p_true.')

        remaining_prompt_answerable = list(set(prompt_answerable_indices) - set(prompt_indices))
        if len(remaining_prompt_answerable) < args.p_true_num_fewshot:
            raise ValueError(
                'Not enough answerable examples left for p_true few-shot prompt: '
                f'{len(remaining_prompt_answerable)} available, '
                f'{args.p_true_num_fewshot} requested.')

        p_true_indices = random.sample(remaining_prompt_answerable, args.p_true_num_fewshot)
        if prompt_from_target_train:
            excluded_target_train_answerable_indices.update(p_true_indices)

        p_true_few_shot_prompt, p_true_responses, _ = p_true_utils.construct_few_shot_prompt(
            model=model,
            dataset=prompt_dataset,
            indices=p_true_indices,
            prompt=prompt,
            brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt,
            num_generations=args.num_generations,
            metric=metric,
        )
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80 * '#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80 * '#')
    else:
        p_true_few_shot_prompt = None

    # Build the merged evaluation pool.
    combined_examples = _build_eval_pool(
        target_train_dataset=target_train_dataset,
        validation_dataset=validation_dataset,
        args=args,
        excluded_train_answerable_indices=excluded_target_train_answerable_indices,
    )

    if not combined_examples:
        raise ValueError('Combined evaluation pool is empty.')

    logging.info(80 * '=')
    logging.info('Generating answers on merged train+validation pool.')
    logging.info('Combined pool size: %d', len(combined_examples))
    logging.info('Train prompt source: %s', prompt_dataset_name)
    logging.info(80 * '=')

    accuracies, generations, results_dict, p_trues = [], {}, {}, []
    jsonl_path = os.path.join(out_dir, 'combined_generations.jsonl')

    num_selected = min(args.num_samples, len(combined_examples))

    if num_selected != args.num_samples:
        logging.warning(
            'Not enough samples in combined pool. Using all %d samples.',
            len(combined_examples),
        )

    selected_indices = random.sample(range(len(combined_examples)), num_selected)
    experiment_details['combined'] = {
        'indices': selected_indices,
        'num_total_examples': len(combined_examples),
        'num_selected_examples': num_selected,
        'excluded_target_train_answerable_indices': sorted(excluded_target_train_answerable_indices),
    }

    # Validation-compatible alias for downstream uncertainty script compatibility.
    experiment_details['validation'] = experiment_details['combined']

    # Save early so selected_indices survive a crash and resume can find them.
    utils.save(experiment_details, 'experiment_details.pkl')

    if args.num_samples > len(combined_examples):
        logging.warning(
            'Not enough samples in combined pool. Using all %d samples.',
            len(combined_examples),
        )

    # --- Resume from a previous incomplete run ---
    processed_ids = set()
    jsonl_mode = 'w'
    resume_dir = getattr(args, 'resume_dir', None)
    if resume_dir and os.path.exists(resume_dir):
        import pickle as _pkl
        resume_details_pkl = os.path.join(resume_dir, 'experiment_details.pkl')
        resume_jsonl_path = os.path.join(resume_dir, 'combined_generations.jsonl')

        if os.path.exists(resume_details_pkl):
            with open(resume_details_pkl, 'rb') as _f:
                _prev = _pkl.load(_f)
            selected_indices = _prev['combined']['indices']
            num_selected = len(selected_indices)
            logging.info('Resume: loaded %d selected_indices from prior run', num_selected)

        if os.path.exists(resume_jsonl_path):
            with open(resume_jsonl_path) as _f:
                for _line in _f:
                    _rec = json.loads(_line)
                    for _eid, _data in _rec.items():
                        processed_ids.add(_eid)
                        accuracies.append(_data.get('most_likely_answer', {}).get('accuracy', 0.0))
                        if 'p_true' in _data:
                            p_trues.append(_data['p_true'])
            logging.info('Resume: %d samples already done, skipping them', len(processed_ids))

        out_dir = resume_dir
        os.environ['SU_LOCAL_RUN_DIR'] = out_dir
        jsonl_path = resume_jsonl_path
        jsonl_mode = 'a'

    # Exclude already-processed samples so we only iterate what remains.
    if processed_ids:
        remaining_indices = [
            i for i in selected_indices
            if f"{combined_examples[i]['source_split']}::{combined_examples[i]['source_index']}"
            not in processed_ids
        ]
        logging.info('Resume: %d remaining of %d total', len(remaining_indices), num_selected)
    else:
        remaining_indices = selected_indices

    bsz = args.generation_batch_size

    with open(jsonl_path, jsonl_mode) as jsonl_file:
        for batch_start in tqdm(range(0, len(remaining_indices), bsz)):
            batch_indices = remaining_indices[batch_start: batch_start + bsz]
            batch_wrapped = [combined_examples[idx] for idx in batch_indices]

            if (batch_start // bsz + 1) % max(1, 10 // bsz) == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # Build per-question metadata and prompts.
            batch_meta = []
            local_prompts = []
            for wrapped_example in batch_wrapped:
                example = wrapped_example['example']
                source_split = wrapped_example['source_split']
                source_index = wrapped_example['source_index']
                question, context = example['question'], example['context']
                example_id = f"{source_split}::{source_index}"
                correct_answer = example['answers']['text']

                generations[example_id] = {
                    'question': question,
                    'context': context,
                    'source_split': source_split,
                    'source_index': source_index,
                    'original_id': example['id'],
                }

                current_input = make_prompt(
                    context, question, None, BRIEF, args.brief_always and args.enable_brief)
                local_prompts.append(prompt + current_input)
                logging.info('Current input: '.ljust(15) + current_input)

                batch_meta.append({
                    'example': example,
                    'example_id': example_id,
                    'question': question,
                    'context': context,
                    'source_split': source_split,
                    'source_index': source_index,
                    'correct_answer': correct_answer,
                })

            # Combined A+B: greedy pass for all B questions, then one batched
            # high-temp call generating num_generations samples per question.
            # Token IDs are stored instead of hidden-state embeddings to keep the
            # JSONL tiny; embeddings are extracted in a batched forward pass after
            # the generation loop via _extract_embeddings_from_token_ids().
            greedy_results = model.predict_batch_questions(
                local_prompts, temperature=0.0, do_sample=False, return_token_ids=True)
            sampled_results = model.predict_batch_questions(
                local_prompts, temperature=1.0, do_sample=True,
                num_return_sequences=args.num_generations, return_token_ids=True)

            it_base = batch_start + 1
            for i, meta in enumerate(batch_meta):
                it = it_base + i
                example = meta['example']
                example_id = meta['example_id']
                question = meta['question']
                context = meta['context']
                source_split = meta['source_split']
                source_index = meta['source_index']
                correct_answer = meta['correct_answer']

                predicted_answer, token_log_likelihoods, greedy_token_info = greedy_results[i]
                # greedy_token_info = {'generated_ids': list[int], 'prompt_ids': list[int]}

                acc = metric(predicted_answer, example, model) if correct_answer else 0.0

                logging.info('Iteration ' + str(it) + ':  ' + 80 * '#')
                if args.use_context:
                    logging.info('context: '.ljust(15) + str(context))
                logging.info('source split: '.ljust(15) + source_split)
                logging.info('source index: '.ljust(15) + str(source_index))
                logging.info('question: '.ljust(15) + question)
                logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                logging.info('correct answer: '.ljust(15) + str(correct_answer))
                logging.info('accuracy: '.ljust(15) + str(acc))

                accuracies.append(acc)
                most_likely_answer_dict = {
                    'response': predicted_answer,
                    'token_log_likelihoods': token_log_likelihoods,
                    'generated_ids': greedy_token_info.get('generated_ids', []),
                    'accuracy': acc,
                }
                generations[example_id].update({
                    'prompt_token_ids': greedy_token_info.get('prompt_ids', []),
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': utils.get_reference(example),
                })

                full_responses = []
                for sample_idx, (pred_ans, lls, tok_info) in enumerate(sampled_results[i], start=1):
                    acc_s = (metric(pred_ans, example, model)
                             if (correct_answer and args.compute_accuracy_at_all_temps) else 0.0)
                    logging.info('high-t prediction '.ljust(15) + str(sample_idx) + ' : ' + pred_ans)
                    gen_ids = tok_info.get('generated_ids', []) if tok_info else []
                    full_responses.append((pred_ans, lls, gen_ids, acc_s))

                generations[example_id]['responses'] = full_responses

                # All fields are now plain Python types (lists of int/float/str) —
                # no tensor-to-list conversion needed for JSON serialization.
                sample_data = generations[example_id]

                if args.compute_p_true:
                    p_true = p_true_utils.calculate_p_true(
                        model,
                        question,
                        most_likely_answer_dict['response'],
                        [r[0] for r in full_responses],
                        p_true_few_shot_prompt,
                        hint=args.p_true_hint,
                    )
                    p_trues.append(p_true)
                    logging.info('p_true: %s', p_true)
                    sample_data['p_true'] = p_true

                jsonl_file.write(json.dumps({example_id: sample_data}) + '\n')
                jsonl_file.flush()
                del generations[example_id]

    # Extract embeddings from stored token IDs and build the generations dict.
    # The JSONL stores token IDs instead of float tensors (much smaller files);
    # embeddings are computed here in a single batched forward pass per batch.
    logging.info('Generation loop complete. Extracting embeddings from token IDs…')
    emb_bsz = 128  # Large batch (no KV cache growth), with auto-retry to smaller sizes on OOM
    generations = _extract_embeddings_from_token_ids(model, jsonl_path, bsz=emb_bsz)
    utils.save(generations, 'combined_generations.pkl')

    # Save validation-compatible copies so the existing uncertainty computation
    # script can run unchanged.
    # NOTE: Do NOT copy the raw JSONL here — it has 'generated_ids' (token IDs),
    # not 'embedding' dicts. compute_uncertainty_measures prefers JSONL over PKL,
    # so we let it fall back to the PKL below which has proper embeddings.
    utils.save(generations, 'validation_generations.pkl')

    accuracy = float(np.mean(accuracies)) if accuracies else float('nan')
    print(f'Overall combined split accuracy: {accuracy}')

    if args.compute_p_true:
        results_dict['uncertainty_measures'] = {
            'p_false': [1 - p for p in p_trues],
            'p_false_fixed': [1 - np.exp(p) for p in p_trues],
        }
    utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model


if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to a previous incomplete run directory to resume from.')
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info('STARTING `generate_answers_combined`!')
    main(args)
    logging.info('FINISHED `generate_answers_combined`!')

    if args.compute_uncertainties:
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
