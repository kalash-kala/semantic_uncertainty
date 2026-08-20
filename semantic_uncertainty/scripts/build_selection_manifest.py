#!/usr/bin/env python3
"""Build a row-selection manifest so several machines evaluate identical rows.

`generate_answers_combined.py --selection_manifest FILE` reads this file and
selects those exact rows by example id, bypassing the RNG. Generate the manifest
once, commit it, and every server runs the same row set by construction instead
of relying on `random.sample` happening to replay identically.

This does NOT load a model, so it runs in seconds on a CPU-only box.

Usage:
    python scripts/build_selection_manifest.py \
        --dataset nq --num_samples 50000 --random_seed 10 \
        --out manifests/nq_50k_seed10.json

The pool is built by importing the pipeline's own `_build_eval_pool`, so this
cannot drift from what a real run would construct. The few-shot / p_true
exemplar draws are replayed with a dedicated RNG, matching what
generate_answers_combined does whenever --selection_manifest is set.
"""

import argparse
import json
import os
import random
import sys

# Import from the pipeline itself rather than reimplementing pool construction.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_answers_combined import _build_eval_pool  # noqa: E402
from uncertainty.data.data_utils import load_ds  # noqa: E402
from uncertainty.utils import utils  # noqa: E402


class _Args:
    """Minimal stand-in for the argparse namespace `_build_eval_pool` reads."""

    def __init__(self, answerable_only):
        self.answerable_only = answerable_only


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--num_samples', type=int, required=True)
    p.add_argument('--out', required=True, help='Path to write the manifest JSON')
    p.add_argument('--random_seed', type=int, default=10)
    p.add_argument('--num_few_shot', type=int, default=5)
    p.add_argument('--p_true_num_fewshot', type=int, default=20)
    p.add_argument('--compute_p_true', default=True,
                   action=argparse.BooleanOptionalAction)
    p.add_argument('--answerable_only', default=False,
                   action=argparse.BooleanOptionalAction)
    args = p.parse_args()

    train_dataset, validation_dataset = load_ds(args.dataset, seed=args.random_seed)
    print(f'Loaded {args.dataset}: train={len(train_dataset)} '
          f'validation={len(validation_dataset)}')

    # Replay the exemplar draws that remove rows from the pool. Uses a dedicated
    # RNG exactly as generate_answers_combined does under --selection_manifest,
    # so the excluded set here matches the excluded set there.
    exemplar_rng = random.Random(args.random_seed)
    prompt_answerable_indices, _ = utils.split_dataset(train_dataset)
    prompt_indices = exemplar_rng.sample(prompt_answerable_indices, args.num_few_shot)

    excluded = set(prompt_indices)
    if args.compute_p_true:
        remaining = list(set(prompt_answerable_indices) - set(prompt_indices))
        p_true_indices = exemplar_rng.sample(remaining, args.p_true_num_fewshot)
        excluded.update(p_true_indices)
    print(f'Excluded {len(excluded)} train rows (few-shot + p_true exemplars)')

    combined_examples = _build_eval_pool(
        target_train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        args=_Args(args.answerable_only),
        excluded_train_answerable_indices=excluded,
    )
    pool_size = len(combined_examples)
    print(f'Pool size: {pool_size}')

    num_selected = min(args.num_samples, pool_size)
    if num_selected != args.num_samples:
        print(f'WARNING: requested {args.num_samples} but pool holds only '
              f'{pool_size}; taking all of them.')

    # The manifest defines the row set, so any deterministic draw is fine; a
    # dedicated RNG keeps it reproducible from this script alone.
    selection_rng = random.Random(args.random_seed)
    selected_indices = selection_rng.sample(range(pool_size), num_selected)
    ids = [str(combined_examples[i]['example']['id']) for i in selected_indices]

    if len(set(ids)) != len(ids):
        raise ValueError(
            f'Manifest ids are not unique ({len(ids) - len(set(ids))} duplicates). '
            'Selecting by id would be ambiguous.'
        )

    manifest = {
        'dataset': args.dataset,
        'random_seed': args.random_seed,
        'num_samples': args.num_samples,
        'num_few_shot': args.num_few_shot,
        'p_true_num_fewshot': args.p_true_num_fewshot,
        'compute_p_true': bool(args.compute_p_true),
        'answerable_only': bool(args.answerable_only),
        'pool_size': pool_size,
        'ids': ids,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)

    print(f'Wrote {len(ids)} ids -> {args.out}')
    print(f'First 3 ids: {ids[:3]}')


if __name__ == '__main__':
    main()
