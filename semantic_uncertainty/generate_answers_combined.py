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

# Example command qwen + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_qwen_triviaqa_combined.log 2>&1 &
# Example command llama + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_llama_triviaqa_combined.log 2>&1 &
# Example command Gemma + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_gemma_triviaqa_combined.log 2>&1 &
# Example command mistral + trivia_qa: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_mistral_triviaqa_combined.log 2>&1 &

# Example command gemma + trivia_qa + CUDA_VISIBLE_DEVICES=1: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface CUDA_VISIBLE_DEVICES=1 python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_gemma_triviaqa_combined.log 2>&1 &
# Example command Llama + trivia_qa + CUDA_VISIBLE_DEVICES=1: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface CUDA_VISIBLE_DEVICES=1 python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_llama_triviaqa_combined.log 2>&1 &
# Example command mistral + trivia_qa + CUDA_VISIBLE_DEVICES=1: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface CUDA_VISIBLE_DEVICES=1 python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_mistral_triviaqa_combined.log 2>&1 &
# Example command Qwen + trivia_qa + CUDA_VISIBLE_DEVICES=1: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface CUDA_VISIBLE_DEVICES=1 python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset trivia_qa --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_qwen_triviaqa_combined.log 2>&1 &

# Example command qwen + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_qwen_svamp_combined.log 2>&1 &
# Example command gemma + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_gemma_svamp_combined.log 2>&1 &
# Example command Llama + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_llama_svamp_combined.log 2>&1 &
# Example command mistral + svamp: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset svamp --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_mistral_svamp_combined.log 2>&1 &

# Example command qwen + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_qwen_sciq_combined.log 2>&1 &
# Example command gemma + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name google/gemma-3-12b-it --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_gemma_sciq_combined.log 2>&1 &
# Example command Llama + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_llama_sciq_combined.log 2>&1 &
# Example command mistral + sciq: nohup conda run --no-capture-output -n semantic_uncertainty env HF_HOME=/data/.cache/huggingface python generate_answers_combined.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset sciq --model_max_new_tokens 15 --num_generations 10 --num_samples 100 --compute_uncertainties > uncertainty_run_mistral_sciq_combined.log 2>&1 &

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

    if args.num_samples > len(combined_examples):
        logging.warning(
            'Not enough samples in combined pool. Using all %d samples.',
            len(combined_examples),
        )

    with open(jsonl_path, 'w') as jsonl_file:
        for it, combined_idx in enumerate(tqdm(selected_indices), start=1):
            if ((it + 1) % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()

            wrapped_example = combined_examples[combined_idx]
            example = wrapped_example['example']
            source_split = wrapped_example['source_split']
            source_index = wrapped_example['source_index']

            # question, context = example['question'], example['context']
            # example_id = f"{source_split}::{example['id']}"
            # correct_answer = example['answers']['text']

            # generations[example_id] = {
            #     'question': question,
            #     'context': context,
            #     'source_split': source_split,
            #     'source_index': source_index,
            #     'original_id': example['id'],
            # }

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
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []
            num_generations = args.num_generations + 1  # 1 low-t + N high-t

            for i in range(num_generations):
                if i == 0:
                    temperature = 0.0
                    do_sample = False
                else:
                    temperature = 1.0
                    do_sample = True

                predicted_answer, token_log_likelihoods, embedding = model.predict(
                    local_prompt,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                embedding = embedding.cpu() if embedding is not None else None

                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0

                if i == 0:
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
                        'embedding': embedding,
                        'accuracy': acc,
                    }
                    generations[example_id].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example),
                    })
                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    full_responses.append((predicted_answer, token_log_likelihoods, embedding, acc))

            generations[example_id]['responses'] = full_responses

            sample_data = generations[example_id]
            if sample_data['most_likely_answer'].get('embedding') is not None:
                if hasattr(sample_data['most_likely_answer']['embedding'], 'tolist'):
                    sample_data['most_likely_answer']['embedding'] = (
                        sample_data['most_likely_answer']['embedding'].tolist())

            clean_responses = []
            for ans, likelihoods, emb, acc in full_responses:
                emb_list = emb.tolist() if hasattr(emb, 'tolist') else emb
                clean_responses.append((ans, likelihoods, emb_list, acc))
            sample_data['responses'] = clean_responses

            jsonl_file.write(json.dumps({example_id: sample_data}) + '\n')
            jsonl_file.flush()

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

    utils.save(generations, 'combined_generations.pkl')

    # Save validation-compatible copies so the existing uncertainty computation
    # script can run unchanged.
    validation_jsonl_path = os.path.join(out_dir, 'validation_generations.jsonl')
    shutil.copyfile(jsonl_path, validation_jsonl_path)
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
