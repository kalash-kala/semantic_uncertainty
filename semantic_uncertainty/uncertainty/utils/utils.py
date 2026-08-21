"""Utility functions."""
import os
import logging
import argparse
import pickle
import re
import uuid

import wandb

from evaluate import load

from uncertainty.models.huggingface_models import HuggingfaceModel
from uncertainty.utils import openai as oai

BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n',
    # Chain-of-thought. Used with --reasoning. The '####' delimiter is the GSM8K
    # convention and is what ANSWER_DELIMITER / extract_final_answer key off, so
    # the three must stay in sync.
    'cot': ("Solve the following math problem. Reason step by step, then give "
            "the final numeric answer on a new line starting with '#### '.\n")}

ANSWER_DELIMITER = '####'

# Stop sequences for chain-of-thought generation. The default set contains '\n',
# which truncates reasoning at the first line break and is the single hardest
# blocker on CoT — no token budget can work around it. Here we keep only the
# few-shot record separator and the field markers.
COT_STOP_SEQUENCES = ['\n\n\n\n', 'Question:', 'Context:']

# The answerable_math CSV has no rationale column (only id,question,answer,
# answerable,category,relevant_ids,source), so few-shot CoT exemplars cannot be
# drawn from the data — using the dataset's own bare numeric answers as
# demonstrations is exactly what teaches the model to skip reasoning. These are
# hand-written and deliberately generic.
COT_FEWSHOT_EXEMPLARS = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as "
     "many clips in May. How many clips did she sell altogether?",
     "In April Natalia sold 48 clips.\n"
     "In May she sold half as many, so she sold 48 / 2 = 24 clips.\n"
     "Altogether she sold 48 + 24 = 72 clips.\n"
     f"{ANSWER_DELIMITER} 72"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How "
     "many bolts in total does it take?",
     "The robe takes 2 bolts of blue fiber.\n"
     "It takes half that much white fiber, so 2 / 2 = 1 bolt of white fiber.\n"
     "In total that is 2 + 1 = 3 bolts.\n"
     f"{ANSWER_DELIMITER} 3"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
     "minutes of babysitting. How much did she earn?",
     "Weng earns $12 per 60 minutes, so per minute she earns 12 / 60 = $0.2.\n"
     "For 50 minutes she earned 50 * 0.2 = $10.\n"
     f"{ANSWER_DELIMITER} 10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only "
     "half of the money she needs. Her parents decided to give her $15 for "
     "that purpose, and her grandparents twice as much as her parents. How "
     "much more money does Betty need to buy the wallet?",
     "Betty has half of $100, which is 100 / 2 = $50.\n"
     "Her parents give her $15 and her grandparents give twice as much, "
     "so 2 * 15 = $30.\n"
     "Now she has 50 + 15 + 30 = $95.\n"
     "She still needs 100 - 95 = $5.\n"
     f"{ANSWER_DELIMITER} 5"),
]


def extract_final_answer(text):
    """Pull the final answer out of a (possibly chain-of-thought) response.

    Returns the text after the first '####' delimiter. If the model never emitted
    one — which happens when it runs out of tokens mid-derivation — fall back to
    the last number in the response, and finally to the stripped text.

    The fallback matters: without it a correct derivation that was truncated
    before the delimiter would be scored as wrong.
    """
    if text is None:
        return ''
    if ANSWER_DELIMITER in text:
        # Use the FIRST '####', not the last: models sometimes re-echo the
        # delimiter afterwards (e.g. "#### 70\n#### Final Answer: 70\n####
        # \\####"), and the trailing occurrence can be empty or garbage,
        # silently corrupting the extracted answer.
        tail = text.split(ANSWER_DELIMITER, 1)[1]
        # Stop at a newline so a trailing few-shot fragment cannot leak in.
        return tail.split('\n')[0].strip()
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].strip()
    return text.strip()


def normalize_numeric_answer(text):
    """Canonicalise an answer for comparison: '[9.0]', '9.0', '$9', '9' -> '9'.

    answerable_math stores references as the string '[9.0]', so without this the
    reference and a correct prediction never match.
    """
    if text is None:
        return ''
    s = str(text).strip().strip('[]').replace(',', '').replace('$', '').strip()
    m = re.search(r'-?\d*\.?\d+', s)
    if not m:
        return s.lower()
    num = m.group(0)
    try:
        f = float(num)
    except ValueError:
        return num
    # Render integral floats without the trailing '.0' so 9.0 == 9.
    return str(int(f)) if f == int(f) else str(f)


def get_parser(stages=['generate', 'compute']):
    entity = os.getenv('WANDB_SEM_UNC_ENTITY', None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False,
        help="Keep default wandb clean.")
    parser.add_argument('--entity', type=str, default=entity)
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="squad",
        choices=['squad', 'math', 'llm', 'llm_gpt-3.5', 'llm_gpt-4'],
        help="Metric to assign accuracy to generations. Use 'math' with "
             "--reasoning: it exact-matches the extracted final answer, whereas "
             "'squad' F1-scores the whole response and so penalises correct "
             "chain-of-thought derivations.")
    parser.add_argument(
        "--compute_accuracy_at_all_temps",
        action=argparse.BooleanOptionalAction, default=True,
        help="Compute accuracy at all temperatures or only t<<1.")
    parser.add_argument(
        "--experiment_lot", type=str, default='Unnamed Experiment',
        help="Keep default wandb clean.")
    if 'generate' in stages:
        parser.add_argument(
            "--model_name", type=str, default="Llama-2-7b-chat", help="Model name",
        )
        parser.add_argument(
            "--model_max_new_tokens", type=int, default=50,
            help="Max number of tokens generated.",
        )
        parser.add_argument(
            "--dataset", type=str, default="trivia_qa",
            choices=['trivia_qa', 'trivia_qa_nocontext', 'squad', 'bioasq', 'nq', 'svamp', 'sciq', 'gsm8k', 'answerable_math', 'math_500', 'advqa', 'vqav2', 'okvqa'],
            help="Dataset to use")
        parser.add_argument(
            "--ood_train_dataset", type=str, default=None,
            choices=['trivia_qa', 'trivia_qa_nocontext', 'squad', 'bioasq', 'nq', 'svamp', 'sciq', 'gsm8k', 'answerable_math', 'math_500', 'advqa', 'vqav2', 'okvqa'],
            help="Dataset to use to assemble few-shot prompt, p_true prompt, and train p_ik.")
        parser.add_argument(
            "--num_samples", type=int, default=400,
            help="Number of samples to use")
        parser.add_argument(
            "--num_few_shot", type=int, default=5,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_num_fewshot", type=int, default=20,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_hint", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--num_generations", type=int, default=10,
            help="Number of generations to use")
        parser.add_argument(
            "--temperature", type=float, default=1.0,
            help="Temperature")
        parser.add_argument(
            "--use_mc_options", type=bool, default=True,
            help="Include MC options question?")
        parser.add_argument(
            "--get_training_set_generations", default=True,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--use_context", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--get_training_set_generations_most_likely_only", default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Only get embedding of most likely answer for training set. "
                "This is all that's needed for p_true."))
        parser.add_argument('--compute_p_true', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_always", default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--enable_brief", default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_prompt", default='default', type=str)
        parser.add_argument(
            "--prompt_type", default='default', type=str)
        parser.add_argument(
            "--compute_uncertainties", default=True,
            action=argparse.BooleanOptionalAction,
            help='Trigger compute_uncertainty_measures.py')
        parser.add_argument(
            "--answerable_only", default=False,
            action=argparse.BooleanOptionalAction,
            help='Exclude unanswerable questions.')
        parser.add_argument(
            "--generation_batch_size", type=int, default=1,
            help="Number of questions to process in one model.generate() call. "
                 "Combined with num_return_sequences for A+B batching.")
        parser.add_argument(
            "--enable_thinking", default=False,
            action=argparse.BooleanOptionalAction,
            help="Enable thinking/reasoning mode for models that support it (like Qwen3).")
        parser.add_argument(
            "--reasoning", default=False,
            action=argparse.BooleanOptionalAction,
            help="Chain-of-thought mode: drop '\\n' from the stop sequences, use "
                 "the 'cot' brief prompt with hand-written reasoning exemplars, "
                 "and score/cluster on the extracted final answer instead of the "
                 "full derivation. Needs a large --model_max_new_tokens (~256).")
        # --- hidden-state storage control -----------------------------------
        # Hidden states dominate run size (~270 KB/question for a 4096-dim model
        # at 3 positions x 11 sequences). They are a recomputable cache: the
        # JSONL keeps prompt_token_ids + generated_ids, so extract_hidden_states.py
        # can rebuild them by teacher forcing. Default off; opt in when needed.
        parser.add_argument(
            "--save_hidden_states", default=False,
            action=argparse.BooleanOptionalAction,
            help="Extract and save hidden states to PKL. Off by default: they are "
                 "rebuildable from the token IDs in the JSONL. Enable for p_ik or "
                 "any probing work that needs embeddings in the same run.")
        parser.add_argument(
            "--embedding_positions", type=str,
            default='first_answer,last_prompt,last_token',
            help="Comma-separated subset of first_answer,last_prompt,last_token to "
                 "keep. Ignored unless --save_hidden_states.")
        parser.add_argument(
            "--embedding_sequences", type=str, default='all',
            choices=['all', 'greedy'],
            help="'all' stores embeddings for the greedy answer and every sampled "
                 "response; 'greedy' stores only the greedy answer (~11x smaller). "
                 "Ignored unless --save_hidden_states.")

    if 'compute' in stages:
        parser.add_argument('--recompute_accuracy',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--eval_wandb_runid', type=str,
                            help='wandb run id of the dataset to evaluate on')
        parser.add_argument('--train_wandb_runid', type=str, default=None,
                            help='wandb run id of the dataset from which training embeddings and p_true samples will be taken')
        parser.add_argument('--num_eval_samples', type=int, default=int(1e19))
        parser.add_argument('--compute_predictive_entropy',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_ik', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_ik_answerable', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_context_entails_response', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--analyze_run', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--assign_new_wandb_id', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--restore_entity_eval', type=str, default=entity)
        parser.add_argument('--restore_entity_train', type=str, default=entity)
        parser.add_argument('--condition_on_question',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--strict_entailment',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_all_generations', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_num_generations', type=int, default=-1)
        parser.add_argument("--entailment_model", default='deberta', type=str)
        parser.add_argument(
            "--entailment_cache_id", default=None, type=str,
            help='Restore entailment predictions from previous run for GPT-4/LLaMa-Entailment.')
        parser.add_argument('--entailment_cache_only', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_true_in_compute_stage',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--reuse_entailment_model',
                            default=False, action=argparse.BooleanOptionalAction,
                            help='Use entailment model as p_true model.')
    return parser


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def construct_cot_fewshot_prompt(brief, make_prompt, n_shot=None):
    """Few-shot prompt whose exemplars demonstrate reasoning, not bare answers.

    Deliberately does NOT read the dataset: answerable_math's answers are bare
    numbers, and demonstrations override instructions — showing 'Answer: 9' four
    times teaches the model to skip the derivation no matter what the brief says.
    """
    exemplars = COT_FEWSHOT_EXEMPLARS
    if n_shot is not None:
        exemplars = exemplars[:max(0, n_shot)]
    # brief once at the top; brief_always=False stops make_prompt repeating it
    # in front of every exemplar.
    prompt = brief
    for question, worked_solution in exemplars:
        prompt += make_prompt(None, question, worked_solution, brief, False)
    return prompt


def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if 'gpt' in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        logging.warning('Redo llm check.')
        predicted_answer, _, _ = model.predict(prompt, 1)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        logging.warning('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)


def get_gpt_metric(metric_name):

    model_name = '_'.join(metric_name.split('_')[1:])

    class EntailmentGPT():
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, prompt, temperature):
            return oai.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}
    return reference


def init_model(args):
    mn = args.model_name
    # Under --reasoning the default stop set would cut the derivation at the
    # first newline, so swap in the CoT set.
    stops = COT_STOP_SEQUENCES if getattr(args, 'reasoning', False) else 'default'
    if ('llama' in mn.lower() or 'falcon' in mn or 'mistral' in mn.lower() or 'qwen' in mn.lower()
            or 'gemma' in mn.lower() or 'pixtral' in mn.lower()):
        if getattr(args, 'reasoning', False) and args.model_max_new_tokens < 128:
            logging.warning(
                'reasoning=True but model_max_new_tokens=%d. Chain-of-thought '
                'needs ~256; derivations will be truncated mid-way.',
                args.model_max_new_tokens)
        model = HuggingfaceModel(
            mn, stop_sequences=stops,
            max_new_tokens=args.model_max_new_tokens,
            enable_thinking=getattr(args, 'enable_thinking', False))
    else:
        raise ValueError(f'Unknown model_name `{mn}`.')
    return model


def get_make_prompt(args):
    if args.prompt_type == 'default':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += 'Answer:'
            return prompt
    else:
        raise ValueError

    return make_prompt


def get_metric(metric):
    if metric == 'squad':

        # Load once per process — using PID as experiment_id avoids race
        # conditions when multiple workers share the same cache dir, while
        # eliminating the per-call reload overhead.
        _squad_metric = load("squad_v2", experiment_id=f"pid{os.getpid()}")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            results = _squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0

    elif metric == 'math':
        # Exact match on the normalised final answer.
        #
        # SQuAD F1 over the whole response is actively wrong for chain-of-thought:
        # scored against a reference of "9", a correct 200-token derivation earns
        # almost no token overlap, so measured accuracy would FALL as true
        # accuracy rose. Compare only the extracted final answer.
        def metric(response, example, *args, **kwargs):
            reference = get_reference(example)
            golds = reference['answers']['text']
            pred = normalize_numeric_answer(extract_final_answer(response))
            if not pred:
                return 0.0
            return 1.0 if any(
                pred == normalize_numeric_answer(g) for g in golds) else 0.0

    # Reuses the globally active model for these.
    elif metric == 'llm':
        metric = llm_metric
    elif metric == 'llm_gpt-3.5':
        metric = get_gpt_metric(metric)
    elif metric == 'llm_gpt-4':
        metric = get_gpt_metric(metric)
    else:
        raise ValueError

    return metric


def save(object, file):
    import os
    out_dir = os.environ.get("SU_LOCAL_RUN_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    # with open(f'{wandb.run.dir}/{file}', 'wb') as f:
    #     pickle.dump(object, f)
    # wandb.save(f'{wandb.run.dir}/{file}')
    with open(f'{out_dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
