"""Data Loading Utilities."""
import ast
import os
import re
import csv
import json
import hashlib
from collections import Counter
import numpy as np
import datasets


def _remove_duplicate_questions(dataset):
    """Remove duplicate examples based on question text, keeping first occurrence.

    Args:
        dataset: List of examples or HF Dataset with 'question' field

    Returns:
        List of unique examples (no duplicate questions)
    """
    if dataset is None or len(dataset) == 0:
        return dataset

    # Convert HF Dataset to list if needed
    if not isinstance(dataset, list):
        dataset = [d for d in dataset]

    seen_questions = set()
    unique_dataset = []

    for item in dataset:
        question_text = item['question'].lower().strip()
        # For image-conditioned data the question text alone is not the
        # example: "what color is the bus?" is a different question against a
        # different image. Key on (question, image) so only true duplicates go.
        if 'image_id' in item:
            key = (question_text, item['image_id'])
        else:
            key = question_text
        if key not in seen_questions:
            seen_questions.add(key)
            unique_dataset.append(item)

    return unique_dataset


def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""
    user = os.environ['USER']

    train_dataset, validation_dataset = None, None

    md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'],
            'context': x['Body'],
            'type': x['Type'],
            'equation': x['Equation'],
            'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

        reformat = lambda x: {
            'question': x['question'] + '?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "trivia_qa_nocontext":
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

        def reformat(x):
            return {
                'question': x['question'],
                'context': '',
                'id': x['question_id'],
                'answers': {'text': [x['answer']['value']] + x['answer']['aliases']},
            }

        train_dataset = [reformat(d) for d in dataset["train"]]
        validation_dataset = [reformat(d) for d in dataset["validation"]]

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        scratch_dir = os.getenv('SCRATCH_DIR', '.')
        path = f"{scratch_dir}/{user}/semantic_uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Split into training and validation set.
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "gsm8k":
        # HF dataset: openai/gsm8k, config "main"
        # Splits: train, test
        dataset = datasets.load_dataset("openai/gsm8k", "main")
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        def extract_final_answer(answer_text):
            # GSM8K answers are usually of the form "... #### 72"
            if "####" in answer_text:
                return answer_text.split("####")[-1].strip()

            # Fallback: pick last number if delimiter is missing
            matches = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer_text)
            return matches[-1].replace(",", "") if matches else answer_text.strip()

        def reformat(x):
            return {
                'question': x['question'],
                'context': '',
                'id': md5hash(str(x['question'])),
                'answers': {'text': [extract_final_answer(x['answer'])]},
            }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "sciq":
        # HF dataset: allenai/sciq, default config
        # Splits: train, validation, test
        dataset = datasets.load_dataset("allenai/sciq")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        def reformat(x):
            item = {
                'question': x['question'],
                'context': x.get('support', ''),
                'id': md5hash(str(x['question'])),
                'answers': {'text': [x['correct_answer']]},
            }

            if add_options:
                item['options'] = [
                    x['correct_answer'],
                    x['distractor1'],
                    x['distractor2'],
                    x['distractor3'],
                ]

            return item

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]
        test_dataset = [reformat(d) for d in test_dataset]

        validation_dataset += test_dataset

    elif dataset_name == "math_500":
        # HF dataset: HuggingFaceH4/MATH-500
        # Split: test only
        dataset = datasets.load_dataset("HuggingFaceH4/MATH-500")

        def reformat(x):
            return {
                'question': x['problem'],
                'context': '',
                'id': x['unique_id'],
                'answers': {'text': [str(x['answer'])]},
                'solution': x['solution'],
                'subject': x['subject'],
                'level': x['level'],
            }

        # Evaluation-only dataset
        train_dataset = []
        validation_dataset = [reformat(d) for d in dataset["test"]]

    elif dataset_name == "answerable_math":
        # Local CSVs from technion-cs-nlp/LLMsKnow
        # Columns: id, question, answer, answerable, category, relevant_ids, source
        # Sources: GSM8K, SVAMP, MultiArith, ASDiv — all math word problems
        # Answers are Python-list strings like "[9.0]"; answerable is always True here
        train_path = "/home/kalashkala/AnswerableMath.csv"
        test_path = "/home/kalashkala/AnswerableMath_test.csv"

        def _parse_answer(answer_str):
            """Convert answer field like '[9.0]' to a clean numeric string."""
            try:
                parsed = ast.literal_eval(answer_str.strip())
                if isinstance(parsed, list) and parsed:
                    val = parsed[0]
                else:
                    val = parsed
                # Drop the decimal if it's a whole number (9.0 → "9")
                if isinstance(val, float) and val == int(val):
                    return str(int(val))
                return str(val)
            except (ValueError, SyntaxError):
                return answer_str.strip()

        def _load_csv(path):
            rows = []
            with open(path, newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append({
                        'question': row['question'].strip(),
                        'context': '',
                        'id': md5hash(row['id'] + row['source']),
                        'answers': {'text': [_parse_answer(row['answer'])]},
                        'answerable': row['answerable'].strip(),
                        'source': row['source'].strip(),
                    })
            return rows

        train_dataset = _load_csv(train_path)
        validation_dataset = _load_csv(test_path)

    elif dataset_name == "advqa":
        # AdVQA val split (human-adversarial VQA over COCO val2017 images).
        # Open-ended: each question carries 10 free-form annotator answers.
        root = "/data/kalashkala/vqa_data"
        q_path = f"{root}/advqa/v1_OpenEnded_mscoco_val2017_advqa_questions.json"
        a_path = f"{root}/advqa/v1_mscoco_val2017_advqa_annotations.json"
        img_dir = f"{root}/coco/val2017"

        with open(q_path, encoding='utf-8') as fh:
            questions = json.load(fh)['questions']
        with open(a_path, encoding='utf-8') as fh:
            annotations = json.load(fh)['annotations']
        ann_by_qid = {a['question_id']: a for a in annotations}

        rows = []
        for q in questions:
            ann = ann_by_qid.get(q['question_id'])
            if ann is None:
                continue
            # A binary answer space is effectively 2-option multiple choice: it
            # puts a 50% floor under accuracy and caps semantic-entropy
            # clustering at 2 clusters. Filter before the split so both splits
            # are consistently filtered.
            if ann['answer_type'] == 'yes/no':
                continue
            all_answers = [a['answer'].strip() for a in ann['answers']]
            counts = Counter(all_answers)
            consensus, support = counts.most_common(1)[0]
            rows.append({
                'question': q['question'].strip(),
                'context': '',
                'id': md5hash(f"advqa-{q['question_id']}"),
                # Ground truth is the consensus answer, not all 10: squad_v2
                # maxes over references, and "unanswerable" is a literal
                # annotator answer in 12.8% of questions, so an all-10
                # reference set would credit an image-blind model that always
                # says "unanswerable". all_answers is kept so the label rule
                # (or an LLM judge) can be revisited without regenerating.
                'answers': {'text': [consensus]},
                'all_answers': all_answers,
                'consensus_support': support,
                'answer_type': ann['answer_type'],
                'image_id': q['image_id'],
                'question_id': q['question_id'],
                'image_path': f"{img_dir}/{q['image_id']:012d}.jpg",
            })

        rows.sort(key=lambda r: r['question_id'])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(rows))
        rows = [rows[i] for i in perm]
        # The train slice is unused at --num_few_shot 0 but keeps
        # _build_eval_pool and split_dataset on their normal path.
        train_dataset = rows[:1000]
        validation_dataset = rows[1000:]

    elif dataset_name == "vqav2":
        # VQA v2 val2014, pre-sampled to 1/4 of the images (random, seed 42)
        # and yes/no questions dropped, mirroring the advqa filtering rationale
        # (binary answers put a 50% guessing floor under accuracy and cap
        # semantic-entropy clustering at 2 clusters). Ground truth is the
        # official 'multiple_choice_answer' consensus field.
        root = "/data/kalashkala/vqa_data/vqav2/sampled"
        q_path = f"{root}/sampled_questions.json"
        a_path = f"{root}/sampled_annotations.json"
        img_dir = "/data/kalashkala/vqa_data/coco/val2014"

        with open(q_path, encoding='utf-8') as fh:
            questions = json.load(fh)
        with open(a_path, encoding='utf-8') as fh:
            annotations = json.load(fh)
        ann_by_qid = {a['question_id']: a for a in annotations}

        rows = []
        for q in questions:
            ann = ann_by_qid.get(q['question_id'])
            if ann is None:
                continue
            all_answers = [a['answer'].strip() for a in ann['answers']]
            counts = Counter(all_answers)
            _, support = counts.most_common(1)[0]
            rows.append({
                'question': q['question'].strip(),
                'context': '',
                'id': md5hash(f"vqav2-{q['question_id']}"),
                'answers': {'text': [ann['multiple_choice_answer']]},
                'all_answers': all_answers,
                'consensus_support': support,
                'answer_type': ann['answer_type'],
                'image_id': q['image_id'],
                'question_id': q['question_id'],
                'image_path': f"{img_dir}/COCO_val2014_{q['image_id']:012d}.jpg",
            })

        rows.sort(key=lambda r: r['question_id'])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(rows))
        rows = [rows[i] for i in perm]
        # The train slice is unused at --num_few_shot 0 but keeps
        # _build_eval_pool and split_dataset on their normal path.
        train_dataset = rows[:1000]
        validation_dataset = rows[1000:]

    elif dataset_name == "okvqa":
        # OK-VQA: knowledge-based VQA over COCO train2014+val2014 images.
        # Unlike AdVQA/VQAv2, answer_type is "other" for 100% of questions
        # (no yes/no category), so no binary-guessing-floor filter is needed.
        # Both splits carry full annotations (unlike AdVQA, where only val
        # has them), and the dataset is small (~14k total), so we pool both
        # splits into one set rather than treating val as the sole eval pool.
        root = "/data/kalashkala/vqa_data/okvqa"
        img_root = "/data/kalashkala/vqa_data/coco"

        rows = []
        for split in ["train2014", "val2014"]:
            q_path = f"{root}/OpenEnded_mscoco_{split}_questions.json"
            a_path = f"{root}/mscoco_{split}_annotations.json"
            with open(q_path, encoding='utf-8') as fh:
                questions = json.load(fh)['questions']
            with open(a_path, encoding='utf-8') as fh:
                annotations = json.load(fh)['annotations']
            ann_by_qid = {a['question_id']: a for a in annotations}

            for q in questions:
                ann = ann_by_qid.get(q['question_id'])
                if ann is None:
                    continue
                all_answers = [a['answer'].strip() for a in ann['answers']]
                counts = Counter(all_answers)
                consensus, support = counts.most_common(1)[0]
                rows.append({
                    'question': q['question'].strip(),
                    'context': '',
                    'id': md5hash(f"okvqa-{split}-{q['question_id']}"),
                    'answers': {'text': [consensus]},
                    'all_answers': all_answers,
                    'consensus_support': support,
                    'answer_type': ann['answer_type'],
                    'question_type': ann['question_type'],
                    'image_id': q['image_id'],
                    'question_id': q['question_id'],
                    'image_path': f"{img_root}/{split}/COCO_{split}_{q['image_id']:012d}.jpg",
                })

        rows.sort(key=lambda r: r['id'])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(rows))
        rows = [rows[i] for i in perm]
        # The train slice is unused at --num_few_shot 0 but keeps
        # _build_eval_pool and split_dataset on their normal path.
        train_dataset = rows[:1000]
        validation_dataset = rows[1000:]

    else:
        raise ValueError

    # Convert HF Datasets to lists if needed
    if not isinstance(train_dataset, list):
        train_dataset = [d for d in train_dataset]
    if not isinstance(validation_dataset, list):
        validation_dataset = [d for d in validation_dataset]

    # Remove duplicate questions from both splits
    train_dataset = _remove_duplicate_questions(train_dataset)
    validation_dataset = _remove_duplicate_questions(validation_dataset)

    return train_dataset, validation_dataset