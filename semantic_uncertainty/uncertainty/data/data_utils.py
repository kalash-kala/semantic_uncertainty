"""Data Loading Utilities."""
import ast
import os
import re
import csv
import json
import hashlib
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
        if question_text not in seen_questions:
            seen_questions.add(question_text)
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