"""Data Loading Utilities."""
import os
import re
import json
import hashlib
import datasets


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

        def reformat(x):
            item = {
                'question': x['question'],
                'context': x.get('support', ''),
                'id': md5hash(str(x['question'])),
                'answers': {'text': [x['correct_answer']]},
            }

            # Optional: include answer choices if needed elsewhere
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

    else:
        raise ValueError

    return train_dataset, validation_dataset