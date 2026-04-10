# Updates to `semantic_uncertainty` for Modern LLMs

This document outlines the modifications made to the `semantic_uncertainty` codebase to safely and easily run newer Large Language Models, including Qwen2.5, Gemma-3, Mistral v0.3, and Llama 3.1. 

Previously, the codebase utilized inflexible string matching and manual base-path appending (e.g., `huggyllama/` or `mistralai/`), preventing users from directly passing HuggingFace Hub identifiers for novel models. Furthermore, token limits were hardcoded. 

## 1. Whitelisting Modenames (`uncertainty/utils/utils.py`)
**Goal**: Allow `init_model(args)` to execute without failing validation.
* Updated the main string filter check inside `init_model` from restricting models to just `llama`, `falcon`, and `mistral`.
* **Changes**: The `if` statement now explicitly checks if `'qwen' in mn.lower() or 'gemma' in mn.lower()`. 

## 2. Direct Repository Loading (`uncertainty/models/huggingface_models.py`)
**Goal**: Subvert hardcoded prefixes and allow explicit HF identifiers like `Qwen/Qwen2.5-7B-Instruct`.
* The `HuggingfaceModel.__init__` procedure originally prefixed base directories blindly. 
* **Changes**: Added a new conditional branch `if '/' in model_name:`. If the passed `model_name` contains a slash (indicating a direct HuggingFace Hub path), it skips manual path assembly. It directly passes the `model_name` to `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`.
* We ported the native `-8bit` and `-4bit` suffix trimming to this branch so you can safely use strings like `Qwen/Qwen2.5-7B-Instruct-8bit`.

## 3. Dynamic Context Limits (`uncertainty/models/huggingface_models.py`)
**Goal**: Support the vastly expanded context windows of modern models.
* The original setup artificially capped token generation limits to `2048` or `4096` right after model loading with `self.token_limit`. 
* **Changes**: Replaced this hardcoded toggle with dynamic extraction. The limit is now assigned via `self.model.config.max_position_embeddings`, defaulting to the fallback 2048/4096 mapping `except AttributeError`.

## 4. Tokenizer Cleanup & Padding Updates (`uncertainty/models/huggingface_models.py`)
**Goal**: Make sure padding token assignment and unhandled `token_type_ids` don't break generation.
* The `.generate()` loop strips `token_type_ids` dynamically for certain models. Qwen and Gemma pass these through from their respective tokenizers as well, which breaks their downstream forward passes.
* **Changes**: The conditional `if 'llama' in ...` blocks in the `predict` function have been expanded again to whitelist `qwen` and `gemma`.
* **Changes**: Updated the `pad_token_id` logic to safely test for an explicit `pad_token_id` before defaulting back to `eos_token_id`, which behaves securely with Qwen.

## 5. WandB Uncoupling & Local Artifact Saving (`uncertainty/utils/utils.py`)
**Goal**: Make the pipeline independent of Weights & Biases (WandB) while safely retaining the codebase's original intent by commenting it out.
* Local save outputs via `wandb.run.dir` and remote syncs via `wandb.save` were hooked across standard modules.
* **Changes**: Fully commented out `wandb.save()` and substituted `wandb.run.dir` calls. The script now checks for the local `SU_LOCAL_RUN_DIR` path environment variable to securely dump any trailing artifacts to your specific working directory without demanding cloud synchronization.

## 6. JSONL Streaming Output Support (`semantic_uncertainty/generate_answers.py`)
**Goal**: Eliminate huge, crash-prone whole-dataset `.pkl` memory dumps and save incremental outputs immediately.
* The script formally waited until the conclusion of the evaluation batches to serialize hundreds of queries at once into a massive Pickle file object.
* **Changes**: Commented out all WandB triggers (`wandb.init`, `wandb.log`, `wandb.config`). Configured a localized JSON-lines (`.jsonl`) stream to instantaneously flush every single answer-generation to the hard drive reliably. 
* **Changes**: Added a custom PyTorch tensor trap to cleanly convert nested embeddings sequentially into standard lists (`.tolist()`) to perfectly support strictly-typed JSON serializations.

## 7. JSONL Decoupled Evaluation Setup (`semantic_uncertainty/compute_uncertainty_measures.py`)
**Goal**: Make sure downstream hallucination calculations don't utilize `wandb.Api()` to fetch datasets, and instead properly decode your new sequential JSON files.
* The original framework overtly synchronized `train_generations` recursively by downloading elements from a `wandb.Api` context.
* **Changes**: Bypassed API triggers entirely by routing the overarching `restore()` functionality internally towards the offline script `local_run/` outputs.
* **Changes**: Enveloped `.pkl` readers into flexible `try/except` catchers that primarily search for the newly constructed `generations.jsonl` logs, iterating the array line-by-batch smoothly. 
* **Changes**: Wrapped the end of the semantic evaluation iteration loop cleanly so that the final scores similarly stream reliably to a new localized file called `uncertainty_measures.jsonl`.

## 8. Adding New HuggingFace Datasets (`uncertainty/data/data_utils.py`)
**Goal**: Integrate custom datasets from the HuggingFace Hub or local files into the pipeline.
* The pipeline expects a specific dictionary schema: `question`, `context`, `answers` (with a `text` list), and `id`.
* **Instructions**:
    1. Open `uncertainty/data/data_utils.py`.
    2. Add a new `elif dataset_name == "your_dataset":` block.
    3. Use `datasets.load_dataset("path/to/repo")` to download the data.
    4. Use the `.map()` function or a list comprehension to reformat your columns into the required internal schema.
    5. *(Optional)* Add your dataset name to the `choices` list in `uncertainty/utils/utils.py` to enable command-line validation.

### Code Templates for `load_ds()`

**Example: Local JSON Dataset**
```python
    elif dataset_name == "local_data":
        with open("data.json", "r") as f:
            raw = json.load(f)
        reformat = lambda x: {
            'question': x['query'],
            'context': x.get('context', ''),
            'answers': {'text': [x['ans']]},
            'id': str(x['unique_id'])
        }
        full_dataset = [reformat(d) for d in raw]
        train_dataset = full_dataset[:10]
        validation_dataset = full_dataset[10:]
```

**Example: HuggingFace Dataset**
```python
    elif dataset_name == "hf_data":
        dataset = datasets.load_dataset("username/repo")
        def map_columns(ex):
            return {
                'question': ex['question_col'],
                'context': ex.get('context_col', ''),
                'answers': {'text': ex['answer_list_col']},
                'id': str(ex['id_col'])
            }
        train_dataset = dataset["train"].map(map_columns)
        validation_dataset = dataset["test"].map(map_columns)
```
