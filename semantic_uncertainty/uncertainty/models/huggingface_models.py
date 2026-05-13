"""Implement HuggingfaceModel models."""
import copy
import logging
from collections import Counter
import torch

import accelerate

from transformers import AutoTokenizer, AutoProcessor
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    Gemma3ForConditionalGeneration = None
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        if '/' in model_name:
            model_id = model_name
            if model_id.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True,)}
                model_id = model_id[:-len('-8bit')]
            elif model_id.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(load_in_4bit=True,)}
                model_id = model_id[:-len('-4bit')]
            else:
                kwargs = {}

            if 'gemma-3' in model_id.lower():
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.tokenizer = self.processor.tokenizer
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map='auto',
                    max_memory={0: '80GIB'},
                    torch_dtype=torch.bfloat16,
                    **kwargs,
                ).eval()
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, device_map='auto', token_type_ids=None,
                    clean_up_tokenization_spaces=False)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map='auto',
                    max_memory={0: '80GIB'},
                    **kwargs,
                )

        elif 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            if 'Llama-2' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf'
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None)

            llama65b = '65b' in model_name and base == 'huggyllama'
            llama2_70b = '70b' in model_name and base == 'meta-llama'

            if ('7b' in model_name or '13b' in model_name) or eightbit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto",
                    max_memory={0: '80GIB'}, **kwargs,)

            elif llama2_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()
                max_mem = 15 * 4686198491

                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16', skip_keys='past_key_values')
            else:
                raise ValueError

        elif 'mistral' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            if model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,)}
                model_name = model_name[:-len('-4bit')]
            else:
                kwargs = {}

            model_id = f'mistralai/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='auto',
                max_memory={0: '80GIB'},
                **kwargs,
            )

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                **kwargs,
            )
        else:
            raise ValueError

        self.model_name = model_name
        # Left-padding is required for batched generation so all sequences
        # end at the same position and generation alignment is correct.
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        
        try:
            self.token_limit = min(8192, self.model.config.max_position_embeddings)
        except AttributeError:
            self.token_limit = 8192

    def predict(self, input_data, temperature, return_full=False, do_sample=True):

        # Implement prediction.
        mn_lower = self.model_name.lower()
        if hasattr(self, 'processor') and 'gemma-3' in mn_lower:
            if isinstance(input_data, str) and '-it' in mn_lower:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": input_data}]}
                ]
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                )
            elif not isinstance(input_data, str):
                # Assume it's a list of messages (possibly multimodal)
                inputs = self.processor.apply_chat_template(
                    input_data, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                )
            else:
                inputs = self.processor(text=input_data, return_tensors="pt")
            
            # Cast inputs to bfloat16 as recommended for Gemma-3 multimodal
            inputs = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif hasattr(self, 'processor'):
            inputs = self.processor(text=input_data, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        mn_lower = self.model_name.lower()
        if 'llama' in mn_lower or 'falcon' in mn_lower or 'mistral' in mn_lower or 'qwen' in mn_lower or 'gemma' in mn_lower:
            if 'token_type_ids' in inputs:  # Some HF models have changed.
                del inputs['token_type_ids']
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None) or self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f | do_sample: %s', temperature, do_sample)

        # Get prompt embedding (last token of input)
        with torch.no_grad():
            prompt_outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                output_hidden_states=True,
            )
        if 'decoder_hidden_states' in prompt_outputs.keys():
            prompt_hidden = prompt_outputs.decoder_hidden_states
        else:
            prompt_hidden = prompt_outputs.hidden_states
        prompt_last_token_embedding = prompt_hidden[-1][0, -1, :].cpu() if prompt_hidden else None

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            do_sample=do_sample,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
        )

        if 'gemma-3' in self.model_name.lower():
            generate_kwargs['min_new_tokens'] = 1

        if do_sample:
            generate_kwargs['temperature'] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        n_input_tokens = inputs['input_ids'].shape[1]
        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # Token-based stripping to get the answer
        answer_tokens = outputs.sequences[0][n_input_tokens:]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate input_data_offset in full_answer for remaining logic
        # This is safe because answer is always a suffix of full_answer 
        # when skipping special tokens consistently.
        input_data_offset = len(full_answer) - len(answer)

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                if 'falcon' not in self.model_name.lower():
                    raise ValueError(error_msg)
                else:
                    logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()

        # Get the number of tokens until the stop word comes up.
        # Get token counts / hidden states / likelihoods.
        # For Gemma-3, avoid re-tokenizing decoded text to infer n_generated,
        # because that bookkeeping is what is mismatching.
        if 'gemma-3' in mn_lower:
            # Count actual generated steps directly from generation outputs.
            n_generated = len(outputs.scores) if outputs.scores is not None else answer_tokens.shape[0]

            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            log_likelihoods = [score.item() for score in transition_scores[0]]

            if len(log_likelihoods) == 0:
                logging.warning('Gemma returned no transition scores.')
                return sliced_answer, [-100.0], {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}

            # Keep counts aligned with actual returned scores.
            n_generated = min(n_generated, len(log_likelihoods))
            log_likelihoods = log_likelihoods[:n_generated]

            if len(log_likelihoods) == self.max_new_tokens:
                logging.warning('Generation interrupted by max_token limit.')

            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states

            # Hidden states may still be missing even when text + scores are present.
            if hidden is None or len(hidden) == 0:
                logging.warning('Gemma returned empty hidden states; using valid log-likelihoods and no embedding.')
                return sliced_answer, log_likelihoods, {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}

            n_hidden_steps = min(n_generated, len(hidden))
            if n_hidden_steps <= 0:
                logging.warning('Gemma had no usable hidden-state steps; using valid log-likelihoods and no embedding.')
                return sliced_answer, log_likelihoods, {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}

            last_input = hidden[n_hidden_steps - 1]
            last_layer = last_input[-1]
            last_token_emb = last_layer[:, -1, :].cpu()

            # Extract first answer embedding (first generated token)
            first_answer_emb = hidden[0][-1][0, -1, :].cpu() if hidden else None

            # Keep likelihoods aligned with the step used for embedding.
            log_likelihoods = log_likelihoods[:n_hidden_steps]

            embeddings_dict = {
                'first_answer': first_answer_emb,
                'last_prompt': prompt_last_token_embedding,
                'last_token': last_token_emb
            }

            return sliced_answer, log_likelihoods, embeddings_dict

        else:
            # Existing logic for models that are already working fine.
            token_stop_index = self.tokenizer(
                full_answer[:input_data_offset + stop_at],
                return_tensors="pt"
            )['input_ids'].shape[1]
            n_input_token = len(inputs['input_ids'][0])
            n_generated = token_stop_index - n_input_token

            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states

            if n_generated <= 0 or hidden is None or len(hidden) == 0:
                logging.warning("Zero-token generation or empty hidden states.")
                return sliced_answer, [-100.0], {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}

            if len(hidden) == 1:
                logging.warning(
                    'Taking first and only generation for hidden! '
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer,
                )
                last_input = hidden[0]
            elif ((n_generated - 1) >= len(hidden)):
                logging.error(
                    'Taking last state because n_generated is too large'
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s, slice_answer: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer, sliced_answer
                )
                last_input = hidden[-1]
            else:
                last_input = hidden[n_generated - 1]

            last_layer = last_input[-1]
            last_token_emb = last_layer[:, -1, :].cpu()

            # Extract first answer embedding (first generated token)
            first_answer_emb = hidden[0][-1][0, -1, :].cpu() if hidden else None

            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            log_likelihoods = [score.item() for score in transition_scores[0]]
            if len(log_likelihoods) == 1:
                logging.warning('Taking first and only generation for log likelihood!')
                log_likelihoods = log_likelihoods
            else:
                log_likelihoods = log_likelihoods[:n_generated]

            if len(log_likelihoods) == self.max_new_tokens:
                logging.warning('Generation interrupted by max_token limit.')

            if len(log_likelihoods) == 0:
                raise ValueError

            embeddings_dict = {
                'first_answer': first_answer_emb,
                'last_prompt': prompt_last_token_embedding,
                'last_token': last_token_emb
            }

            return sliced_answer, log_likelihoods, embeddings_dict

    def predict_batch_questions(self, prompts, temperature, do_sample, num_return_sequences=1,
                                return_token_ids=False):
        """Generate answers for a batch of B different prompts.

        When num_return_sequences=1 (greedy or single sample):
            returns list of B (answer, log_likelihoods, embedding_or_token_info) tuples.
        When num_return_sequences=N (combined A+B):
            returns list of B lists, each containing N (answer, log_likelihoods, ...) tuples.
            HF output ordering: [q0_s0..q0_s(N-1), q1_s0..q1_s(N-1), ..., q(B-1)_s(N-1)].

        When return_token_ids=True, the embedding slot is replaced by a dict:
            {'generated_ids': list[int], 'prompt_ids': list[int] or None}
        This skips hidden-state extraction during generation (saves GPU memory),
        allowing embeddings to be recomputed later via extract_embeddings_batch().

        Stop words are stripped post-hoc (stopping criteria skipped to avoid
        one sequence halting the whole batch).
        """
        mn_lower = self.model_name.lower()
        B = len(prompts)

        if hasattr(self, 'processor') and 'gemma-3' in mn_lower:
            # Gemma-3: apply_chat_template per prompt then left-pad manually.
            token_lists = []
            for prompt in prompts:
                if isinstance(prompt, str) and '-it' in mn_lower:
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                    tok = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt")
                elif not isinstance(prompt, str):
                    tok = self.processor.apply_chat_template(
                        prompt, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt")
                else:
                    tok = self.processor(text=prompt, return_tensors="pt")
                token_lists.append(tok['input_ids'][0])

            max_len = max(t.shape[0] for t in token_lists)
            pad_id = self.tokenizer.pad_token_id
            padded = torch.stack([
                torch.cat([torch.full((max_len - t.shape[0],), pad_id, dtype=torch.long), t])
                for t in token_lists
            ])
            attn_mask = (padded != pad_id).long()
            inputs = {
                'input_ids': padded.to(self.model.device),
                'attention_mask': attn_mask.to(self.model.device),
            }
            inputs = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
                      for k, v in inputs.items()}
        elif hasattr(self, 'processor'):
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.model.device)
        else:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        if 'llama' in mn_lower or 'falcon' in mn_lower or 'mistral' in mn_lower or 'qwen' in mn_lower or 'gemma' in mn_lower:
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None) or self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        # With left-padding, all sequences in the batch share the same padded
        # input length. Generated tokens always start at index max_input_len.
        # (real_input_lens are the unpadded lengths, but are NOT the right offset
        # to use when slicing outputs.sequences — that must be max_input_len.)
        max_input_len = inputs['input_ids'].shape[1]  # same for all sequences after padding

        if return_token_ids:
            # Skip the prompt embedding forward pass — embeddings will be extracted
            # later via extract_embeddings_batch() using the stored token IDs.
            prompt_last_token_embeddings = None
        else:
            # Get prompt embeddings (last token of input) for all sequences
            with torch.no_grad():
                prompt_outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', None),
                    output_hidden_states=True,
                )
            if 'decoder_hidden_states' in prompt_outputs.keys():
                prompt_hidden = prompt_outputs.decoder_hidden_states
            else:
                prompt_hidden = prompt_outputs.hidden_states
            # prompt_hidden[-1] is last layer, shape [B, max_input_len, hidden_dim]
            prompt_last_token_embeddings = prompt_hidden[-1][:, -1, :].cpu() if prompt_hidden else None

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=not return_token_ids,  # skip hidden states when saving token IDs
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
        )
        if 'gemma-3' in mn_lower:
            generate_kwargs['min_new_tokens'] = 1
        if do_sample:
            generate_kwargs['temperature'] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        # outputs.sequences.shape = [B*N, max_input_len + max_new_tokens]
        # Sequence index for question i, sample j = i*N + j
        N = num_return_sequences

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

        # --- Early return: token IDs instead of hidden-state embeddings ---
        if return_token_ids:
            def _extract_ids(seq_idx):
                answer_tokens = outputs.sequences[seq_idx][max_input_len:]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                stop_at = len(answer)
                if self.stop_sequences is not None:
                    for stop in self.stop_sequences:
                        idx = answer.find(stop)
                        if idx != -1 and idx < stop_at:
                            stop_at = idx
                sliced_answer = answer[:stop_at].strip()
                n_gen = len(answer_tokens)
                if stop_at < len(answer) and n_gen > 0:
                    try:
                        n_gen = min(
                            self.tokenizer(answer[:stop_at], return_tensors='pt')['input_ids'].shape[1],
                            n_gen)
                    except Exception:
                        pass
                lls = [s.item() for s in transition_scores[seq_idx]]
                if n_gen > 0:
                    lls = lls[:n_gen]
                if not lls:
                    lls = [-100.0]
                if len(lls) == self.max_new_tokens:
                    logging.warning('Generation interrupted by max_token limit.')
                generated_ids = answer_tokens[:max(n_gen, 0)].cpu().tolist()
                return sliced_answer, lls, generated_ids

            results = []
            for i in range(B):
                attn = inputs.get('attention_mask', None)
                if attn is not None:
                    actual_prompt_ids = inputs['input_ids'][i][attn[i].bool()].cpu().tolist()
                else:
                    actual_prompt_ids = inputs['input_ids'][i][
                        inputs['input_ids'][i] != (pad_token_id or self.tokenizer.eos_token_id)
                    ].cpu().tolist()

                if N == 1:
                    ans, lls, gen_ids = _extract_ids(i)
                    results.append((ans, lls, {'generated_ids': gen_ids, 'prompt_ids': actual_prompt_ids}))
                else:
                    samples = []
                    for j in range(N):
                        ans, lls, gen_ids = _extract_ids(i * N + j)
                        samples.append((ans, lls, {
                            'generated_ids': gen_ids,
                            'prompt_ids': actual_prompt_ids if j == 0 else None,
                        }))
                    results.append(samples)
            return results
        # --- End token ID early return ---

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        def _extract(seq_idx, prompt_idx):
            """Extract (answer, log_likelihoods, embeddings_dict) for one sequence.

            All sequences share the same max_input_len due to left-padding.
            Generated tokens start at index max_input_len in outputs.sequences.

            Returns embeddings as a dict with keys: 'first_answer', 'last_prompt', 'last_token'.
            """
            # Generated tokens always start at max_input_len (the padded input length),
            # NOT at real_input_len. Using real_input_len would include the tail of the
            # prompt (the un-padded portion) in the "answer", producing garbage outputs.
            answer_tokens = outputs.sequences[seq_idx][max_input_len:]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # Find earliest stop sequence.
            stop_at = len(answer)
            if self.stop_sequences is not None:
                for stop in self.stop_sequences:
                    idx = answer.find(stop)
                    if idx != -1 and idx < stop_at:
                        stop_at = idx
            sliced_answer = answer[:stop_at].strip()

            # n_gen: number of generated tokens, capped at stop position.
            # outputs.scores has one entry per generated step; its length equals
            # the number of tokens actually generated across the batch.
            n_gen_total = len(answer_tokens)  # tokens the model actually generated
            if 'gemma-3' in mn_lower:
                n_gen_total = len(outputs.scores) if outputs.scores is not None else n_gen_total

            if stop_at < len(answer) and n_gen_total > 0:
                try:
                    tokens_to_stop = self.tokenizer(answer[:stop_at], return_tensors='pt')['input_ids'].shape[1]
                    n_gen = min(tokens_to_stop, n_gen_total)
                except Exception:
                    n_gen = n_gen_total
            else:
                n_gen = n_gen_total

            if n_gen <= 0 and sliced_answer:
                n_gen = 1

            log_likelihoods = [s.item() for s in transition_scores[seq_idx]]
            if len(log_likelihoods) == 0:
                logging.warning('No transition scores for seq %d.', seq_idx)
                return sliced_answer, [-100.0], {'first_answer': None, 'last_prompt': None, 'last_token': None}

            if n_gen > 0:
                log_likelihoods = log_likelihoods[:n_gen]

            if len(log_likelihoods) == self.max_new_tokens:
                logging.warning('Generation interrupted by max_token limit.')

            if n_gen <= 0 or hidden is None or len(hidden) == 0:
                logging.warning('Zero-token generation or empty hidden states for seq %d.', seq_idx)
                return sliced_answer, [-100.0], {'first_answer': None, 'last_prompt': None, 'last_token': None}

            # Extract last_token embedding
            if len(hidden) == 1:
                last_input = hidden[0]
            elif n_gen - 1 >= len(hidden):
                logging.error('n_gen too large for seq %d: %d >= %d', seq_idx, n_gen - 1, len(hidden))
                last_input = hidden[-1]
            else:
                last_input = hidden[n_gen - 1]

            last_token_emb = last_input[-1][seq_idx, -1, :].cpu()

            # Extract first_answer embedding (first generated token)
            first_answer_emb = hidden[0][-1][seq_idx, -1, :].cpu() if hidden else None

            # Extract last_prompt embedding from prompt_hidden
            last_prompt_emb = prompt_last_token_embeddings[prompt_idx] if prompt_last_token_embeddings is not None else None

            embeddings_dict = {
                'first_answer': first_answer_emb,
                'last_prompt': last_prompt_emb,
                'last_token': last_token_emb
            }

            return sliced_answer, log_likelihoods, embeddings_dict

        results = []
        for i in range(B):
            if N == 1:
                results.append(_extract(i, i))
            else:
                samples = [_extract(i * N + j, i) for j in range(N)]
                results.append(samples)

        return results

    def extract_embeddings_batch(self, sequence_infos, extract_all_hidden=False):
        """Extract embeddings for a batch of (prompt, generated) sequences.

        sequence_infos: list of {'prompt_ids': list[int], 'generated_ids': list[int]}

        Each entry runs a single forward pass (no generation) over the full
        sequence (prompt + generated tokens) and returns the hidden states at
        the three positions used downstream:
          - last_prompt : last prompt token (just before generation)
          - first_answer: first generated token
          - last_token  : last generated token

        Returns: list of {'first_answer': Tensor, 'last_prompt': Tensor, 'last_token': Tensor}
        """
        mn_lower = self.model_name.lower()
        pad_id = (getattr(self.tokenizer, 'pad_token_id', None) or self.tokenizer.eos_token_id)

        # Build full sequences (prompt + generated) and track boundary positions.
        full_seqs = []
        prompt_lens = []
        gen_lens = []
        for info in sequence_infos:
            p_ids = info.get('prompt_ids') or []
            g_ids = info.get('generated_ids') or []
            full_seqs.append(torch.tensor(p_ids + g_ids, dtype=torch.long))
            prompt_lens.append(len(p_ids))
            gen_lens.append(len(g_ids))

        max_len = max(s.shape[0] for s in full_seqs) if full_seqs else 1
        padded = torch.stack([
            torch.cat([torch.full((max_len - s.shape[0],), pad_id, dtype=torch.long), s])
            for s in full_seqs
        ])
        attn_mask = (padded != pad_id).long()

        device = next(self.model.parameters()).device
        padded = padded.to(device)
        attn_mask = attn_mask.to(device)
        if hasattr(self, 'processor') and 'gemma-3' in mn_lower:
            padded = padded.to(torch.bfloat16) if torch.is_floating_point(padded) else padded

        with torch.no_grad():
            outputs = self.model(
                input_ids=padded,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # Move all hidden states to CPU immediately so the GPU is freed before
        # we do any per-sample indexing. Keeping `outputs` alive (all 32 layers
        # on GPU) while iterating caused the memory footprint to ratchet up.
        if 'decoder_hidden_states' in outputs.keys():
            all_hidden = tuple(h.cpu() for h in outputs.decoder_hidden_states)
        else:
            all_hidden = tuple(h.cpu() for h in outputs.hidden_states)

        del outputs, padded, attn_mask
        torch.cuda.empty_cache()

        results = []
        for i, (p_len, g_len) in enumerate(zip(prompt_lens, gen_lens)):
            pad_len = max_len - p_len - g_len
            last_prompt_pos = pad_len + p_len - 1

            if g_len > 0:
                first_answer_pos = pad_len + p_len
                last_token_pos = pad_len + p_len + g_len - 1

            if extract_all_hidden:
                # Slice per-sample hidden states across all layers at the 3 key positions.
                # Use .clone() so each 1-D slice owns its storage — without it the view
                # carries the entire [B, seq_len, hidden] batch tensor into the pickle,
                # inflating the file by ~30× (e.g. 22 GB instead of ~700 MB).
                last_prompt_pos_all = (
                    tuple(layer[i, last_prompt_pos, :].clone() for layer in all_hidden)
                    if p_len > 0 else None
                )
                if g_len > 0:
                    first_answer_pos_all = tuple(layer[i, first_answer_pos, :].clone() for layer in all_hidden)
                    last_token_pos_all = tuple(layer[i, last_token_pos, :].clone() for layer in all_hidden)
                    first_answer_emb = first_answer_pos_all[-1]
                    last_token_emb = last_token_pos_all[-1]
                else:
                    first_answer_pos_all = None
                    last_token_pos_all = None
                    first_answer_emb = None
                    last_token_emb = None
                last_prompt_emb = last_prompt_pos_all[-1] if last_prompt_pos_all is not None else None
                all_hidden_dict = {
                    'last_prompt': last_prompt_pos_all,
                    'first_answer': first_answer_pos_all,
                    'last_token': last_token_pos_all,
                }
            else:
                # Sampled sequences: only extract scalar embeddings from last layer.
                # Skipping all-layer slicing reduces per-sequence storage from ~1.62 MB to ~48 KB.
                last_layer = all_hidden[-1]  # shape [B, seq_len, hidden]
                last_prompt_emb = last_layer[i, last_prompt_pos, :].clone() if p_len > 0 else None
                if g_len > 0:
                    first_answer_emb = last_layer[i, first_answer_pos, :].clone()
                    last_token_emb = last_layer[i, last_token_pos, :].clone()
                else:
                    first_answer_emb = None
                    last_token_emb = None
                all_hidden_dict = None

            results.append({
                'first_answer': first_answer_emb,
                'last_prompt': last_prompt_emb,
                'last_token': last_token_emb,
                'all_hidden': all_hidden_dict,
            })

        del all_hidden
        return results

    def predict_batch_samples(self, input_data, temperature, num_return_sequences):
        """Generate num_return_sequences high-temperature samples in one forward pass.

        Stopping criteria are skipped (stopping on seq[0] would truncate all others);
        stop words are stripped post-hoc from each decoded answer.

        Returns a list of (answer, token_log_likelihoods, embedding) of length
        num_return_sequences, matching the per-item format of predict().
        """
        mn_lower = self.model_name.lower()

        if hasattr(self, 'processor') and 'gemma-3' in mn_lower:
            if isinstance(input_data, str) and '-it' in mn_lower:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": input_data}]}
                ]
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                )
            elif not isinstance(input_data, str):
                inputs = self.processor.apply_chat_template(
                    input_data, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                )
            else:
                inputs = self.processor(text=input_data, return_tensors="pt")
            inputs = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
                      for k, v in inputs.items()}
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif hasattr(self, 'processor'):
            inputs = self.processor(text=input_data, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if 'llama' in mn_lower or 'falcon' in mn_lower or 'mistral' in mn_lower or 'qwen' in mn_lower or 'gemma' in mn_lower:
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None) or self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        # Get prompt embedding (last token of input)
        with torch.no_grad():
            prompt_outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                output_hidden_states=True,
            )
        if 'decoder_hidden_states' in prompt_outputs.keys():
            prompt_hidden = prompt_outputs.decoder_hidden_states
        else:
            prompt_hidden = prompt_outputs.hidden_states
        prompt_last_token_embedding = prompt_hidden[-1][0, -1, :].cpu() if prompt_hidden else None

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
        )
        if 'gemma-3' in mn_lower:
            generate_kwargs['min_new_tokens'] = 1

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        n_input_tokens = inputs['input_ids'].shape[1]

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        results = []

        if 'gemma-3' in mn_lower:
            n_generated = len(outputs.scores) if outputs.scores is not None else (
                outputs.sequences.shape[1] - n_input_tokens)

            for i in range(num_return_sequences):
                answer_tokens = outputs.sequences[i][n_input_tokens:]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                sliced_answer = answer
                if self.stop_sequences is not None:
                    stop_idx = len(answer)
                    for stop in self.stop_sequences:
                        idx = answer.find(stop)
                        if idx != -1 and idx < stop_idx:
                            stop_idx = idx
                    sliced_answer = answer[:stop_idx]
                sliced_answer = sliced_answer.strip()

                log_likelihoods = [score.item() for score in transition_scores[i]]
                if len(log_likelihoods) == 0:
                    logging.warning('Gemma returned no transition scores for seq %d.', i)
                    results.append((sliced_answer, [-100.0], {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}))
                    continue

                n_gen_i = min(n_generated, len(log_likelihoods))
                log_likelihoods = log_likelihoods[:n_gen_i]

                if len(log_likelihoods) == self.max_new_tokens:
                    logging.warning('Generation interrupted by max_token limit.')

                if hidden is None or len(hidden) == 0:
                    results.append((sliced_answer, log_likelihoods, {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}))
                    continue

                n_hidden_steps = min(n_gen_i, len(hidden))
                if n_hidden_steps <= 0:
                    results.append((sliced_answer, log_likelihoods, {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}))
                    continue

                last_layer = hidden[n_hidden_steps - 1][-1]  # [N, 1, hidden]
                last_token_emb = last_layer[i, -1, :].cpu()
                first_answer_emb = hidden[0][-1][i, -1, :].cpu() if hidden else None
                log_likelihoods = log_likelihoods[:n_hidden_steps]
                embeddings_dict = {
                    'first_answer': first_answer_emb,
                    'last_prompt': prompt_last_token_embedding,
                    'last_token': last_token_emb
                }
                results.append((sliced_answer, log_likelihoods, embeddings_dict))

        else:
            for i in range(num_return_sequences):
                full_answer = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                answer_tokens = outputs.sequences[i][n_input_tokens:]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                input_data_offset = len(full_answer) - len(answer)

                stop_at = len(answer)
                sliced_answer = answer
                if self.stop_sequences is not None:
                    for stop in self.stop_sequences:
                        idx = answer.find(stop)
                        if idx != -1 and idx < stop_at:
                            stop_at = idx
                    sliced_answer = answer[:stop_at]
                sliced_answer = sliced_answer.strip()

                # Calculate tokens generated: direct token count is more reliable than character offsets
                n_gen_total = len(outputs.sequences[i]) - n_input_tokens

                # Try to map character-level stop position to token count
                if stop_at < len(answer) and n_gen_total > 0:
                    try:
                        answer_up_to_stop = answer[:stop_at]
                        tokens_to_stop = self.tokenizer(answer_up_to_stop, return_tensors='pt')['input_ids'].shape[1]
                        n_gen_i = min(tokens_to_stop, n_gen_total)
                    except Exception as e:
                        logging.debug('Error mapping stop position to tokens for seq %d: %s. Using full count.', i, e)
                        n_gen_i = n_gen_total
                else:
                    n_gen_i = n_gen_total

                # Ensure n_gen_i is at least 1 if we have an answer
                if n_gen_i <= 0 and sliced_answer.strip():
                    n_gen_i = 1

                if n_gen_i <= 0 or hidden is None or len(hidden) == 0:
                    logging.debug('Zero-token generation or empty hidden states for seq %d (n_gen_i=%d).',  i, n_gen_i)
                    results.append((sliced_answer, [-100.0], {'first_answer': None, 'last_prompt': prompt_last_token_embedding, 'last_token': None}))
                    continue

                if len(hidden) == 1:
                    last_input = hidden[0]
                elif n_gen_i - 1 >= len(hidden):
                    logging.error(
                        'n_gen_i too large for seq %d: %d >= %d', i, n_gen_i - 1, len(hidden))
                    last_input = hidden[-1]
                else:
                    last_input = hidden[n_gen_i - 1]

                last_layer = last_input[-1]  # [N, seq, hidden]
                last_token_emb = last_layer[i, -1, :].cpu()
                first_answer_emb = hidden[0][-1][i, -1, :].cpu() if hidden else None

                log_likelihoods = [score.item() for score in transition_scores[i]]
                if len(log_likelihoods) > 1:
                    log_likelihoods = log_likelihoods[:n_gen_i]

                if len(log_likelihoods) == self.max_new_tokens:
                    logging.warning('Generation interrupted by max_token limit.')

                if len(log_likelihoods) == 0:
                    raise ValueError(f'Empty log likelihoods for seq {i}.')

                embeddings_dict = {
                    'first_answer': first_answer_emb,
                    'last_prompt': prompt_last_token_embedding,
                    'last_token': last_token_emb
                }
                results.append((sliced_answer, log_likelihoods, embeddings_dict))

        return results

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']
        target_ids_true = tokenized_prompt_true.clone()
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()