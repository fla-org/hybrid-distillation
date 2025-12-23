import argparse
import os
import json
import random
import re
import copy
import logging
from typing import Callable, List, Tuple, Set, Dict

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from english_words import get_english_words_set

import fla

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synthetic_word_pool(n: int) -> List[str]:
    """
    Produce a pool like: ['aaa','bbb',...,'zzz','aaa1','bbb1',...], length n.
    Deterministic, simple, and visually easy to scan.
    """
    pool = []
    letters = [chr(ord('a') + i) for i in range(26)]
    i = 0
    while len(pool) < n:
        base = letters[i % 26] * 2
        suffix = i // 26
        pool.append(base if suffix == 0 else f"{base}{suffix}")
        i += 1
    return pool

def parse_first_int(text: str) -> str:
    m = re.search(r'\d+', text)
    return m.group(0) if m else ""

def parse_vars(text: str, prefix: str = "X") -> Set[str]:
    """Return set like {'X1','X2',...} from generated text."""
    return set(re.findall(rf'\b{re.escape(prefix)}\d+\b', text))

def extract_word_sequence_from_prompt(prompt: str) -> List[str]:
    """
    For CWE/FWE prompts we always put the word sequence on ONE line,
    between '... separated by spaces:\n' and '\nWhat are ...'.
    """
    m = re.search(r'separated by spaces:\n(.*?)\nWhat are', prompt, flags=re.S)
    return m.group(1).split() if m else []

def parse_topk_from_text(text: str, allowed: Set[str], k: int) -> List[str]:
    """
    Extract up to k unique tokens, preserving order, filtered to 'allowed'.
    """
    text = text[:text.find("Explanation")] if text.find("Explanation") != -1 else text
    toks = re.findall(r'[A-Za-z0-9_]+', text)
    seen, out = set(), []
    for t in toks:
        if t in allowed and t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) == k:
                break
    return out


def create_retrieval_dataset(num_samples: int, num_pairs: int, output_dir: str, **kwargs) -> list:
    dataset_path = os.path.join(output_dir, 'dataset', f'retrieval_task_{num_pairs}.json')
    if os.path.exists(dataset_path):
        logging.info(f"Dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    word_source = list(get_english_words_set(['web2']))
    logging.info(f"Creating a new retrieval dataset with {num_samples} samples, each with {num_pairs} pairs.")
    dataset = []
    for _ in range(num_samples):
        keys = random.sample(word_source, num_pairs)
        values = [random.randint(0, 100) for _ in range(num_pairs)]
        data_dict = dict(zip(keys, values))
        query_key = random.choice(keys)
        correct_value = data_dict[query_key]
        dict_str = "\n".join([f"{k}:{v}" for k, v in data_dict.items()])
        prompt = f"Memorize the following dictionary:\n{dict_str}\nThe value of the key '{query_key}' is"
        dataset.append({"prompt": prompt, "answer": str(correct_value)})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"Dataset saved to {dataset_path}")
    return dataset

def create_ar_dataset(num_samples: int, num_pairs: int, output_dir: str, tokenizer=None, **kwargs) -> list:
    if tokenizer is None:
        raise ValueError("The 'associative_recall' task requires a tokenizer to be provided for dataset creation.")
    
    safe_model_name = tokenizer.name_or_path.replace("/", "_")
    dataset_path = os.path.join(output_dir, 'dataset', f'ar_task_{safe_model_name}_{num_pairs}.json')
    if os.path.exists(dataset_path):
        logging.info(f"AR Dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    logging.info("Generating a word source from the tokenizer's vocabulary...")
    vocab = tokenizer.get_vocab()
    word_source = [tokenizer.decode([token_id]).strip() for token_id in vocab.values() if re.fullmatch(r'[a-zA-Z]{3,}', tokenizer.decode([token_id]).strip())]
    logging.info(f"Found {len(word_source)} suitable words in vocabulary.")
    if len(word_source) < num_pairs:
        raise ValueError(f"Not enough suitable words in tokenizer vocab ({len(word_source)}) to create pairs ({num_pairs}).")

    logging.info(f"Creating a new AR dataset with {num_samples} samples, each with {num_pairs} pairs.")
    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating AR Dataset"):
        keys = random.sample(word_source, num_pairs)
        values = [random.randint(0, 9) for _ in range(num_pairs)]
        data_dict = dict(zip(keys, values))
        query_keys = random.sample(keys, 3)
        correct_value = sum(data_dict[k] for k in query_keys)
        dict_str = "\n".join([f"{k}: {v}" for k, v in data_dict.items()])
        prompt = (f"Given the following key-value pairs:\n{dict_str}\n"
                  f"What is the value of {query_keys[0]} + {query_keys[1]} + {query_keys[2]}?")
        dataset.append({"prompt": prompt, "answer": str(correct_value)})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"AR Dataset saved to {dataset_path}")
    return dataset

def create_vt_dataset(
    num_samples: int,
    num_pairs: int,
    output_dir: str,
    **kwargs
) -> list:
    """
    Multi-hop Tracing: Variable Tracking (VT)
    - num_chains: how many independent chains (1 target 'X' + distractors)
    - num_hops: number of name-binding hops per chain (X1 = V; X2 = X1; ... up to X{num_hops+1})
    - noise_ratio: number of noise statements relative to total signal statements
    """
    num_chains = int(kwargs.get("num_chains", 2))
    num_hops = int(kwargs.get("num_hops", 2))
    noise_ratio = float(kwargs.get("noise_ratio", 1.0))

    dataset_path = os.path.join(
        output_dir, 'dataset',
        f'vt_task_{num_chains}chains_{num_hops}hops_nr{str(noise_ratio).replace(".","p")}.json'
    )
    if os.path.exists(dataset_path):
        logging.info(f"VT dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    # chain letters: first one is target 'X', others are distractors
    distractor_letters = ['Y', 'Z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    assert num_chains >= 1, "num_chains must be >= 1"

    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating VT Dataset"):
        target_value = random.randint(10000, 99999)
        # Build statements for the target X-chain
        statements = [f"VAR X1 = {target_value}"]
        for i in range(2, num_hops + 2):
            statements.append(f"VAR X{i} = X{i-1}")

        # Distractor chains
        for c in range(1, num_chains):
            L = distractor_letters[(c-1) % len(distractor_letters)]
            val = random.randint(10000, 99999)
            while val == target_value:
                val = random.randint(10000, 99999)
            statements.append(f"VAR {L}1 = {val}")
            for i in range(2, num_hops + 2):
                statements.append(f"VAR {L}{i} = {L}{i-1}")

        # Noise proportional to statement count
        n_noise = int(round(noise_ratio * len(statements)))
        noise = []
        for i in range(n_noise):
            # inject innocuous lines, numbers, and unrelated variables (avoid 'X')
            letter = random.choice(['Q','R','S','T','U','V','W'])
            if random.random() < 0.5:
                noise.append("......")
            else:
                noise.append(f"VAR {letter}{random.randint(1, 9)} = {random.randint(10000, 99999)}")

        # Interleave & shuffle
        full = statements + noise
        random.shuffle(full)
        context = " ...... ".join(full)

        prompt = (
            f" ...... {context} ......\n"
            f"Find all variables that are assigned the value {target_value}.\n"
            f"Answer:"
        )
        gold_vars = " ".join([f"X{i}" for i in range(1, num_hops + 2)])
        dataset.append({"prompt": prompt, "answer": gold_vars})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"VT Dataset saved to {dataset_path}")
    return dataset


def create_cwe_dataset(
    num_samples: int,
    num_pairs: int,
    output_dir: str,
    **kwargs
) -> list:
    """
    Aggregation: Common Words Extraction (CWE).
    - num_cw: number of common words (K; also the expected output size)
    - num_ucw: number of uncommon words (increases with sequence length)
    - freq_cw: frequency of each common word
    - freq_ucw: frequency of each uncommon word (should be < freq_cw to avoid ties)
    """
    num_cw = int(kwargs.get("num_cw", 10))
    num_ucw = int(kwargs.get("num_ucw", 10 * num_cw))
    freq_cw = int(kwargs.get("freq_cw", 2))
    freq_ucw = int(kwargs.get("freq_ucw", 1))
    assert freq_cw > freq_ucw, "freq_cw must be strictly greater than freq_ucw."

    dataset_path = os.path.join(
        output_dir, 'dataset',
        f'cwe_task_cw{num_cw}_ucw{num_ucw}_f{freq_cw}-{freq_ucw}.json'
    )
    if os.path.exists(dataset_path):
        logging.info(f"CWE dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    vocab_size = num_cw + num_ucw + 16
    vocab = synthetic_word_pool(vocab_size)

    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating CWE Dataset"):
        common_words = random.sample(vocab, num_cw)
        rest = [w for w in vocab if w not in common_words]
        uncommon_words = random.sample(rest, num_ucw)

        bag = []
        for w in common_words:
            bag.extend([w] * freq_cw)
        for w in uncommon_words:
            bag.extend([w] * freq_ucw)

        random.shuffle(bag)
        seq = " ".join(bag)

        K = num_cw  # "K equals the number of common words"
        prompt = (
            f"Given the following list of words separated by spaces:\n{seq}\n"
            f"What are the {K} most common words in the above list?\n"
            f"Answer:"
        )
        # Any order is fine (all common words have the same frequency), but fix a deterministic order for the label.
        gold = " ".join(sorted(common_words))
        dataset.append({"prompt": prompt, "answer": gold})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"CWE Dataset saved to {dataset_path}")
    return dataset


def create_fwe_dataset(
    num_samples: int,
    num_pairs: int,
    output_dir: str,
    **kwargs
) -> list:
    """
    Aggregation: Frequent Words Extraction (FWE).
    - zeta_alpha: Zipf/Zeta parameter (Î± > 1). Larger => heavier head.
    - vocab_size: size of synthetic vocabulary
    - num_words: total tokens in the sequence (context length)
    - K is fixed to 3 (as in RULER)
    """
    alpha = float(kwargs.get("zeta_alpha", 2.0))
    vocab_size = int(kwargs.get("vocab_size", 200))
    num_words = int(kwargs.get("num_words", num_pairs))
    K = int(kwargs.get("K", 3))
    assert K == 3, "FWE uses K=3 as per task definition."

    dataset_path = os.path.join(
        output_dir, 'dataset',
        f'fwe_task_a{str(alpha).replace(".","p")}_V{vocab_size}_N{num_words}.json'
    )
    if os.path.exists(dataset_path):
        logging.info(f"FWE dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    vocab = synthetic_word_pool(vocab_size)

    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating FWE Dataset"):
        # Sample ranks ~ Zipf(alpha); clip to [1, vocab_size]
        ranks = np.random.zipf(alpha, size=num_words)
        ranks = np.clip(ranks, 1, vocab_size)
        tokens = [vocab[r - 1] for r in ranks]
        seq = " ".join(tokens)

        # Count frequencies and get top-3 (break ties deterministically by token)
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        top3 = [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]

        prompt = (
            f"Given the following list of words separated by spaces:\n{seq}\n"
            f"What are the 3 most frequently appeared words in the above coded text?\n"
            f"Answer:"
        )
        dataset.append({"prompt": prompt, "answer": " ".join(top3)})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"FWE Dataset saved to {dataset_path}")
    return dataset


def create_phonebook_dataset(
    num_samples: int,
    num_pairs: int,
    output_dir: str,
    **kwargs
) -> list:
    """
    Retrieval-style: Phone Book.
    Builds a list of entries "Firstname Lastname: 10-digit-number", then
    appends two few-shot exemplars and asks for the phone of a 3rd name.

    Args via kwargs:
      - pb_min_len (int): min number of entries per phone book (default 8)
      - pb_max_len (int): max number of entries per phone book (default 30)
      - pb_few_shots (int): fixed at 3 to match original format (two shown, one queried)
    """
    pb_min_len = int(kwargs.get("pb_min_len", 8))
    pb_max_len = int(kwargs.get("pb_max_len", 30))
    pb_few_shots = int(kwargs.get("pb_few_shots", 3))
    assert pb_few_shots == 3, "This task assumes 3 few-shot picks (2 examples + 1 query)."

    dataset_path = os.path.join(
        output_dir, 'dataset',
        f'phonebook_task_len{pb_min_len}-{pb_max_len}_fs{pb_few_shots}.json'
    )
    if os.path.exists(dataset_path):
        logging.info(f"PhoneBook dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    # Simple name pool from english words; avoid extra dependency on 'names'
    words = list(get_english_words_set(['web2']))
    name_pool = [w for w in words if w.isalpha() and 3 <= len(w) <= 10]
    if len(name_pool) < 1000:
        # Fallback if env's wordset is small
        letters = [chr(ord('a') + i) for i in range(26)]
        name_pool = [f"{a}{b}{c}" for a in letters for b in letters for c in letters]

    def make_name() -> str:
        return f"{random.choice(name_pool).title()} {random.choice(name_pool).title()}"

    def rand_phone() -> str:
        first = random.randint(6, 9)
        rest = [random.randint(0, 9) for _ in range(9)]
        return str(first) + ''.join(str(d) for d in rest)

    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating PhoneBook Dataset"):
        length = random.randint(pb_min_len, pb_max_len)
        entries = [f"{make_name()}: {rand_phone()}" for _ in range(length)]

        # pick 3 distinct indices: two few-shot display + one query
        idx_fs = random.sample(range(length), pb_few_shots)
        few_shot_block = f"\n\n{entries[idx_fs[0]]}\n{entries[idx_fs[1]]}\n\n"
        prompt = "\n".join(entries) + few_shot_block
        query_name = entries[idx_fs[2]].split(":")[0] + ":"
        prompt += query_name
        label = entries[idx_fs[2]].split(":")[1].strip()

        dataset.append({"prompt": prompt, "answer": label})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"PhoneBook Dataset saved to {dataset_path}")
    return dataset


def create_ar_mutihop_dataset(
    num_samples: int,
    num_pairs: int,
    output_dir: str,
    tokenizer=None,
    ar_max_hops: int = 2,
    **kwargs
) -> list:
    """
    Multi-hop associative recall:
      - Some entries map directly to an integer in [0..9]
      - Some entries alias another key (possibly aliasing again), forming chains
      - All chains are acyclic and have length <= ar_max_hops
      - Query asks for the sum of three keys; the model must resolve aliases first.

    Saved under: dataset/ar_mutihop_task_{safe_model_name}_{num_pairs}_hops{ar_max_hops}.json
    """
    if tokenizer is None:
        raise ValueError("The 'associative_recall_mutihop' task requires a tokenizer.")

    safe_model_name = tokenizer.name_or_path.replace("/", "_")
    dataset_path = os.path.join(
        output_dir, 'dataset',
        f'ar_mutihop_task_{safe_model_name}_{num_pairs}_hops{ar_max_hops}.json'
    )
    if os.path.exists(dataset_path):
        logging.info(f"AR-Mutihop dataset already exists at {dataset_path}. Loading it.")
        with open(dataset_path, 'r') as f:
            return json.load(f)

    logging.info("Generating a word source from the tokenizer's vocabulary (AR-multi-hop)...")
    vocab = tokenizer.get_vocab()
    word_source = [
        token.strip()
        for token_id in vocab.values()
        for token in [tokenizer.decode([token_id]).strip()]
        if re.fullmatch(r'[a-zA-Z]{3,}', token)
    ]
    logging.info(f"Found {len(word_source)} suitable words in vocabulary.")
    if len(word_source) < num_pairs:
        raise ValueError(
            f"Not enough suitable words in tokenizer vocab ({len(word_source)}) "
            f"to create pairs ({num_pairs})."
        )

    def resolve(key: str, mapping: Dict[str, object]) -> int:
        """Follow aliases until an int is found. Safety: guard cycles."""
        visited = set()
        cur = key
        while True:
            if cur in visited:
                # Should never happen given construction, but be safe:
                raise RuntimeError(f"Cycle detected when resolving '{key}'")
            visited.add(cur)

            val = mapping[cur]
            if isinstance(val, int):
                return val
            # val must be a key name (string)
            cur = val

    dataset = []
    alias_prob = 0.4  # ~60% base values, ~40% aliases is a good default

    for _ in tqdm(range(num_samples), desc="Generating AR Mutihop Dataset"):
        keys = random.sample(word_source, num_pairs)

        # mapping: str -> (int or str key alias)
        mapping: Dict[str, object] = {}
        depth: Dict[str, int] = {}
        keys_by_depth: Dict[int, List[str]] = {0: []}

        for k in keys:
            # Can we create an alias that keeps depth <= ar_max_hops?
            parent_candidates = [cand for cand in keys_by_depth.keys() if cand <= ar_max_hops - 1 and keys_by_depth.get(cand)]
            can_alias = len(parent_candidates) > 0

            make_alias = can_alias and (random.random() < alias_prob)

            if not make_alias:
                # Assign a base integer
                v = random.randint(0, 9)
                mapping[k] = v
                depth[k] = 0
                keys_by_depth.setdefault(0, []).append(k)
            else:
                # Pick any parent whose depth <= ar_max_hops-1, prefer higher depths to induce multi-hop
                weighted_depths = []
                for d in parent_candidates:
                    # Weight by (d+1) to bias towards deeper parents (more hops)
                    weighted_depths.extend([d] * (d + 1))
                pd = random.choice(weighted_depths)
                parent = random.choice(keys_by_depth[pd])

                mapping[k] = parent            # alias to another key
                depth[k] = depth[parent] + 1   # new depth = parent depth + 1
                keys_by_depth.setdefault(depth[k], []).append(k)

        # Ensure at least one 2-hop chain if allowed
        if ar_max_hops >= 2 and max(depth.values() or [0]) < 2:
            # Promote a 2-hop chain by rewiring two keys if needed
            base_keys = [k for k, d in depth.items() if d == 0]
            other_keys = [k for k in keys if k not in base_keys]
            if base_keys and len(other_keys) >= 2:
                mid = random.choice(other_keys)
                top = random.choice([k for k in other_keys if k != mid])

                mapping[mid] = random.choice(base_keys)
                depth[mid] = 1
                keys_by_depth.setdefault(1, []).append(mid)

                mapping[top] = mid
                depth[top] = 2
                keys_by_depth.setdefault(2, []).append(top)

        # Build the prompt lines in the insertion order of `keys`
        lines = []
        for k in keys:
            v = mapping[k]
            lines.append(f"{k}: {v}" if isinstance(v, int) else f"{k}: {v}")

        # Choose three query keys; ensure at least one alias appears in the query if possible
        query_keys = random.sample(keys, 3)
        alias_keys = [k for k in keys if depth[k] >= 1]
        if alias_keys and all(depth[q] == 0 for q in query_keys):
            query_keys[0] = random.choice(alias_keys)

        total = sum(resolve(q, mapping) for q in query_keys)

        prompt = (
            "Given the following key-value pairs:\n"
            + "\n".join(lines)
            + f"\nWhat is the value of {query_keys[0]} + {query_keys[1]} + {query_keys[2]}? "
              "Only return the final value."
        )
        dataset.append({"prompt": prompt, "answer": str(total)})

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"AR Mutihop Dataset saved to {dataset_path}")
    return dataset


def parse_generated_text(text: str) -> str:
    """Backward-compat: first integer (for retrieval)."""
    return parse_first_int(text)

def evaluate(model, tokenizer, dataset: list, task_name: str, device: torch.device, batch_size: int = 8) -> float:
    """
    Evaluates the model's performance on a given task using Exact Match accuracy.
    - synthetic_retrieval: exact match on integer value
    - associative_recall: answer substring match
    - variable_tracking: set equality over variables {X1, X2, ...}
    - common_words_extraction: set equality over K tokens
    - frequent_words_extraction: ordered top-3 equality
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Model in Batches"):
            batch = dataset[i:i + batch_size]
            prompts = [item['prompt'] for item in batch]
            expected_answers = [item['answer'] for item in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for j, text in enumerate(generated_texts):
                gold = expected_answers[j]

                if task_name == "synthetic_retrieval":
                    if parse_generated_text(text) == gold:
                        correct += 1

                elif task_name in {"associative_recall", "associative_recall_mutihop"}:
                    if gold in text:
                        correct += 1

                elif task_name == "variable_tracking":
                    # Compare sets of variables (order-agnostic)
                    pred_vars = parse_vars(text, prefix="X")
                    gold_vars = set(gold.split())
                    if pred_vars == gold_vars:
                        correct += 1

                elif task_name == "common_words_extraction":
                    # Order-agnostic: K == number of gold tokens
                    seq_words = set(extract_word_sequence_from_prompt(prompts[j]))
                    K = len(gold.split())
                    pred = set(parse_topk_from_text(text, seq_words, K))
                    if pred == set(gold.split()):
                        correct += 1

                elif task_name == "frequent_words_extraction":
                    seq_words = set(extract_word_sequence_from_prompt(prompts[j]))
                    pred = parse_topk_from_text(text, seq_words, 3)
                    if pred == gold.split():
                        correct += 1

                elif task_name == "phone_book":
                    # mimic your original script: compare the first generated line to the 10-digit label
                    first_line = generated_texts[j].splitlines()[0].strip()
                    if first_line == gold:
                        correct += 1
                else:
                    # Unknown task (shouldn't happen with parser choices)
                    pass
    return correct / len(dataset)

def zero_out_attention_hook(module, input, output):
    """
    Zero the attention *output* regardless of whether the module returns
    a tensor or a tuple whose first element is the hidden states.
    """
    if isinstance(output, tuple):
        if isinstance(output[0], torch.Tensor):
            output[0].zero_()
        return output
    elif isinstance(output, torch.Tensor):
        output.zero_()
        return output
    else:
        return output


def run_experiment(args: argparse.Namespace, task_config: dict):
    task_name = args.task
    task_display_name = task_name.replace('_', ' ').title()
    logging.info(f"--- Starting Task: {task_display_name} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    for sub_dir in ['dataset', 'results', 'plots']:
        os.makedirs(os.path.join(args.output_dir, sub_dir), exist_ok=True)

    safe_model_name = args.model_name.replace("/", "_")

    if task_name in ["synthetic_retrieval", "associative_recall"]:
        experiment_suffix = f'{safe_model_name}_{task_name}_pairs_{args.num_pairs}'
    elif task_name == "variable_tracking":
        experiment_suffix = f'{safe_model_name}_{task_name}_{args.num_chains}chains_{args.num_hops}hops_nr{str(args.noise_ratio).replace(".","p")}'
    elif task_name == "common_words_extraction":
        experiment_suffix = f'{safe_model_name}_{task_name}_cw{args.num_cw}_ucw{args.num_ucw}_f{args.freq_cw}-{args.freq_ucw}'
    elif task_name == "frequent_words_extraction":
        experiment_suffix = f'{safe_model_name}_{task_name}_a{str(args.zeta_alpha).replace(".","p")}_V{args.vocab_size}_N{args.num_words}'
    elif task_name == "phone_book":
        experiment_suffix = f'{safe_model_name}_{task_name}_len{args.pb_min_len}-{args.pb_max_len}_fs{args.pb_few_shots}'
    elif task_name == "associative_recall_mutihop":
        experiment_suffix = f'{safe_model_name}_{task_name}_pairs_{args.num_pairs}_hops_{args.ar_max_hops}'

    logging.info(f"Loading tokenizer and model for '{args.model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    dataset_creator = task_config['creator']
    dataset = dataset_creator(
        args.num_samples, args.num_pairs, args.output_dir,
        tokenizer=tokenizer,
        # VT params
        num_chains=args.num_chains,
        num_hops=args.num_hops,
        noise_ratio=args.noise_ratio,
        # CWE params
        freq_cw=args.freq_cw,
        freq_ucw=args.freq_ucw,
        num_cw=args.num_cw,
        num_ucw=args.num_ucw,
        # FWE params
        zeta_alpha=args.zeta_alpha,
        vocab_size=args.vocab_size,
        num_words=args.num_words,
        K=args.K,
        # PhoneBook params
        pb_min_len=args.pb_min_len,
        pb_max_len=args.pb_max_len,
        pb_few_shots=args.pb_few_shots,
        # AR-multi-hop param
        ar_max_hops=args.ar_max_hops,
    )

    logging.info(f"Evaluating baseline performance for {task_display_name} task...")
    baseline_accuracy = evaluate(model, tokenizer, dataset, task_name, device, args.batch_size)
    logging.info(f"âœ… {task_display_name} Baseline Accuracy: {baseline_accuracy:.4f}")
    baseline_path = os.path.join(args.output_dir, 'results', f'baseline_{experiment_suffix}.json')
    with open(baseline_path, 'w') as f:
        json.dump({"model_name": args.model_name, "task": task_name, "baseline_accuracy": baseline_accuracy, "num_pairs": args.num_pairs}, f)

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        get_attn = lambda m, i: m.transformer.h[i].attn
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        if 'fla' in args.model_name:
            get_attn = lambda m, i: m.model.layers[i].attn
        else:
            get_attn = lambda m, i: m.model.layers[i].self_attn
    else:
        logging.error(f"Cannot determine layer path for model '{args.model_name}'.")
        return

    num_layers = len(layers)
    logging.info(f"Found {num_layers} layers. Starting layer-wise ablation...")
    layer_performance = []
    for i in range(num_layers):
        logging.info(f"--- Ablating Layer {i}/{num_layers-1} ---")
        attn_layer = get_attn(model, i)
        hook = attn_layer.register_forward_hook(zero_out_attention_hook)
        accuracy = evaluate(model, tokenizer, dataset, task_name, device, args.batch_size)
        performance_drop = baseline_accuracy - accuracy
        layer_performance.append({"layer_index": i, "accuracy": accuracy, "performance_drop": performance_drop})
        logging.info(f"Layer {i} Ablated Accuracy: {accuracy:.4f}, Drop: {performance_drop:.4f}")
        hook.remove()

    results_df = pd.DataFrame(layer_performance)
    results_path = os.path.join(args.output_dir, 'results', f'ablation_{experiment_suffix}.csv')
    results_df.to_csv(results_path, index=False)
    logging.info(f"ðŸ“Š Layer-wise stats saved to {results_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(results_df['layer_index'], results_df['performance_drop'], color=task_config['plot_color'], edgecolor='k')
    ax.set_xlabel("Transformer Layer Index", fontsize=12, weight='bold')
    ax.set_ylabel("Performance Drop (Baseline - Ablated Accuracy)", fontsize=12, weight='bold')
    ax.set_title(f"Impact of Zeroing Out Attention Output per Layer\nModel: {args.model_name} | Task: {task_display_name}", fontsize=14, weight='bold')
    ax.set_xticks(results_df['layer_index'])
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    fig.tight_layout()

    plot_path = os.path.join(args.output_dir, 'plots', f'ablation_{experiment_suffix}.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Visualization saved to {plot_path}.")


def main():
    """Parses command-line arguments and dispatches to the correct task function."""
    TASK_CONFIGS = {
        'synthetic_retrieval': {
            'creator': create_retrieval_dataset,
            'plot_color': 'c',
        },
        'associative_recall': {
            'creator': create_ar_dataset,
            'plot_color': 'm',
        },
        'variable_tracking': {
            'creator': create_vt_dataset,
            'plot_color': 'tab:orange',
        },
        'common_words_extraction': {
            'creator': create_cwe_dataset,
            'plot_color': 'tab:blue',
        },
        'frequent_words_extraction': {
            'creator': create_fwe_dataset,
            'plot_color': 'tab:green',
        },
        'phone_book': {
            'creator': create_phonebook_dataset,
            'plot_color': 'tab:red',
        },
        'associative_recall_mutihop': {
            'creator': create_ar_mutihop_dataset,
            'plot_color': 'tab:purple',
        },
    }

    parser = argparse.ArgumentParser(description="Run layer-wise analysis experiments for different tasks.")
    parser.add_argument("--task", type=str, required=True, choices=TASK_CONFIGS.keys(), help="The specific task to run.")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Name of the Hugging Face model to evaluate.")
    parser.add_argument("--output_dir", type=str, default="experiment_results", help="Directory to save all artifacts.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate for the task.")
    parser.add_argument("--num_pairs", type=int, default=50, help="Task-specific size knob (kept for back-compat).")

    # VT hyperparams
    parser.add_argument("--num_chains", type=int, default=2, help="(VT) Number of variable chains.")
    parser.add_argument("--num_hops", type=int, default=2, help="(VT) Number of hops per chain.")
    parser.add_argument("--noise_ratio", type=float, default=1.0, help="(VT) Noise-to-signal ratio (by statements).")

    # CWE hyperparams
    parser.add_argument("--freq_cw", type=int, default=2, help="(CWE) Frequency of each common word.")
    parser.add_argument("--freq_ucw", type=int, default=1, help="(CWE) Frequency of each uncommon word.")
    parser.add_argument("--num_cw", type=int, default=10, help="(CWE) Number of common words (also K).")
    parser.add_argument("--num_ucw", type=int, default=100, help="(CWE) Number of uncommon words.")

    # FWE hyperparams
    parser.add_argument("--zeta_alpha", type=float, default=2.0, help="(FWE) Zipf/Zeta alpha (>1).")
    parser.add_argument("--vocab_size", type=int, default=200, help="(FWE) Size of synthetic vocabulary.")
    parser.add_argument("--num_words", type=int, default=400, help="(FWE) Total tokens in the coded text.")
    parser.add_argument("--K", type=int, default=3, help="(FWE) Top-K to return (fixed to 3).")

    # PhoneBook hyperparams
    parser.add_argument("--pb_min_len", type=int, default=8, help="(PhoneBook) Min number of entries per list.")
    parser.add_argument("--pb_max_len", type=int, default=30, help="(PhoneBook) Max number of entries per list.")
    parser.add_argument("--pb_few_shots", type=int, default=3, help="(PhoneBook) Must be 3 (two examples + one query).")

    # AR mutihop hyperparams
    parser.add_argument(
        "--ar_max_hops", type=int, default=2,
        help="(AR-multi-hop) Maximum alias chain length. For example, 2 allows k -> k' -> int."
    )

    args = parser.parse_args()

    selected_task_config = TASK_CONFIGS[args.task]
    run_experiment(args, selected_task_config)

if __name__ == "__main__":
    main()