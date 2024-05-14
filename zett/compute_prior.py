from dataclasses import dataclass
from transformers import HfArgumentParser, AutoTokenizer
from datasets import load_dataset, Dataset
from functools import partial
from pathlib import Path
import numpy as np
import ahocorasick
from collections import Counter
import pickle
from tqdm import tqdm
import math

from zett.utils import tokenize_function, get_prior, default_pretokenize


@dataclass
class Args:
    tokenizer_name: str
    mode: str = "reestimate"
    dataset_name: str = "datasets/train_small.parquet"
    block_size: int = 128
    num_workers: int = 64


def compute_substring_prior(
    tokenizer,
    datasets=None,
    pretoken_counts=None,
    num_workers=None,
    return_pretoken_counts=False,
    normalize_fn=None,
):
    if pretoken_counts is None:

        def get_pretoken_counts(examples):
            if normalize_fn is not None:
                normalize = normalize_fn
            if tokenizer._tokenizer.normalizer is None:
                normalize = lambda x: x
            else:
                normalize = tokenizer._tokenizer.normalizer.normalize_str

            if tokenizer._tokenizer.pre_tokenizer is None:
                pre_tokenize = default_pretokenize
                pre_tokenize_first = True  # save to do in the default case
            else:
                pre_tokenize = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str
                pre_tokenize_first = False

            if pre_tokenize_first:
                pretoken_counts = Counter(
                    normalize(token)
                    for e in examples["text"]
                    for token, _ in pre_tokenize(e)
                )
            else:
                pretoken_counts = Counter(
                    token
                    for e in examples["text"]
                    for token, _ in pre_tokenize(normalize(e))
                )

            return {"pretoken_counts": [pickle.dumps(dict(pretoken_counts))]}

        all_counts = []

        for dataset in datasets:
            all_counts.append(
                dataset.map(
                    get_pretoken_counts,
                    remove_columns=dataset.column_names,
                    batch_size=math.ceil(len(dataset) / num_workers),
                    batched=True,
                    num_proc=num_workers,
                )
            )
        all_pretoken_counts = [
            Counter(pickle.loads(x))
            for c in tqdm(all_counts, desc="Aggregating token counts...")
            for x in c["pretoken_counts"]
        ]
        pretoken_counts = Counter()

        for c in tqdm(all_pretoken_counts, desc="Adding token counts..."):
            pretoken_counts += c

    pretoken_strings, pretoken_count_list = list(zip(*pretoken_counts.items()))
    pretoken_dataset = Dataset.from_dict(
        {"pretoken_strings": pretoken_strings, "pretoken_counts": pretoken_count_list}
    )

    def get_token_counts(examples):
        token_counts = Counter()

        automaton = ahocorasick.Automaton()
        for key, index in tokenizer.get_vocab().items():
            automaton.add_word(key, index)

        automaton.make_automaton()

        for pretoken, count in zip(
            examples["pretoken_strings"], examples["pretoken_counts"]
        ):
            for _, index in automaton.iter(pretoken):
                token_counts[index] += count

        return {"token_counts": [pickle.dumps(dict(token_counts))]}

    counts = pretoken_dataset.map(
        get_token_counts,
        remove_columns=pretoken_dataset.column_names,
        batch_size=len(pretoken_dataset),
        batched=True,  # this is faster without multiprocessing but cpu util is low - not clear why
    )

    all_token_counts = [
        Counter(pickle.loads(x))
        for x in tqdm(counts["token_counts"], desc="Aggregating token counts...")
    ]
    token_counts = Counter()

    for c in tqdm(all_token_counts, desc="Adding token counts..."):
        token_counts += c

    prior = np.ones(len(tokenizer))  # laplace smoothing
    for token, count in token_counts.items():
        prior[token] += count

    prior /= prior.sum()
    prior = np.log(prior)
    prior[tokenizer.all_special_ids] = 0.0  # convention

    if not return_pretoken_counts:
        return prior
    else:
        return prior, pretoken_counts


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dataset = load_dataset(
        "parquet", data_files={"train": args.dataset_name}, split="train"
    )

    substring_prior = compute_substring_prior(
        tokenizer, dataset, num_workers=args.num_workers
    )

    dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, args=args),
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=args.num_workers,
    )
    dataset.set_format("numpy")
    token_prior = get_prior(args.mode, dataset["input_ids"], tokenizer)

    np.save(
        Path(args.tokenizer_name) / "priors.npy",
        {
            "substring": substring_prior,
            "token": token_prior,
        },
    )
