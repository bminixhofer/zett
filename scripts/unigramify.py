from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerFast
from dataclasses import dataclass
from tokenizers import Tokenizer, models
import datasets
from datasets import load_dataset, load_from_disk
from collections import Counter
import numpy as np
from tqdm.auto import tqdm
import pyomo.environ as pyo
from pprint import pprint
from typing import List
import copy

from zett.compute_prior import compute_substring_prior
from zett.utils import CHARS_TO_BYTES
from zett.tokenizer_converters import convert_to_byte_level


def loss_fn(
    params,
    current_indices,
    current_counts,
    target_indices,
    target_counts,
    weights,
    margin=0.1,
    size_average=True,
):
    current_scores = (np.take(params, current_indices) * current_counts).sum(axis=1)
    current_scores = np.where(current_scores != 0, current_scores, -np.inf)
    target_scores = (np.take(params, target_indices) * target_counts).sum(axis=-1)

    loss = (
        np.maximum(0.0, margin - target_scores[:, None] + current_scores).sum(-1)
        * weights
    )
    if size_average:
        loss = loss.mean()

    return loss


@dataclass
class Args:
    train_dataset_names: List[str]
    valid_dataset_names: List[str]
    output: str
    tokenizer_name: str = "gpt2"
    keep_normalizer: bool = False
    keep_pretokenizer: bool = False
    to_byte_level: bool = False
    max_n_train_pretokens: int = 100_000
    num_workers: int = 96
    top_n_encodings: int = 16
    max_token_length: int = 16
    margin: float = 1e-3
    regularization_strength: float = 0.01
    norm: str = "l1"  # or "linf" or "none"


def get_initial_tokenizer(reference, datasets, normalize_fn=None, num_workers=None):
    vocab = reference.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

    tokenizer = Tokenizer(
        models.Unigram(
            [(inv_vocab[i], 0.0) for i in range(len(inv_vocab))],  # dummy tokenizer
            unk_id=reference.unk_token_id,
        )
    )
    if reference._tokenizer.normalizer is not None:
        tokenizer.normalizer = reference._tokenizer.normalizer

    if reference._tokenizer.pre_tokenizer is not None:
        tokenizer.pre_tokenizer = reference._tokenizer.pre_tokenizer

    tokenizer.decoder = reference._tokenizer.decoder

    transformers_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, clean_up_tokenization_spaces=False
    )
    transformers_tokenizer = convert_to_byte_level(
        transformers_tokenizer,
        match_special_tokens_to=reference,
        keep_normalizer=True,
        keep_pretokenizer=True,
    )[0]

    substring_prior, pretoken_counts = compute_substring_prior(
        transformers_tokenizer,
        datasets,
        num_workers=num_workers,
        return_pretoken_counts=True,
        normalize_fn=normalize_fn,
    )
    transformers_tokenizer._tokenizer.model.set_scores(substring_prior)
    return transformers_tokenizer, pretoken_counts


def counter_to_indices_counts(counts, max_token_length):
    keys, values = zip(*counts.most_common(max_token_length))

    indices = np.zeros((max_token_length), dtype=np.int32)
    indices[: len(keys)] = np.array(keys)

    counts = np.zeros((max_token_length), dtype=np.float32)
    counts[: len(keys)] = np.array(values)

    return indices, counts


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    print(args)

    original_reference = AutoTokenizer.from_pretrained(args.tokenizer_name)
    normalize_fn = (
        original_reference._tokenizer.normalizer.normalize_str
        if original_reference._tokenizer.normalizer is not None
        else None
    )
    reference, n_added = convert_to_byte_level(
        copy.deepcopy(original_reference),
        keep_normalizer=args.keep_normalizer,
        keep_pretokenizer=args.keep_pretokenizer,
    )

    print(f"Added {n_added} tokens to tokenizer.")

    valid_datasets = [
        load_dataset("parquet", data_files={"valid": name}, split="valid")
        for name in args.valid_dataset_names
    ]

    tokenizer, pretoken_counts = get_initial_tokenizer(
        reference,
        [load_from_disk(name) for name in args.train_dataset_names],
        num_workers=args.num_workers,
        normalize_fn=normalize_fn,
    )
    _, valid_pretoken_counts = compute_substring_prior(
        reference,
        valid_datasets,
        num_workers=args.num_workers,
        return_pretoken_counts=True,
        normalize_fn=normalize_fn,
    )

    pretoken_counts_to_use = pretoken_counts.most_common(args.max_n_train_pretokens)
    sum_of_counts = sum([v for k, v in pretoken_counts_to_use])

    scores = np.array(tokenizer._tokenizer.model.get_scores())
    original_scores = scores.copy()

    tokenizer._tokenizer.model.set_scores(original_scores)

    all_target_indices = []
    all_target_counts = []
    all_target_tokens = []
    all_weights = []

    for pretoken, c in pretoken_counts_to_use:
        target_tokens = tuple(
            [t.id for t in reference._tokenizer.model.tokenize(pretoken)]
        )
        all_weights.append(c)
        counter = Counter(target_tokens)
        target_indices, target_counts = counter_to_indices_counts(
            counter, args.max_token_length
        )

        all_target_indices.append(target_indices)
        all_target_counts.append(target_counts)
        all_target_tokens.append(target_tokens)

    all_target_indices = np.stack(all_target_indices)
    all_target_counts = np.stack(all_target_counts)
    all_weights = np.array(all_weights, dtype=np.float32)
    all_weights = all_weights / all_weights.mean()  # normalize to mean 1

    all_current_indices = np.zeros(
        (len(pretoken_counts_to_use), args.max_token_length, args.top_n_encodings),
        dtype=np.int32,
    )
    all_current_counts = np.zeros(
        (len(pretoken_counts_to_use), args.max_token_length, args.top_n_encodings),
        dtype=np.int32,
    )

    for i, (pretoken, count) in tqdm(
        enumerate(pretoken_counts_to_use), total=len(pretoken_counts_to_use)
    ):
        target_tokens = all_target_tokens[i]

        x = [
            counter_to_indices_counts(Counter(tokens), args.max_token_length)
            for tokens, _ in tokenizer._tokenizer.model.get_top_n_encodings(
                pretoken, args.top_n_encodings
            )
            if tuple(tokens) != target_tokens
        ]

        if len(x) > 0:
            current_indices, current_counts = list(zip(*x))
            all_current_indices[i, :, : len(current_indices)] = np.stack(
                current_indices, 1
            )
            all_current_counts[i, :, : len(current_counts)] = np.stack(
                current_counts, 1
            )

    model = pyo.ConcreteModel()
    model.scores = pyo.VarList(domain=pyo.Reals)
    for _ in range(len(original_scores)):
        model.scores.add()

    model.pretoken_slacks = pyo.VarList(domain=pyo.NonNegativeReals)

    if args.norm != "none":
        model.norm_slacks = pyo.VarList(domain=pyo.NonNegativeReals)
        model.norm_constraints = pyo.ConstraintList()

        if args.norm == "l1":
            for _ in range(len(original_scores)):
                model.norm_slacks.add()
        elif args.norm == "linf":
            model.norm_slacks.add()

    model.pretoken_constraints = pyo.ConstraintList()

    objective_expr = 0

    for i in tqdm(range(len(pretoken_counts_to_use))):
        tindices = np.where(all_target_counts[i, :] > 0)[0]

        base_expr = sum(
            -model.scores[1 + all_target_indices[i, idx]] * all_target_counts[i, idx]
            for idx in tindices
        )

        for j in range(args.top_n_encodings):
            cindices = np.where(all_current_counts[i, :, j] > 0)[0]

            if len(cindices) == 0:
                continue

            model.pretoken_slacks.add()
            expr = (
                base_expr
                + sum(
                    model.scores[1 + all_current_indices[i, idx, j]]
                    * all_current_counts[i, idx, j]
                    for idx in cindices
                )
                - model.pretoken_slacks[len(model.pretoken_slacks)]
            )

            model.pretoken_constraints.add(expr=expr <= -args.margin)

            objective_expr += (
                model.pretoken_slacks[len(model.pretoken_slacks)] * all_weights[i]
            )

    # l1/inf norm of diff to original scores
    if args.norm != "none":
        for i in range(len(original_scores)):
            if args.norm == "l1":
                norm_index = 1 + i
            else:
                norm_index = 1

            model.norm_constraints.add(
                expr=model.scores[1 + i] - model.norm_slacks[norm_index]
                <= original_scores[i]
            )
            model.norm_constraints.add(
                expr=-model.scores[1 + i] - model.norm_slacks[norm_index]
                <= -original_scores[i]
            )
            objective_expr += (
                model.norm_slacks[norm_index] * args.regularization_strength
            )

    model.objective = pyo.Objective(expr=objective_expr, sense=pyo.minimize)

    solver = pyo.SolverFactory("cplex")
    solver.options["lpmethod"] = 4  # barrier solver to support parallelism
    solver.options["threads"] = args.num_workers

    solution = solver.solve(model)

    scores = np.array([model.scores[i + 1].value for i in range(len(original_scores))])
    scores[reference.all_special_ids] = original_scores[reference.all_special_ids]

    tokenizer._tokenizer.model.set_scores(scores)

    if not args.to_byte_level:
        tokenizer._tokenizer.model.set_pieces(
            tokenizer._tokenizer.model.get_pieces()[: len(original_reference)]
        )  # remove tokens introduced from byte-level conversion

    tokenizer.save_pretrained(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.output)

    losses = loss_fn(
        scores,
        all_current_indices,
        all_current_counts,
        all_target_indices,
        all_target_counts,
        all_weights,
        margin=0.0,
        size_average=False,
    )

    print(f"n wrong tokens: {(losses > 0).sum()} ({(losses > 0).mean():.1%})")
    print("Top 50 wrong tokens: ")
    pprint(
        [
            (
                i,
                pretoken_counts_to_use[i],
                " ".join(
                    x.value
                    for x in reference._tokenizer.model.tokenize(
                        pretoken_counts_to_use[i][0]
                    )
                ),
                " ".join(
                    x.value
                    for x in tokenizer._tokenizer.model.tokenize(
                        pretoken_counts_to_use[i][0]
                    )
                ),
            )
            for i in np.where(losses > 0)[0][:50]
        ]
    )

    n_correct = 0
    n_original_correct = 0
    n_total = 0

    for x, v in tqdm(valid_pretoken_counts.most_common()):
        decoded = bytes(CHARS_TO_BYTES[b] for b in x).decode("utf-8").strip()

        original_reference_tokens = original_reference.encode(
            decoded, add_special_tokens=False
        )
        if len(original_reference_tokens) == 0:
            continue

        reference_tokens = reference.encode(decoded, add_special_tokens=False)
        new_tokens = tokenizer.encode(decoded, add_special_tokens=False)

        if reference_tokens == original_reference_tokens:
            n_original_correct += v

        if new_tokens == original_reference_tokens:
            n_correct += v

        n_total += v

    avg_logp_diff = np.abs(scores - original_scores).mean()

    print(args)
    print(f"Original accuracy: {n_original_correct / n_total:.4%}")
    print(f"Unigramified accuracy: {n_correct / n_total:.4%}")
    print(f"Avg. logp diff: {avg_logp_diff:.4f}")
