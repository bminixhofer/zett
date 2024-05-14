import transformers
from transformers import AutoTokenizer, HfArgumentParser
from dataclasses import dataclass
from datasets import load_dataset
import numpy as np
import torch
from tqdm import tqdm

from zett.tokenizer_converters import convert_to_byte_level
from zett.utils import copy_tokenizer_auxiliaries, make_whitespace_consistent


@dataclass
class Args:
    output: str
    tokenizer_name: str
    model_name_or_path: str = "FacebookAI/xlm-roberta-base"
    model_class: str = "AutoModelForMaskedLM"
    fvt_mode: str = "no"  # "fvt", "bfvt"
    fallback_mode: str = "unk"  # "random", "unk"
    save_flax: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    source_tokenizer = convert_to_byte_level(
        AutoTokenizer.from_pretrained(args.model_name_or_path)
    )[0]
    target_tokenizer = convert_to_byte_level(
        AutoTokenizer.from_pretrained(args.tokenizer_name),
        match_special_tokens_to=source_tokenizer,
        make_whitespace_consistent=True,
    )[0]

    source_model = getattr(transformers, args.model_class).from_pretrained(
        args.model_name_or_path
    )
    source_embeddings_in = source_model.get_input_embeddings().weight.data

    if not source_model.config.tie_word_embeddings:
        source_embeddings_out = source_model.get_output_embeddings().weight.data

        source_embeddings = torch.cat(
            [source_embeddings_in, source_embeddings_out], dim=1
        )
    else:
        source_embeddings = source_embeddings_in

    source_vocab = source_tokenizer.get_vocab()

    if args.fallback_mode == "random":
        target_embeddings = np.random.normal(
            loc=source_embeddings.mean(0),
            scale=source_embeddings.std(0),
            size=(len(target_tokenizer), source_embeddings.shape[1]),
        )
    else:
        target_embeddings = (
            source_embeddings[[source_tokenizer.unk_token_id]]
            .repeat(len(target_tokenizer), 1)
            .numpy()
        )

    overlap_indices = []

    for i in tqdm(range(len(target_tokenizer)), "Computing embeddings.."):
        token = target_tokenizer.convert_ids_to_tokens(i)
        idx = source_vocab.get(token)

        found = False

        if idx is not None and idx < len(source_embeddings):
            overlap_indices.append(i)
            target_embeddings[i] = source_embeddings[idx]
        elif args.fvt_mode != "no":
            decomposed = source_tokenizer._tokenizer.model.tokenize(token)
            if args.fvt_mode == "fvt" and not any(
                x.id >= len(source_embeddings) for x in decomposed
            ):
                constituent_idx = np.array([x.id for x in decomposed])
                if len(constituent_idx) > 0:
                    overlap_indices.append(i)
                    target_embeddings[i] = source_embeddings[constituent_idx].mean(0)
            elif args.fvt_mode == "bfvt":
                constituent_idx = np.array(
                    [x.id for x in decomposed if x.id < len(source_embeddings)]
                )
                if len(constituent_idx) > 0:
                    overlap_indices.append(i)
                    target_embeddings[i] = source_embeddings[constituent_idx].mean(0)

    print(f"Overlapping tokens: {len(overlap_indices)}/{len(target_tokenizer)}")
    source_model.resize_token_embeddings(len(target_tokenizer))
    source_model.config.vocab_size = len(target_tokenizer)

    source_model.get_input_embeddings().weight.data[:] = torch.from_numpy(
        target_embeddings[:, : source_embeddings_in.shape[1]]
    )

    if not source_model.config.tie_word_embeddings:
        source_model.get_output_embeddings().weight.data[:] = torch.from_numpy(
            target_embeddings[:, source_embeddings_in.shape[1] :]
        )

    source_model.save_pretrained(args.output)
    source_tokenizer.save_pretrained(
        args.output
    )  # to get tokenizer_config.json and other metadata
    target_tokenizer.save_pretrained(args.output)

    if args.save_flax:
        flax_model = getattr(transformers, "Flax" + args.model_class).from_pretrained(
            args.output, from_pt=True
        )
        flax_model.save_pretrained(args.output)
