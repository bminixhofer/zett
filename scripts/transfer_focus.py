import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from deepfocus import FOCUS
from dataclasses import dataclass
import numpy as np
from flax import traverse_util
import torch

from zett.model import IN_EMBEDDING_PATHS, OUT_EMBEDDING_PATHS, BIAS_PATHS
from zett.utils import load_params
from zett.tokenizer_converters import convert_to_byte_level


@dataclass
class Args:
    output: str
    tokenizer_name: str
    target_model: str = "xlm-roberta-base"
    revision: str = None
    model_class: str = "AutoModelForMaskedLM"
    lang_code: str = "en"
    save_pt: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    source_tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    downstream_config = AutoConfig.from_pretrained(args.target_model)
    downstream_model = getattr(transformers, "Flax" + args.model_class).from_config(
        config=downstream_config, _do_init=False
    )

    embedding_path_in = IN_EMBEDDING_PATHS[downstream_model.config.model_type]
    embedding_path_out = (
        OUT_EMBEDDING_PATHS[downstream_model.config.model_type]
        if not downstream_model.config.tie_word_embeddings
        else None
    )
    bias_path = BIAS_PATHS[downstream_model.config.model_type]

    flat_downstream_params = traverse_util.flatten_dict(
        load_params(args.target_model, revision=args.revision)
    )

    source_embeddings_in = flat_downstream_params[embedding_path_in].astype(np.float32)

    if embedding_path_out is not None:
        source_embeddings_out = flat_downstream_params[embedding_path_out].T.astype(
            np.float32
        )
    else:
        source_embeddings_out = None

    target_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    target_tokenizer = convert_to_byte_level(
        target_tokenizer,
        match_special_tokens_to=source_tokenizer,
        make_whitespace_consistent=True,
    )[0]

    if target_tokenizer.get_vocab() == source_tokenizer.get_vocab():
        target_embeddings = source_embeddings_in
    else:
        target_embeddings, sources = FOCUS(
            source_embeddings=torch.from_numpy(source_embeddings_in),
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            auxiliary_embedding_mode="fasttext-wordlevel",
            language_identifier=args.lang_code,
        )
        target_embeddings = target_embeddings.numpy()

    if embedding_path_out is not None:
        if target_tokenizer.get_vocab() == source_tokenizer.get_vocab():
            target_embeddings_out = source_embeddings_out
        else:
            target_embeddings_out, out_sources = FOCUS(
                source_embeddings=torch.from_numpy(source_embeddings_out),
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                auxiliary_embedding_mode="fasttext-wordlevel",
                language_identifier=args.lang_code,
            )
            target_embeddings_out = target_embeddings_out.numpy()
    else:
        target_embeddings_out = None

    not_found_tokens = list(set(target_tokenizer.get_vocab().values()) - set(sources))
    print(f"Setting {len(not_found_tokens)} tokens to UNK token id.")
    target_embeddings[not_found_tokens] = source_embeddings_in[
        source_tokenizer.unk_token_id
    ][None]

    downstream_model.config.vocab_size = len(target_tokenizer)
    flat_downstream_params[embedding_path_in] = target_embeddings

    if embedding_path_out is not None:
        target_embeddings_out[not_found_tokens] = source_embeddings_out[
            source_tokenizer.unk_token_id
        ][None]
        flat_downstream_params[embedding_path_out] = target_embeddings_out.T

    if bias_path is not None:
        flat_downstream_params[bias_path] = np.zeros(
            (len(target_tokenizer),)
        )  # zero out bias

    downstream_model.save_pretrained(
        args.output,
        params=traverse_util.unflatten_dict(flat_downstream_params),
        max_shard_size="20GB",  # needed for 7B PT conversion https://github.com/huggingface/transformers/issues/20248
    )
    source_tokenizer.save_pretrained(
        args.output
    )  # to get tokenizer_config.json and other metadata
    target_tokenizer.save_pretrained(args.output)

    if args.save_pt:
        pt_model = getattr(transformers, args.model_class).from_pretrained(
            args.output, from_flax=True
        )
        pt_model.save_pretrained(args.output)
