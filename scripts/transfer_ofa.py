import sys
import os
from gensim.models import KeyedVectors
import transformers
from transformers import AutoTokenizer, HfArgumentParser
from dataclasses import dataclass
import numpy as np
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import torch

from zett.tokenizer_converters import convert_to_byte_level

OFA_PATH = "/mnt/nas_home/bm644/ofa"
OFA_EMBEDDING_PATH = "/mnt/nas_home/bm644/colexnet_vectors_minlang_50_200_10_updated.wv"

sys.path.insert(0, os.path.join(OFA_PATH, "ofa"))

import ofa
from utils import WordEmbedding


@dataclass
class Args:
    output: str
    tokenizer_name: str
    target_model: str = "FacebookAI/xlm-roberta-base"
    model_class: str = "AutoModelForMaskedLM"
    save_flax: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    loaded_n2v = KeyedVectors.load(OFA_EMBEDDING_PATH)
    multilingual_embeddings = WordEmbedding(loaded_n2v)

    source_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_tokenizer = convert_to_byte_level(
        AutoTokenizer.from_pretrained(args.tokenizer_name),
        match_special_tokens_to=source_tokenizer,
        make_whitespace_consistent=True,
    )[0]

    source_model = getattr(transformers, args.model_class).from_pretrained(
        args.target_model
    )

    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    assert len(source_tokenizer) == len(source_embeddings)

    print(f"Number of tokens in source tokenizer: {len(source_tokenizer)}")
    print(f"Number of tokens in target tokenizer: {len(target_tokenizer)}")

    tempdir = TemporaryDirectory()
    save_path = os.path.join(tempdir.name, "ofa")
    os.makedirs(save_path)

    ofa_args = SimpleNamespace(
        max_n_word_vectors=None,
        neighbors=10,
        temperature=0.1,
        source_language_set="None",
        target_language_set="None",
        keep_dim=f"[{source_embeddings.shape[1]}]",
        factorize=False,
        do_save=True,
        save_path=save_path,
        source_model_name="xlm-roberta-base",
    )

    if target_tokenizer.get_vocab() == source_tokenizer.get_vocab():
        target_embeddings = source_embeddings
    else:
        ofa.run_ofa(
            ofa_args,
            multilingual_embeddings=multilingual_embeddings,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_embeddings=source_embeddings,
        )
        target_embeddings = np.load(
            os.path.join(
                save_path + f"xlm_all_{source_embeddings.shape[1]}", "target_matrix.npy"
            )
        )
        target_embeddings = torch.from_numpy(target_embeddings)

    if not source_model.config.tie_word_embeddings:
        source_embeddings_out = source_model.get_output_embeddings().weight

        if target_tokenizer.get_vocab() == source_tokenizer.get_vocab():
            target_embeddings_out = source_embeddings_out
        else:
            ofa.run_ofa(
                ofa_args,
                multilingual_embeddings=multilingual_embeddings,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                source_embeddings=source_embeddings_out,
            )
            target_embeddings_out = np.load(
                os.path.join(
                    save_path + f"xlm_all_{source_embeddings.shape[1]}",
                    "target_matrix.npy",
                )
            )
            target_embeddings_out = torch.from_numpy(target_embeddings_out)
    else:
        target_embeddings_out = None

    source_model.resize_token_embeddings(len(target_tokenizer))
    source_model.get_input_embeddings().weight.data = target_embeddings

    if not source_model.config.tie_word_embeddings:
        source_model.get_output_embeddings().weight.data = target_embeddings_out

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
