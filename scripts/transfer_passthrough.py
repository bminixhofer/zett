from dataclasses import dataclass
import transformers
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from flax.serialization import msgpack_restore
from flax import traverse_util
import os
import numpy as np

from zett.model import IN_EMBEDDING_PATHS, OUT_EMBEDDING_PATHS
from zett.utils import load_params


@dataclass
class Args:
    output: str
    hypernet_checkpoint: str
    target_model: str = "gpt2"
    revision: str = None
    copy_inner_parameters_from: str = None
    model_class = "AutoModelForCausalLM"
    save_pt: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    downstream_config = AutoConfig.from_pretrained(args.target_model)
    downstream_model = getattr(transformers, "Flax" + args.model_class).from_config(
        config=downstream_config, _do_init=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    if args.copy_inner_parameters_from is not None:
        flat_downstream_params = traverse_util.flatten_dict(
            load_params(args.copy_inner_parameters_from)
        )
    else:
        flat_downstream_params = traverse_util.flatten_dict(
            load_params(args.target_model, revision=args.revision)
        )

    in_embedding_path = IN_EMBEDDING_PATHS[downstream_model.config.model_type]
    out_embedding_path = (
        OUT_EMBEDDING_PATHS[downstream_model.config.model_type]
        if not downstream_model.config.tie_word_embeddings
        else None
    )

    params = msgpack_restore(
        open(os.path.join(args.hypernet_checkpoint, "flax_model.msgpack"), "rb").read()
    )
    input_embeddings = params["input_embeddings"]["embedding"][: len(tokenizer)]
    if out_embedding_path is not None:
        output_embeddings = params["output_embeddings"]["embedding"][: len(tokenizer)]

    flat_downstream_params[in_embedding_path] = input_embeddings

    if out_embedding_path is not None:
        flat_downstream_params[out_embedding_path] = output_embeddings.T

    downstream_model.config.vocab_size = len(tokenizer)
    downstream_model.save_pretrained(
        args.output,
        params=traverse_util.unflatten_dict(flat_downstream_params),
        max_shard_size="20GB",  # needed for 7B PT conversion https://github.com/huggingface/transformers/issues/20248
    )
    tokenizer.save_pretrained(args.output)

    if args.save_pt:
        pt_model = getattr(transformers, args.model_class).from_pretrained(
            args.output, from_flax=True
        )
        pt_model.save_pretrained(args.output)
