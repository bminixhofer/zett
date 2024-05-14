from dataclasses import dataclass
from itertools import chain
from datasets import load_dataset
from tqdm import tqdm
import torch
from flax.training.common_utils import onehot, shard
from flax import jax_utils, serialization
import optax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import os
import math
import numpy as np
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
)
from functools import partial
from types import SimpleNamespace
from tokenizers import pre_tokenizers
import regex as re
from jax.experimental.multihost_utils import (
    host_local_array_to_global_array,
    global_array_to_host_local_array,
    process_allgather,
)
import copy

from zett.tokenizer_converters import convert_to_byte_level
from zett.model import (
    MODEL_PARALLEL_MAPS,
    IN_EMBEDDING_PATHS,
    OUT_EMBEDDING_PATHS,
)
from zett.utils import keystr, tokenize_function, load_params, get_batch_pspecs
from zett.collator import Collator


@dataclass
class Args:
    model_path: str
    tokenizer_name: str = None
    revision: str = None
    data_file: str = "datasets/valid/python.parquet"
    batch_size: int = 512
    block_size: int = 128
    preprocessing_num_workers: int = 64
    n_subsample: int = None
    data_mode: str = "chunk"
    sample_text_span: bool = False
    use_bias: bool = False
    add_bos: bool = False
    dtype: str = "bfloat16"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    if args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    tokenizer.eos_token_id = (
        tokenizer.pad_token_id
    ) = 0  # TODO: this is potentially problematic, should add to the tokenizer instead
    config = AutoConfig.from_pretrained(args.model_path)
    model = FlaxAutoModelForCausalLM.from_config(
        config=config,
        dtype=getattr(jnp, args.dtype),
        _do_init=False,
    )

    model_parallel_keys = MODEL_PARALLEL_MAPS.get(config.model_type, {})
    in_embedding_path = IN_EMBEDDING_PATHS[config.model_type]
    out_embedding_path = OUT_EMBEDDING_PATHS[config.model_type]

    def get_pspec(path, v):
        path_tuple = tuple(str(keystr(x)) for x in path)
        path = ".".join(path_tuple)

        for key, value in model_parallel_keys.items():
            if re.match(key, path):
                pspec = value
                print(f"Sharding {path} with {pspec}.")
                return P(*pspec)

        return P(*([None] * (np.array(v).ndim)))

    MESH = jax.sharding.Mesh(
        np.array(jax.local_devices()).reshape((1, -1)), ["data", "model"]
    )

    params = load_params(args.model_path, revision=args.revision)
    param_specs = jax.tree_util.tree_map_with_path(get_pspec, params)

    params = host_local_array_to_global_array(params, MESH, param_specs)

    dataset = load_dataset(
        "parquet",
        data_files={"train": args.data_file},
        split=f"train[:{args.n_subsample}]"
        if args.n_subsample is not None
        else "train",
    )

    if args.data_mode == "chunk":
        dataset = dataset.map(
            partial(tokenize_function, block_size=args.block_size, tokenizer=tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=args.preprocessing_num_workers,
        )
        dataset.set_format("numpy")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
    )

    if args.use_bias:
        bias = serialization.msgpack_restore(
            open(os.path.join(args.model_path, "bias.msgpack"), "rb").read()
        )[None, None]
    else:
        bias = None

    @jax.jit
    def step(params, batch):
        labels = batch.pop("labels", batch["input_ids"])

        logits = model(
            **{k: v for k, v in batch.items() if v is not None},
            params=params,
        )[0]
        if args.use_bias:
            logits = logits + bias

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        loss = optax.softmax_cross_entropy(
            shift_logits, onehot(shift_labels, shift_logits.shape[-1])
        )
        return loss

    losses = []
    chars_per_token = []
    bpcs = []

    for batch in tqdm(dataloader):
        batch = {
            k: (v.numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask", "offset_mapping"}
        }

        if args.add_bos:
            batch["input_ids"] = np.pad(
                batch["input_ids"][:, :-1],
                ((0, 0), (1, 0)),
                constant_values=tokenizer.bos_token_id,
            )
            if "attention_mask" in batch:
                batch["attention_mask"] = np.pad(
                    batch["attention_mask"][:, :-1], ((0, 0), (1, 0)), constant_values=1
                )
            if "offset_mapping" in batch:
                batch["offset_mapping"] = np.pad(
                    batch["offset_mapping"][:, :-1],
                    ((0, 0), (1, 0), (0, 0)),
                    constant_values=0,
                )

        offsets_mapping = batch.pop("offset_mapping", None)

        batch_specs = get_batch_pspecs(batch)
        batch = host_local_array_to_global_array(batch, MESH, batch_specs)

        loss = step(params, batch)
        loss = np.array(loss)

        special_tokens_mask = np.isin(batch["input_ids"], tokenizer.all_special_ids)
        loss *= (
            1 - special_tokens_mask[..., 1:]
        )  # do not count special tokens in loss and bpc

        losses.extend(np.array(loss.mean(-1)))

        if offsets_mapping is not None:
            cpt = offsets_mapping[:, 1:, 1] - offsets_mapping[:, 1:, 0]
            bpc = loss.sum(-1) / np.maximum(cpt.sum(-1), 1)
            chars_per_token.extend(cpt.mean(-1))
            bpcs.extend(bpc)

    losses = np.stack(losses)

    print("Avg. loss:", sum(losses) / len(losses))
    print(
        "Avg. chars per token:",
        sum(chars_per_token) / len(chars_per_token) if len(chars_per_token) > 0 else 0,
    )
    print("Avg. bpc:", sum(bpcs) / len(bpcs) if len(bpcs) > 0 else 0)
    print("Avg. perplexity:", math.exp(sum(losses) / len(losses)))
