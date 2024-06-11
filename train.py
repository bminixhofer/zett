import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from pprint import pprint
from functools import partial
import regex as re
import copy

import flax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.core.frozen_dict import unfreeze
from flax.training.common_utils import stack_forest, onehot
from flax import serialization
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from jax.experimental.multihost_utils import (
    host_local_array_to_global_array,
    global_array_to_host_local_array,
    process_allgather,
)
from jax.sharding import PartitionSpec as P, NamedSharding
import random

from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoTokenizer,
    FlaxAutoModel,
    AutoConfig,
    HfArgumentParser,
    set_seed,
    DataCollatorForLanguageModeling,
    FlaxAutoModelForCausalLM,
    FlaxAutoModelForMaskedLM,
)
from datasets import load_dataset

# jax.distributed.initialize()

from zett.model import (
    MODEL_PARALLEL_MAPS,
    Hypernet,
    PassthroughHypernet,
    IN_EMBEDDING_PATHS,
    OUT_EMBEDDING_PATHS,
    BIAS_PATHS,
    HypernetArgs,
    HYPERNET_MODEL_PARALLEL_MAP,
)
from zett.utils import (
    huber_loss,
    pad_to_multiple_of,
    load_params,
    create_learning_rate_fn,
    MADLAD_METADATA,
    NEGATIVE_INF_FILL_VALUE,
    SHARDING,
    get_sample_indices,
    EPSILON,
    get_surface_form_matrix,
    keystr,
    get_batch_pspecs,
    to_global_batch,
)
from zett.collator import Collator
from zett.dataset import TrainDataset, ValidDataset
from zett.tokenizer_converters import convert_to_byte_level

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
np.seterr(all="raise")


@dataclass
class TrainingArguments:
    output_dir: str
    init_from_params: str = None
    backbone_training: str = "no"  # or "full"
    resume_from_checkpoint: str = None
    resume_from_checkpoint_reset_steps: bool = False
    save_state: bool = True
    loss: str = "clm"
    mix_languages: bool = False
    use_adafactor: bool = False
    train_batch_size: int = 256
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    learning_rate_alpha: float = 0.1
    random_learning_rate: float = None
    max_grad_norm: float = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    steps: int = 100_000
    random_warmup_steps: int = 0
    identity_steps: int = 0
    warmup_steps: int = 10000
    logging_steps: int = 500
    save_steps: int = 10000
    eval_steps: int = 10000
    eval_at_step_zero: bool = False
    do_train: bool = True
    seed: int = 42
    gradient_accumulation_steps: int = 1
    debug: bool = False
    run_backbone_in_training_mode: bool = False
    overwrite_special_token_embeddings: bool = True
    learnable_bias: bool = False
    lexical_loss_weight: float = 0.0
    lexical_loss_kind: str = "mse"
    apply_lexical_loss_to_init: bool = False
    add_target_priors_to_bias: bool = True
    reinit_projectors: bool = False
    do_cost_analysis: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = None
    tokenizer_name: Optional[str] = None
    revision: Optional[str] = None
    config_name: Optional[str] = None
    dtype: Optional[str] = "float32"


@dataclass
class DataArguments:
    train_directory: str
    valid_directory: str
    langs: str
    use_passthrough_hypernet: bool = False
    do_sequence_packing: bool = True
    add_prefix_space: bool = True
    add_eos: bool = True
    n_pools: int = 1
    language_sampling_alpha: float = 0.3
    block_size: int = 128
    extra_valid_tokenizer_names: List[str] = None
    extra_valid_files: List[str] = None
    extra_lang_codes: List[str] = None
    n_valid_subsample: int = None
    target_tokenizer_name: str = None
    pad_to_multiple_of: int = 128
    do_tokenizer_sampling: bool = True
    tokenizer_sample_reweigh_temperature: float = np.inf
    tokenizer_batch_size: int = 512
    tokenizer_sample_min: int = 16384
    tokenizer_sample_max: int = 32768
    tokenizer_sample_mean: float = 32768.0
    tokenizer_sample_std: float = 0.0
    tokenizer_noise_std: float = 0.0
    tokenizer_noise_mean: float = 0.0
    dataloader_num_workers: int = 64
    identity_n_subsample: int = None
    n_token_subsample: int = 8192
    sample_text_span: bool = True
    subsample_mode: str = "random"  # or "positives_only"


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray
    source_embeddings: jnp.ndarray

def prepare_batch(batch, metrics=None):
    batch_metrics = batch.pop("metrics", None)
    lang_code = batch.pop("lang_code")

    if metrics is not None and batch_metrics is not None:
        if f"{lang_code}_avg_byte_length" in metrics:
            metrics[f"{lang_code}_avg_byte_length"].append(
                batch_metrics["avg_byte_length"]
            )
        if f"{lang_code}_unk_ratio" in metrics:
            metrics[f"{lang_code}_unk_ratio"].append(batch_metrics["unk_ratio"])
        if f"{lang_code}_pad_ratio" in metrics:
            metrics[f"{lang_code}_pad_ratio"].append(
                (batch["attention_mask"] == 0).mean()
            )

    return batch


def main():
    global SHARDING

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, HypernetArgs)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, hn_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
        name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    else:
        (
            model_args,
            data_args,
            training_args,
            hn_args,
        ) = parser.parse_args_into_dataclasses()
        name = None

    print("Starting run..")

    if training_args.debug:
        # debug settings
        data_args.train_directory = data_args.valid_directory
        data_args.dataloader_num_workers = 0  # main process
        data_args.n_token_subsample //= 8
        data_args.tokenizer_batch_size //= 8

        training_args.train_batch_size //= 16
        training_args.eval_batch_size //= 16
        training_args.eval_steps = 3
        training_args.logging_steps = 2

        all_local_devices = jax.local_devices()
        jax.device_count = lambda: 1
        jax.local_device_count = lambda: 1
        jax.local_devices = lambda: all_local_devices[:1]
        jax.devices = lambda: all_local_devices[:1]

        SHARDING = PositionalSharding(
            mesh_utils.create_device_mesh((1,), devices=all_local_devices[:1])
        )

        os.environ["WANDB_MODE"] = "disabled"

    MESH = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1), ["model"])

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    data_args.hn_surface_maxlen = hn_args.hn_surface_maxlen
    lang_probs = None

    if data_args.langs.endswith(".txt"):
        lang_data = [x.strip() for x in open(data_args.langs).readlines()]
        if "," in lang_data[0]:
            langs = [x.split(",")[0].strip() for x in lang_data]
            lang_probs = np.array([float(x.split(",")[1].strip()) for x in lang_data])
            lang_probs = lang_probs / lang_probs.sum()
        else:
            langs = lang_data
    else:
        langs = data_args.langs.split(" ")

    if len(langs) == 1:
        lang_probs = np.array([1.0])
    elif lang_probs is None:
        language_counts = MADLAD_METADATA.loc[langs]["n_pages"].values
        lang_probs = (
            language_counts / language_counts.sum()
        ) ** data_args.language_sampling_alpha

    data_args.langs = langs
    if training_args.mix_languages:
        assert len(data_args.langs) > 1
        data_args.langs.insert(0, "all")

    hn_args.n_langs = len(data_args.langs)

    set_seed(training_args.seed)

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
        )
    else:
        raise NotImplementedError()

    reference = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path
    )
    # gpt2 / llama
    if reference.pad_token is None:
        reference.pad_token = reference.eos_token

    # make sure pad token is set, we need it for the hypernet mask
    config.pad_token_id = reference.pad_token_id

    # add hn args
    for k, v in asdict(hn_args).items():
        setattr(config, k, v)

    # add arg for traceability
    config.name = name

    model_class = {
        "clm": FlaxAutoModelForCausalLM,
        "mlm": FlaxAutoModelForMaskedLM,
    }[training_args.loss]

    model = model_class.from_config(
        config=config,
        dtype=getattr(jnp, model_args.dtype),
        _do_init=False,
    )
    model.config.original_vocab_size = model.config.vocab_size

    model_params = load_params(
        model_args.model_name_or_path, revision=model_args.revision
    )

    in_embedding_path = IN_EMBEDDING_PATHS[config.model_type]
    out_embedding_path = (
        OUT_EMBEDDING_PATHS[config.model_type]
        if not config.tie_word_embeddings
        else None
    )
    bias_path = BIAS_PATHS[config.model_type]

    flat_model_params = traverse_util.flatten_dict(model_params)

    source_embeddings_in = pad_to_multiple_of(
        flat_model_params.pop(in_embedding_path).astype(np.float32),
        data_args.pad_to_multiple_of,
    )
    if out_embedding_path is not None:
        source_embeddings_out = pad_to_multiple_of(
            flat_model_params.pop(out_embedding_path).astype(np.float32).T,
            data_args.pad_to_multiple_of,
        )
        source_embeddings = np.concatenate([source_embeddings_in, source_embeddings_out], axis=1)
    else:
        source_embeddings = source_embeddings_in

    model_params = traverse_util.unflatten_dict(flat_model_params)

    config.separate_out_embeddings = out_embedding_path is not None

    if hn_args.hn_embed_using_source_embeddings:
        hn_tokenizer = copy.deepcopy(reference)
    elif hn_args.hn_model_name_or_path is not None:
        hn_tokenizer = AutoTokenizer.from_pretrained(hn_args.hn_model_name_or_path)
    else:
        hn_tokenizer = None

    if hn_tokenizer is not None:
        hn_tokenizer, hn_n_extra_tokens = convert_to_byte_level(hn_tokenizer)
        config.hn_n_extra_tokens = hn_n_extra_tokens

    if training_args.init_from_params is not None:
        flat_pretrained_params = traverse_util.flatten_dict(
            serialization.msgpack_restore(
                open(
                    os.path.join(training_args.init_from_params, "flax_model.msgpack"),
                    "rb",
                ).read()
            )
        )

        full_weights_path = os.path.join(training_args.init_from_params, "full_model")
        if os.path.exists(full_weights_path):
            model_params = load_params(full_weights_path)
    else:
        flat_pretrained_params = {}

    # data
    train_batch_size = training_args.train_batch_size
    eval_batch_size = training_args.eval_batch_size

    if training_args.loss == "mlm":
        inner_collator = DataCollatorForLanguageModeling
    else:
        inner_collator = None

    if training_args.mix_languages:
        train_datasets = [
            TrainDataset(
                # skip "all" for loading data
                data_args.langs[1:],
                data_args.train_directory,
                lang_probs,
                train_batch_size,
                block_size=data_args.block_size,
                do_sequence_packing=data_args.do_sequence_packing,
            )
        ]

        initial_texts = train_datasets[0].get_texts(data_args.tokenizer_batch_size)
        valid_initial_texts = train_datasets[0].get_texts_in_each_language(
            data_args.tokenizer_batch_size
        )

        train_collators = [
            Collator(
                reference,
                hn_tokenizer,
                data_args,
                batch_size=train_batch_size,
                tokenizer_name=data_args.target_tokenizer_name,
                initial_texts={"all": initial_texts},
                with_consistent_whitespace=not data_args.use_passthrough_hypernet,  # TODO: probably hacky
            )
        ]

        train_probs = np.array([1.0])
    else:
        train_datasets = [
            TrainDataset(
                [lang],
                data_args.train_directory,
                np.array([1.0]),
                train_batch_size,
                block_size=data_args.block_size,
                do_sequence_packing=data_args.do_sequence_packing,
            )
            for lang in data_args.langs
        ]

        initial_texts = {
            lang_code: dset.get_texts(data_args.tokenizer_batch_size)
            for lang_code, dset in zip(data_args.langs, train_datasets)
        }
        valid_initial_texts = initial_texts

        train_collators = [
            Collator(
                reference,
                hn_tokenizer,
                data_args,
                batch_size=train_batch_size,
                tokenizer_name=data_args.target_tokenizer_name,
                initial_texts={lang_code: initial_texts[lang_code]},
                inner_collator=inner_collator,
                with_consistent_whitespace=not data_args.use_passthrough_hypernet,  # TODO: probably hacky
            )
            for lang_code in data_args.langs
        ]

        train_probs = lang_probs

    num_workers_per_dataloader = data_args.dataloader_num_workers // len(train_datasets)
    train_dataloaders = [
        DataLoader(
            dset,
            batch_size=1,  # batched internally
            num_workers=num_workers_per_dataloader,
            collate_fn=collator,
            persistent_workers=num_workers_per_dataloader > 0,
        )
        for dset, collator in zip(train_datasets, train_collators)
    ]

    valid_dataset = ValidDataset(
        [lang for lang in data_args.langs if lang != "all"],
        data_args.valid_directory,
        data_args.n_valid_subsample,
        eval_batch_size,
        block_size=data_args.block_size,
    )
    valid_collator = Collator(
        reference,
        hn_tokenizer,
        data_args,
        batch_size=train_batch_size,
        tokenizer_name=data_args.target_tokenizer_name,
        initial_texts=valid_initial_texts,
        inner_collator=inner_collator,
        is_validation=True,
        with_consistent_whitespace=not data_args.use_passthrough_hypernet,  # TODO: probably hacky
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,  # batched internally
        num_workers=data_args.dataloader_num_workers,
        shuffle=False,
        collate_fn=valid_collator,
        persistent_workers=data_args.dataloader_num_workers > 0,
    )

    if training_args.identity_steps > 0:
        identity_data_args = copy.deepcopy(data_args)
        identity_data_args.do_tokenizer_sampling = False

        if hn_args.hn_embed_using_source_embeddings:
            identity_data_args.hn_surface_maxlen = 1

        if data_args.identity_n_subsample is not None:
            identity_data_args.n_token_subsample = data_args.identity_n_subsample

        identity_collator = Collator(
            reference,
            hn_tokenizer,
            identity_data_args,
            tokenizer_name=reference.name_or_path,
            lang_code=data_args.langs[0],  # dummy
            inner_collator=inner_collator,
            with_consistent_whitespace=False,
        )  # would change vocab

        identity_train_dataloader = DataLoader(
            TensorDataset(torch.zeros(training_args.identity_steps)),  # dummy data
            batch_size=1,  # batched internally
            collate_fn=partial(identity_collator, for_identity_step=True),
        )
    else:
        identity_train_dataloader = None

    extra_valid_dataloaders = []
    for i, valid_tokenizer_name in enumerate(
        data_args.extra_valid_tokenizer_names or []
    ):
        valid_data_args = copy.deepcopy(data_args)
        valid_data_args.do_tokenizer_sampling = False
        valid_data_args.n_token_subsample = None
        valid_data_args.sample_text_span = False

        dset = load_dataset(
            "parquet",
            data_files={"train": data_args.extra_valid_files[i]},
            split=f"train[:{data_args.n_valid_subsample}]"
            if data_args.n_valid_subsample is not None
            else "train",
        )
        extra_valid_dataloaders.append(
            DataLoader(
                dset,
                batch_size=training_args.eval_batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=Collator(
                    reference,
                    hn_tokenizer,
                    valid_data_args,
                    tokenizer_name=valid_tokenizer_name,
                    lang_code=data_args.extra_lang_codes[i],
                    inner_collator=inner_collator,
                    is_validation=True,
                    with_consistent_whitespace=not data_args.use_passthrough_hypernet,  # TODO: probably hacky
                ),
            )
        )

    if data_args.use_passthrough_hypernet:
        hypernet = PassthroughHypernet(
            config,
            vocab_size=len(source_embeddings_in),
            dtype=getattr(jnp, model_args.dtype),
        )

        n_anchors = None
        example_surface_forms = None
        example_priors = None
    else:
        hypernet = Hypernet(config, dtype=getattr(jnp, model_args.dtype))

        n_anchors = 2048  # some reasonably small number, should probably be sampled, but does not make a difference realistically
        example_surface_forms = get_surface_form_matrix(
            convert_to_byte_level(copy.deepcopy(reference))[0],
            hn_args.hn_surface_maxlen,
            hn_tokenizer,
        )[0][:n_anchors]
        example_priors = np.zeros(n_anchors, dtype=np.float32)

    hypernet_init_params = (
        jax.random.PRNGKey(training_args.seed),
        jnp.ones((1, hn_args.hn_surface_maxlen), dtype=jnp.int32),
        jnp.ones(1, dtype=jnp.float32),
        source_embeddings[:2],
        jnp.zeros((), dtype=jnp.int32),  # lang index
    )

    rng = jax.random.PRNGKey(training_args.seed + jax.process_index())
    rng, dropout_rng = jax.random.split(rng)

    random_learning_rate_fn, _ = create_learning_rate_fn(training_args)

    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)

        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params)
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    def get_labels(params):
        flat_params = traverse_util.flatten_dict(params)

        def label(path):
            if len(path) >= 2 and path[-2] in {"scaler", "in_scaler"}:
                return "freeze"

            if path[0] == "hypernet" or (
                path[0] == "inner" and training_args.backbone_training == "full"
            ):
                return "train"

            return "freeze"

        flat_mask = {path: label(path) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    labels = get_labels(
        {
            "inner": model_params,
            "hypernet": jax.eval_shape(hypernet.init, *hypernet_init_params)["params"],
        }
    )

    if training_args.use_adafactor:
        inner_optimizer = optax.adafactor(
            learning_rate=random_learning_rate_fn,
            weight_decay_rate=training_args.weight_decay,
            weight_decay_mask=decay_mask_fn,
        )
    else:
        inner_optimizer = optax.adamw(
            learning_rate=random_learning_rate_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    transforms = [
        optax.multi_transform(
            {"train": inner_optimizer, "freeze": optax.set_to_zero()}, labels
        ),
    ]

    if training_args.max_grad_norm is not None:
        transforms.insert(0, optax.clip_by_global_norm(training_args.max_grad_norm))

    optimizer = optax.chain(*transforms)
    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=training_args.gradient_accumulation_steps
        )

    model_fn = model.__call__
    hypernet_fn = hypernet.apply

    def init_state(source_embeddings, model_params, flat_pretrained_params):
        source_embeddings_in = source_embeddings[:, : hn_args.n_embd]
        source_embeddings_out = (
            source_embeddings[:, hn_args.n_embd :]
            if out_embedding_path is not None
            else None
        )

        if data_args.use_passthrough_hypernet:
            flat_hypernet_params = {
                ("input_embeddings", "embedding"): source_embeddings_in,
            }

            if out_embedding_path is not None:
                flat_hypernet_params[
                    ("output_embeddings", "embedding")
                ] = source_embeddings_out

            hypernet_params = traverse_util.unflatten_dict(flat_hypernet_params)

            # no need
            source_embeddings = None
        else:
            hypernet_params = hypernet.init(*hypernet_init_params)["params"]

            # rescale input and output embedding matrices (if enabled)
            if hn_args.hn_rescale_embeddings:
                hypernet_params = hypernet.apply(
                    {"params": hypernet_params},
                    example_surface_forms,
                    example_priors,
                    source_embeddings,
                    jnp.zeros((), dtype=jnp.int32),  # lang index
                    source_embeddings_in,
                    source_embeddings_out,
                    method=hypernet.init_rescaler,
                )["params"]

        if len(flat_pretrained_params) > 0:
            if training_args.reinit_projectors:
                removed_params = []

                for k in list(flat_pretrained_params.keys()):
                    if k[0] in {
                        "fallback_embeddings",
                        "input_projection",
                        "output_projection",
                        "bias_projection",
                        "scaler",
                        "in_scaler",
                    }:
                        removed_params.append(k)
                        del flat_pretrained_params[k]

                print("Reinitialized params:")
                pprint(removed_params)

            flat_hypernet_params = traverse_util.flatten_dict(hypernet_params)
            flat_hypernet_params.update(flat_pretrained_params)
            hypernet_params = traverse_util.unflatten_dict(flat_hypernet_params)

        if training_args.backbone_training == "no":
            # store params in bf16 to save memory
            model_params = jax.tree_map(
                lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                model_params,
            )

        params = {
            "hypernet": hypernet_params,
            "inner": model_params,
        }

        state = TrainState.create(
            apply_fn=hypernet_fn,
            params=params,
            source_embeddings=source_embeddings,
            tx=optimizer,
            dropout_rng=dropout_rng,
        )

        return state

    state_shape = jax.eval_shape(
        init_state, source_embeddings, model_params, flat_pretrained_params
    )

    # shard tree
    model_parallel_keys = HYPERNET_MODEL_PARALLEL_MAP
    model_parallel_keys.update(MODEL_PARALLEL_MAPS.get(config.model_type, {}))

    def get_pspec(path, v):
        path_tuple = tuple(str(keystr(x)) for x in path)
        path = ".".join(path_tuple)

        for key, value in model_parallel_keys.items():
            if re.match(key, path):
                pspec = value
                print(f"Sharding {path} ({v}) with {pspec}.")
                return P(*pspec)

        return P()

    state_pspecs = jax.tree_util.tree_map_with_path(get_pspec, state_shape)
    state_shardings = jax.tree_map(lambda x: NamedSharding(MESH, x), state_pspecs)

    if training_args.resume_from_checkpoint:
        local_state = serialization.from_bytes(
            state_shape,
            open(
                os.path.join(training_args.resume_from_checkpoint, "state.msgpack"),
                "rb",
            ).read(),
        )
        if training_args.resume_from_checkpoint_reset_steps:
            resume_step = 0
            local_state = local_state.replace(
                step=0, tx=optimizer, apply_fn=hypernet.apply, dropout_rng=dropout_rng
            )
        else:
            resume_step = local_state.step
        state = host_local_array_to_global_array(local_state, MESH, state_pspecs)
    else:
        flat_pretrained_shardings = {
            k: v
            for k, v in traverse_util.flatten_dict(
                state_shardings.params["hypernet"]
            ).items()
            if k in flat_pretrained_params
        }
        in_shardings = (
            state_shardings.source_embeddings,
            state_shardings.params["inner"],
            flat_pretrained_shardings,
        )
        init_args = (source_embeddings, model_params, flat_pretrained_params)

        def make_global_array(x, sharding):
            if x is None:
                return None

            if sharding is None:
                return x

            def cb(index):
                return x[index]

            return jax.make_array_from_callback(x.shape, sharding, cb)

        init_args = jax.tree_map(make_global_array, init_args, in_shardings)

        state = jax.jit(
            init_state, in_shardings=in_shardings, out_shardings=state_shardings
        )(*init_args)
        resume_step = 0

    if training_args.apply_lexical_loss_to_init:
        assert not data_args.do_tokenizer_sampling

        initial_input_embeddings, initial_output_embeddings, _ = jax.jit(
            state.apply_fn
        )(
            {"params": state.params["hypernet"]},
            target_surface_forms=train_collators[0].surface_forms,
            target_priors=train_collators[0].scores,
            source_embeddings=state.source_embeddings,
            lang_index=None,  # not implemented
        )

    if training_args.do_cost_analysis:
        compiled_hypernet_fn = (
            jax.jit(hypernet_fn)
            .lower(
                {
                    "params": state.params["hypernet"],
                },
                example_surface_forms,
                example_priors,
                source_embeddings,
                jnp.zeros((), dtype=jnp.int32),
            )
            .compile()
        )

        flat_model_params = traverse_util.flatten_dict(model_params)
        flat_model_params[in_embedding_path] = source_embeddings_in[
            : model.config.vocab_size
        ]
        if out_embedding_path is not None:
            flat_model_params[out_embedding_path] = source_embeddings_out[
                : model.config.vocab_size
            ].T
        model_params = traverse_util.unflatten_dict(flat_model_params)

        compiled_model_fn = (
            jax.jit(model_fn)
            .lower(
                params=model_params,
                input_ids=jnp.arange(n_anchors * hn_args.hn_surface_maxlen).reshape(
                    (-1, data_args.block_size)
                ),
            )
            .compile()
        )

        hypernet_flops_per_token = compiled_hypernet_fn.cost_analysis()[0][
            "flops"
        ] / np.prod(example_surface_forms.shape)
        model_flops_per_token = compiled_model_fn.cost_analysis()[0]["flops"] / (
            n_anchors * hn_args.hn_surface_maxlen
        )

        print("Estimated FLOPs per token:")
        print(f"Hypernet: {hypernet_flops_per_token:.2f}")
        print(f"Model: {model_flops_per_token:.2f}")

        n_hypernet_params = sum(
            np.prod(x.shape)
            for x in traverse_util.flatten_dict(state.params["hypernet"]).values()
        )
        n_model_params = sum(
            np.prod(x.shape) for x in traverse_util.flatten_dict(model_params).values()
        )

        print(f"Hypernet #params: {n_hypernet_params}")
        print(f"Model #params: {n_model_params}")

        sys.exit(0)

    print("Full Trainable parameters:")
    pprint([k for k, v in traverse_util.flatten_dict(labels).items() if v == "train"])
    print("Non-trainable parameters:")
    pprint([k for k, v in traverse_util.flatten_dict(labels).items() if v == "freeze"])

    def loss_fn(
        logits,
        labels,
        attention_mask,
        position_ids,
        loss_mode,
        byte_lengths=None,
        with_bpb=False,
    ):
        attention_mask_2d = attention_mask.any(axis=-1)

        if loss_mode == "clm":
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]

            last_token_in_sequence = position_ids[..., 1:] == 0
            shift_attention_mask = attention_mask_2d[..., 1:] * ~last_token_in_sequence

            loss = (
                optax.softmax_cross_entropy(
                    shift_logits, onehot(shift_labels, shift_logits.shape[-1])
                )
                * shift_attention_mask
            )

            if with_bpb:
                return (
                    loss.sum() / shift_attention_mask.sum(),
                    (loss.sum(-1) / byte_lengths.sum(-1)).mean(),
                )

            return loss.sum() / shift_attention_mask.sum()
        elif loss_mode == "mlm":
            label_mask = jnp.where((labels != -100) & (attention_mask_2d == 1), 1.0, 0.0)
            loss = (
                optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
                * label_mask
            )
            loss = loss.sum() / label_mask.sum()

            if with_bpb:
                raise NotImplementedError()

            return loss

    def identity_train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
        surface_forms = batch.pop("target_surface_forms")
        priors = batch.pop("target_priors")
        ids_to_embed = batch.pop("ids_to_embed")
        lang_index = batch.pop("lang_index")

        source_embeddings_in = (
            state.source_embeddings[:, : hn_args.n_embd]
            if state.source_embeddings is not None
            else None
        )
        source_embeddings_out = (
            state.source_embeddings[:, hn_args.n_embd :]
            if state.source_embeddings is not None and out_embedding_path is not None
            else None
        )

        def compute_loss(params):
            predicted_embeddings_in, predicted_embeddings_out, _ = state.apply_fn(
                {"params": params["hypernet"]},
                target_surface_forms=surface_forms,
                target_priors=priors,
                source_embeddings=state.source_embeddings,
                lang_index=lang_index,
            )

            target_embeddings_in = jnp.take(source_embeddings_in, ids_to_embed, axis=0)
            in_loss = (
                jnp.square(predicted_embeddings_in - target_embeddings_in)
                .sum(-1)
                .mean()
            )

            if out_embedding_path is not None:
                target_embeddings_out = jnp.take(
                    source_embeddings_out, ids_to_embed, axis=0
                )
                out_loss = (
                    jnp.square(predicted_embeddings_out - target_embeddings_out)
                    .sum(-1)
                    .mean()
                )

                loss = (in_loss + out_loss) / 2.0
            else:
                loss = in_loss

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {
            "identity_loss": loss,
            "learning_rate": random_learning_rate_fn(
                state.step // training_args.gradient_accumulation_steps
            ),
        }
        return new_state, metrics

    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        position_ids = batch.pop("position_ids")
        labels = batch.pop("labels")
        target_surface_forms = batch.pop("target_surface_forms")
        target_priors = batch.pop("target_priors")
        ids_to_embed = batch.pop("ids_to_embed")
        lang_index = batch.pop("lang_index")

        source_embeddings_in = (
            state.source_embeddings[:, : hn_args.n_embd]
            if state.source_embeddings is not None
            else None
        )
        source_embeddings_out = (
            state.source_embeddings[:, hn_args.n_embd :]
            if state.source_embeddings is not None and out_embedding_path is not None
            else None
        )

        def compute_embeddings_and_logits(
            params,
            input_ids,
            surface_forms,
            priors,
            mask,
            special_indices,
            special_indices_in_reference,
        ):
            predicted_embeddings_in, predicted_embeddings_out, biases = state.apply_fn(
                {"params": params["hypernet"]},
                target_surface_forms=surface_forms,
                target_priors=priors,
                source_embeddings=state.source_embeddings,
                lang_index=lang_index,
            )
            if training_args.overwrite_special_token_embeddings and source_embeddings_in is not None:
                predicted_embeddings_in = predicted_embeddings_in.at[
                    special_indices
                ].set(source_embeddings_in[special_indices_in_reference])

            params_with_updated_embeddings = traverse_util.flatten_dict(params["inner"])
            params_with_updated_embeddings[in_embedding_path] = predicted_embeddings_in
            if bias_path is not None:
                # disable bias, it is explicitly added afterwards
                params_with_updated_embeddings[bias_path] = jnp.zeros(
                    len(surface_forms), dtype=predicted_embeddings_in.dtype
                )
            if training_args.overwrite_special_token_embeddings and out_embedding_path is not None:
                if source_embeddings_out is not None:
                    predicted_embeddings_out = predicted_embeddings_out.at[
                        special_indices
                    ].set(source_embeddings_out[special_indices_in_reference])
                params_with_updated_embeddings[
                    out_embedding_path
                ] = predicted_embeddings_out.T

            params_with_updated_embeddings = traverse_util.unflatten_dict(
                params_with_updated_embeddings
            )

            logits = model_fn(
                input_ids=input_ids,
                attention_mask=jnp.expand_dims(attention_mask, axis=1),
                position_ids=position_ids,
                params=params_with_updated_embeddings,
                dropout_rng=dropout_rng
                if training_args.run_backbone_in_training_mode
                else None,
                train=training_args.run_backbone_in_training_mode,
            ).logits

            logits = logits + mask[None, None, :]

            if training_args.learnable_bias:
                logits = logits + biases[None, None, :]

            if training_args.add_target_priors_to_bias:
                logits = logits + target_priors[None, None, :]

            return predicted_embeddings_in, predicted_embeddings_out, logits

        def compute_loss(params):
            (
                predicted_embeddings_in,
                predicted_embeddings_out,
                logits,
            ) = compute_embeddings_and_logits(
                params,
                input_ids,
                target_surface_forms,
                target_priors,
                jnp.where(batch["mask"], 0.0, NEGATIVE_INF_FILL_VALUE),
                batch["special_indices"],
                batch["special_indices_in_reference"],
            )
            loss = loss_fn(logits, labels, attention_mask, position_ids, training_args.loss)

            if hn_args.hn_embed_using_source_embeddings:
                if training_args.apply_lexical_loss_to_init:
                    target_in = initial_input_embeddings[ids_to_embed]
                    target_out = (
                        initial_output_embeddings[ids_to_embed]
                        if out_embedding_path is not None
                        else None
                    )
                    lexical_overlap_mask = jnp.ones(
                        len(target_surface_forms), dtype=bool
                    )
                else:
                    target_in = source_embeddings_in[target_surface_forms[:, 0]]
                    if out_embedding_path is not None:
                        target_out = source_embeddings_out[target_surface_forms[:, 0]]
                    else:
                        target_out = None

                    lexical_overlap_mask = (
                        target_surface_forms[:, 1:] == hn_tokenizer.pad_token_id
                    ).all(axis=1)

                if training_args.lexical_loss_kind == "mse":

                    def distance_fn(x, y):
                        return jnp.square(x - y).sum(axis=-1)

                elif training_args.lexical_loss_kind == "rmse":

                    def distance_fn(x, y):
                        return jnp.linalg.norm(x - y, axis=-1)

                elif training_args.lexical_loss_kind == "huber":
                    HUBER_DELTA = 1e-3
                    HUBER_CORRECTION = 30  # approx scale diff

                    def distance_fn(x, y):
                        return (
                            huber_loss(x, y, delta=HUBER_DELTA).sum(axis=-1)
                            / HUBER_DELTA
                            / HUBER_CORRECTION
                        )

                lexical_loss_in = (
                    distance_fn(predicted_embeddings_in, target_in)
                    * lexical_overlap_mask
                ) / jnp.linalg.norm(target_in, axis=1).mean()

                if out_embedding_path is not None:
                    lexical_loss_out = (
                        distance_fn(predicted_embeddings_out, target_out)
                        * lexical_overlap_mask
                    ) / jnp.linalg.norm(target_out, axis=1).mean()

                    lexical_loss = (lexical_loss_in + lexical_loss_out) / 2.0
                else:
                    lexical_loss = lexical_loss_in

                min_lexical_loss = lexical_loss.min()
                max_lexical_loss = lexical_loss.max()
                mean_lexical_loss = lexical_loss.mean()
                mean_lexical_overlap = lexical_overlap_mask.mean()
                lexical_loss = (
                    lexical_loss.sum()
                    / (lexical_overlap_mask.sum() + EPSILON)
                )
                loss = loss + lexical_loss * training_args.lexical_loss_weight
            else:
                min_lexical_loss = 0.0
                max_lexical_loss = 0.0
                mean_lexical_loss = 0.0
                mean_lexical_overlap = 0.0
                lexical_loss = 0.0

            return loss, (lexical_loss, min_lexical_loss, max_lexical_loss, mean_lexical_loss, mean_lexical_overlap)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, (lexical_loss, min_lexical_loss, max_lexical_loss, mean_lexical_loss, mean_lexical_overlap)), grad = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {
            "loss": loss,
            "min_lexical_loss": min_lexical_loss,
            "max_lexical_loss": max_lexical_loss,
            "mean_lexical_loss": mean_lexical_loss,
            "mean_lexical_overlap": mean_lexical_overlap,
            "lexical_loss": lexical_loss,
            "learning_rate": random_learning_rate_fn(
                state.step // training_args.gradient_accumulation_steps
            ),
        }
        return new_state, metrics

    def eval_step(
        state, batch, input_embeddings=None, output_embeddings=None, biases=None
    ):
        metrics = {}
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        position_ids = batch.pop("position_ids")
        mask = jnp.where(batch.pop("mask"), 0.0, NEGATIVE_INF_FILL_VALUE)
        special_indices = batch.pop("special_indices")
        special_indices_in_reference = batch.pop("special_indices_in_reference")
        labels = batch.pop("labels")
        byte_lengths = batch.pop("byte_lengths")
        target_surface_forms = batch.pop("target_surface_forms")
        target_priors = batch.pop("target_priors")
        lang_index = batch.pop("lang_index")

        source_embeddings_in = (
            state.source_embeddings[:, : hn_args.n_embd]
            if state.source_embeddings is not None
            else None
        )
        source_embeddings_out = (
            state.source_embeddings[:, hn_args.n_embd :]
            if state.source_embeddings is not None and out_embedding_path is not None
            else None
        )

        if input_embeddings is None and output_embeddings is None and biases is None:
            input_embeddings, output_embeddings, biases = state.apply_fn(
                {"params": state.params["hypernet"]},
                target_surface_forms=target_surface_forms,
                target_priors=target_priors,
                source_embeddings=state.source_embeddings,
                lang_index=lang_index,
            )
            if training_args.overwrite_special_token_embeddings and source_embeddings_in is not None:
                input_embeddings = input_embeddings.at[special_indices].set(
                    source_embeddings_in[special_indices_in_reference]
                )
            if training_args.overwrite_special_token_embeddings and output_embeddings is not None and source_embeddings_out is not None:
                output_embeddings = output_embeddings.at[special_indices].set(
                    source_embeddings_out[special_indices_in_reference]
                )
        else:
            assert input_embeddings is not None

        params_with_updated_embeddings = traverse_util.flatten_dict(
            state.params["inner"]
        )
        params_with_updated_embeddings[in_embedding_path] = input_embeddings
        if bias_path is not None:
            # disable bias, it is explicitly added afterwards via target_priors
            params_with_updated_embeddings[bias_path] = jnp.zeros(
                len(target_surface_forms), dtype=input_embeddings.dtype
            )

        if out_embedding_path is not None:
            params_with_updated_embeddings[out_embedding_path] = output_embeddings.T

        params_with_updated_embeddings = traverse_util.unflatten_dict(
            params_with_updated_embeddings
        )

        logits = model_fn(
            input_ids=input_ids,
            attention_mask=jnp.expand_dims(attention_mask, axis=1),
            position_ids=position_ids,
            params=params_with_updated_embeddings,
            train=False
        ).logits
        logits = logits + mask[None, None, :]

        if training_args.learnable_bias:
            logits = logits + biases[None, None, :]

        if training_args.add_target_priors_to_bias:
            logits = logits + target_priors[None, None, :]

        if training_args.loss == "clm":
            metrics["loss"], metrics["bpb"] = loss_fn(
                logits,
                labels,
                attention_mask,
                position_ids,
                training_args.loss,
                byte_lengths=byte_lengths,
                with_bpb=True,
            )
        else:
            metrics["acc"] = jnp.sum(jnp.argmax(logits, axis=-1) == labels) / jnp.sum(
                labels != -100
            )
            metrics["loss"] = loss_fn(
                logits,
                labels,
                attention_mask,
                position_ids,
                training_args.loss,
            )

        if data_args.use_passthrough_hypernet:
            # no need to cache in this case
            input_embeddings = None
            output_embeddings = None
            biases = None

        return metrics, input_embeddings, output_embeddings, biases

    if not training_args.debug:
        batch_pspecs = get_batch_pspecs(
            next(iter(train_dataloaders[0]))
        )  # get a dummy batch to compute sharding
        batch_shardings = jax.tree_map(lambda x: NamedSharding(MESH, x), batch_pspecs)

        if training_args.identity_steps > 0:
            identity_batch_pspecs = get_batch_pspecs(
                next(iter(identity_train_dataloader))
            )  # get a dummy batch to compute sharding
            identity_batch_shardings = jax.tree_map(
                lambda x: NamedSharding(MESH, x), identity_batch_pspecs
            )
            identity_train_step = jax.jit(
                identity_train_step,
                in_shardings=(state_shardings, identity_batch_shardings),
                out_shardings=(state_shardings, None),
            )

        train_step = jax.jit(
            train_step,
            in_shardings=(state_shardings, batch_shardings),
            out_shardings=(state_shardings, None),
        )
        eval_step = jax.jit(
            eval_step, in_shardings=(state_shardings, batch_shardings, None, None, None)
        )
    else:
        batch_shardings = None

    if jax.process_index() == 0:
        run = wandb.init(project="zett", name=name)
        model.config.wandb_run_id = run.id

        run.log_code()
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        wandb.config.update(hn_args)

    def eval_loop():
        all_eval_metrics = {}
        valid_names = [
            (name + "_" + os.path.splitext(data_args.extra_valid_files[i])[0]).replace(
                "/", "_"
            )
            for i, name in enumerate(data_args.extra_valid_tokenizer_names)
        ] + ["main"]

        for valid_name, loader in zip(
            valid_names, extra_valid_dataloaders + [valid_dataloader]
        ):
            raw_eval_metrics = []
            eval_lang_indices = []
            metrics_tracker = {}
            for lang_code in data_args.langs:
                metrics_tracker[f"{lang_code}_avg_byte_length"] = []
                metrics_tracker[f"{lang_code}_unk_ratio"] = []

            input_embeddings = None
            output_embeddings = None
            biases = None

            for i, batch in tqdm(
                enumerate(loader),
                desc="Evaluating...",
                disable=jax.process_index() != 0,
            ):
                batch = prepare_batch(batch, metrics_tracker)

                if jax.process_count() > 1:
                    batch = jax.tree_map(
                        lambda x: x[0], process_allgather(batch)
                    )  # sync batch across processes, TODO: put in prepare_batch if it works

                batch = to_global_batch(batch, batch_shardings)

                eval_lang_indices.append(batch["lang_index"])

                model.config.vocab_size = batch["target_surface_forms"].shape[0]
                (
                    metrics,
                    batch_input_embeddings,
                    batch_output_embeddings,
                    batch_biases,
                ) = eval_step(state, batch, input_embeddings, output_embeddings, biases)

                if i == 0 and not loader.collate_fn.data_args.do_tokenizer_sampling:
                    # enough to compute embeddings once for this tokenizer
                    input_embeddings = batch_input_embeddings
                    output_embeddings = batch_output_embeddings
                    biases = batch_biases

                raw_eval_metrics.append(metrics)

            eval_lang_indices = np.array(eval_lang_indices)
            raw_eval_metrics = jax.tree_map(
                lambda x: x.flatten(), stack_forest(raw_eval_metrics)
            )

            eval_metrics = {}
            for k, v in raw_eval_metrics.items():
                for lang_idx, lang_code in enumerate(data_args.langs):
                    mask = eval_lang_indices == lang_idx
                    if mask.sum() > 0:
                        lang_prefix = "" if lang_code is None else lang_code + "_"
                        eval_metrics[f"{lang_prefix}{k}"] = np.mean(v[mask])

            for lang_code in data_args.langs:
                name = f"{lang_code}_avg_byte_length"
                if len(metrics_tracker[name]) > 0:
                    eval_metrics[f"{lang_code}_avg_byte_length"] = np.mean(
                        metrics_tracker[name]
                    )
                    eval_metrics[f"{lang_code}_std_byte_length"] = np.std(
                        metrics_tracker[name]
                    )
                metrics_tracker[name] = []

                name = f"{lang_code}_unk_ratio"
                if len(metrics_tracker[name]) > 0:
                    eval_metrics[name] = np.mean(metrics_tracker[name])
                metrics_tracker[name] = []

            all_eval_metrics.update(
                {"eval/" + valid_name + "_" + k: v for k, v in eval_metrics.items()}
            )

        return all_eval_metrics

    if training_args.do_train:
        train_start = time.time()
        train_metrics = []
        train_lang_indices = []

        logger.info("***** Running training *****")
        logger.info(f"  Batch Size = {training_args.train_batch_size}")
        logger.info(f"  Total optimization steps = {training_args.steps}")

        diters = [iter(train_dataloader) for train_dataloader in train_dataloaders]
        if training_args.identity_steps > 0:
            identity_diter = iter(identity_train_dataloader)
        else:
            identity_diter = None

        metrics_tracker = {}
        for lang_code in data_args.langs:
            metrics_tracker[f"{lang_code}_avg_byte_length"] = []
            metrics_tracker[f"{lang_code}_unk_ratio"] = []
            metrics_tracker[f"{lang_code}_pad_ratio"] = []

        if training_args.eval_at_step_zero:
            eval_metrics = eval_loop()

            if jax.process_index() == 0:
                print(eval_metrics)
                wandb.log(eval_metrics, step=step + 1)

        for step in tqdm(range(training_args.steps), disable=jax.process_index() != 0):
            do_replay = resume_step > step * training_args.gradient_accumulation_steps
            rng, input_rng = jax.random.split(rng)

            for _ in range(training_args.gradient_accumulation_steps):
                diter_index = jax.random.choice(
                    input_rng, len(train_dataloaders), p=train_probs
                )

                current_diter = (
                    identity_diter
                    if step < training_args.identity_steps
                    else diters[diter_index]
                )
                current_step_fn = (
                    identity_train_step
                    if step < training_args.identity_steps
                    else train_step
                )

                try:
                    batch = next(current_diter)
                except StopIteration:
                    if step < training_args.identity_steps:
                        identity_diter = iter(identity_train_dataloader)
                        current_diter = identity_diter
                    else:
                        diters[diter_index] = iter(train_dataloaders[diter_index])
                        current_diter = diters[diter_index]

                    batch = next(current_diter)

                if do_replay:
                    continue

                batch = prepare_batch(batch, metrics_tracker)

                if jax.process_count() > 1:
                    batch = jax.tree_map(
                        lambda x: x[0], process_allgather(batch)
                    )  # sync batch across processes

                if data_args.use_passthrough_hypernet:
                    assert batch["target_surface_forms"].shape[1] == 1

                train_lang_indices.append(batch["lang_index"])
                batch = to_global_batch(batch, batch_shardings)

                model.config.vocab_size = batch["target_surface_forms"].shape[0]

                state, train_metric = current_step_fn(state, batch)
                train_metrics.append(train_metric)

            if do_replay:
                continue

            if (step + 1) % training_args.logging_steps == 0:
                metrics_to_disaggregate = {"loss"}

                raw_metrics = jax.tree_map(
                    lambda x: x.flatten(), stack_forest(train_metrics)
                )
                train_lang_indices = np.array(train_lang_indices)

                metrics = {}
                for k, v in raw_metrics.items():
                    if k in metrics_to_disaggregate:
                        for lang_idx, lang_code in enumerate(data_args.langs):
                            mask = train_lang_indices == lang_idx
                            if mask.sum() > 0:
                                metrics[f"{lang_code}_{k}"] = np.mean(v[mask])
                            else:
                                metrics[f"{lang_code}_{k}"] = np.nan

                    metrics[k] = v.mean()

                metrics["time"] = time.time() - train_start

                for lang_code in data_args.langs:
                    name = f"{lang_code}_avg_byte_length"
                    if len(metrics_tracker[name]) > 0:
                        metrics[f"{lang_code}_avg_byte_length"] = np.mean(
                            metrics_tracker[name]
                        )
                        metrics[f"{lang_code}_std_byte_length"] = np.std(
                            metrics_tracker[name]
                        )
                    metrics_tracker[name] = []

                    name = f"{lang_code}_unk_ratio"
                    if len(metrics_tracker[name]) > 0:
                        metrics[name] = np.mean(metrics_tracker[name])
                    metrics_tracker[name] = []

                    name = f"{lang_code}_pad_ratio"
                    if len(metrics_tracker[name]) > 0:
                        metrics[name] = np.mean(metrics_tracker[name])
                    metrics_tracker[name] = []

                metrics = {"train/" + k: v for k, v in metrics.items()}

                if jax.process_index() == 0:
                    print(metrics)
                    wandb.log(metrics, step=step + 1)
                train_metrics = []
                train_lang_indices = []

            if (step + 1) % training_args.save_steps == 0:
                save_state = jax.device_get(
                    global_array_to_host_local_array(state, MESH, state_pspecs)
                )

                if training_args.save_state:
                    open(
                        os.path.join(training_args.output_dir, "state.msgpack"), "wb"
                    ).write(serialization.to_bytes(save_state))
                model.save_pretrained(
                    training_args.output_dir, params=save_state.params["hypernet"]
                )

                if training_args.backbone_training == "full":
                    full_model_path = os.path.join(
                        training_args.output_dir, "full_model"
                    )
                    model.save_pretrained(
                        full_model_path, params=save_state.params["inner"]
                    )

                if hn_tokenizer is not None:
                    hn_tokenizer.save_pretrained(training_args.output_dir)

            if (step + 1) % training_args.eval_steps == 0:
                eval_metrics = eval_loop()

                if jax.process_index() == 0:
                    print(eval_metrics)
                    wandb.log(eval_metrics, step=step + 1)
    else:
        assert (
            training_args.init_from_params is not None
            or training_args.resume_from_checkpoint is not None
        )
        eval_metrics = eval_loop()

        print(eval_metrics)
        wandb.log(eval_metrics, step=0)


if __name__ == "__main__":
    main()
