import math
from dataclasses import dataclass
from flax import serialization, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
import os
import transformers
from tqdm import tqdm
import sys
import gc

from zett.utils import (
    get_surface_form_matrix,
    SHARDING,
    NEGATIVE_INF_FILL_VALUE,
    load_params,
    get_sample_indices,
)
from zett.model import (
    Hypernet,
    IN_EMBEDDING_PATHS,
    OUT_EMBEDDING_PATHS,
    BIAS_PATHS,
)
from zett.tokenizer_converters import convert_to_byte_level


@dataclass
class Args:
    output: str
    checkpoint_path: str = (
        "output_gpt2_noise_std_1.0_inter_embed_bias_scratch_nlayers=6"
    )
    tokenizer_name: str = "artifacts/gpt2_unigramify"
    model_class: str = "AutoModel"
    # args for target model and projection
    target_model: str = "gpt2"
    copy_inner_parameters_from: str = None
    dtype: str = "bfloat16"
    revision: str = None
    do_batching: bool = True
    batch_size: int = 16384  # must be multiple of 8 for sharding
    sample_batches: bool = False
    min_k: int = 10
    n_samples: int = 100
    lang_path: str = None
    lang_code: str = None
    make_whitespace_consistent: bool = True
    save_pt: bool = False


def batched_inference(target_surface_form_matrix, target_priors, config, args):
    original_length = len(target_surface_form_matrix)

    if args.sample_batches:
        indices = get_sample_indices(
            original_length, target_priors, args.batch_size, args.min_k, args.n_samples
        )
        empty_in_last_batch = 0
    else:
        shuffled_indices = np.random.permutation(original_length)
        total_length = math.ceil(original_length / args.batch_size) * args.batch_size
        padded = np.pad(shuffled_indices, (0, total_length - original_length))
        indices = np.array_split(padded, total_length // args.batch_size)
        empty_in_last_batch = total_length - original_length

    predicted_embeddings_in = np.zeros(
        (original_length, config.hidden_size), dtype=np.float32
    )
    predicted_embeddings_out = (
        np.zeros((original_length, config.hidden_size), dtype=np.float32)
        if embedding_path_out is not None
        else None
    )
    predicted_bias = (
        np.zeros(original_length, dtype=np.float32)
        if bias_path is not None
        else None
    )

    for i, batch_indices in tqdm(enumerate(indices), desc="Predicting batches..."):
        last_batch = i == len(indices) - 1
        (
            predicted_embeddings_in_batch,
            predicted_embeddings_out_batch,
            predicted_bias_batch,
        ) = predict(
                    jax.device_put(target_surface_form_matrix[batch_indices], SHARDING.reshape((-1, 1)), ),
                    jax.device_put(target_priors[batch_indices], SHARDING.reshape((-1,))),
        )

        if last_batch and empty_in_last_batch > 0:
            batch_indices = batch_indices[:-empty_in_last_batch]
            predicted_embeddings_in_batch = predicted_embeddings_in_batch[
                                            :-empty_in_last_batch
                                            ]
            if predicted_embeddings_out is not None:
                predicted_embeddings_out_batch = predicted_embeddings_out_batch[
                                                 :-empty_in_last_batch
                                                 ]
            if predicted_bias is not None:
                predicted_bias_batch = predicted_bias_batch[:-empty_in_last_batch]
        predicted_embeddings_in[batch_indices] += predicted_embeddings_in_batch
        if predicted_embeddings_out is not None:
            predicted_embeddings_out[
                batch_indices
            ] += predicted_embeddings_out_batch
        if predicted_bias is not None:
            predicted_bias[batch_indices] += predicted_bias_batch

    if args.sample_batches:
        # since tokens can be seen multiple times, those embeddings need to be averaged
        values, counts = np.unique(indices, return_counts=True)
        assert sorted(values) == list(values)

        predicted_embeddings_in /= counts[:, None]
        if predicted_embeddings_out is not None:
            predicted_embeddings_out /= counts[:, None]
        if predicted_bias is not None:
            predicted_bias /= counts

    return predicted_embeddings_in, predicted_embeddings_out, predicted_bias


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    config = AutoConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)

    if args.lang_code is not None:
        if hasattr(config, "langs"):
            langs = config.langs
        else:
            assert args.lang_path is not None

            langs = [x.strip() for x in open(args.lang_path).readlines()]

        lang_index = jnp.array(langs.index(args.lang_code), dtype=jnp.int32)
    else:
        lang_index = None

    hypernet = Hypernet(config, dtype=getattr(jnp, args.dtype))
    hypernet_params = serialization.msgpack_restore(
        open(os.path.join(args.checkpoint_path, "flax_model.msgpack"), "rb").read()
    )
    hypernet_params = jax.tree_map(
        lambda x: x.astype(getattr(jnp, args.dtype)), hypernet_params
    )

    previous_tokenizer_of_downstream_model = AutoTokenizer.from_pretrained(
        args.target_model
    )
    hn_tokenizer = type(previous_tokenizer_of_downstream_model).from_pretrained(args.checkpoint_path)
    hn_tokenizer = convert_to_byte_level(hn_tokenizer)[0]
    if hn_tokenizer.pad_token is None:
        hn_tokenizer.pad_token = hn_tokenizer.eos_token

    # downstream model
    downstream_config = AutoConfig.from_pretrained(args.target_model)
    downstream_model = getattr(transformers, "Flax" + args.model_class).from_config(
        config=downstream_config, _do_init=False
    )
    flat_downstream_params = traverse_util.flatten_dict(
        load_params(args.target_model, revision=args.revision)
    )

    bias_path = BIAS_PATHS[downstream_model.config.model_type]
    embedding_path_in = IN_EMBEDDING_PATHS[downstream_model.config.model_type]
    embedding_path_out = (
        OUT_EMBEDDING_PATHS[downstream_model.config.model_type]
        if not downstream_model.config.tie_word_embeddings
        else None
    )

    if args.model_class == "AutoModel":
        embedding_path_in = embedding_path_in[1:]
        bias_path = None

    source_embeddings_in = flat_downstream_params[embedding_path_in]

    if embedding_path_out is not None:
        source_embeddings_out = flat_downstream_params[embedding_path_out].T
        source_embeddings_stacked = np.concatenate(
            [source_embeddings_in, source_embeddings_out], axis=1
        )
    else:
        source_embeddings_out = None
        source_embeddings_stacked = source_embeddings_in

    if args.copy_inner_parameters_from is not None:
        flat_downstream_params = traverse_util.flatten_dict(
            load_params(args.copy_inner_parameters_from)
        )

    tokenizer = convert_to_byte_level(
        tokenizer,
        make_whitespace_consistent=args.make_whitespace_consistent,
        match_special_tokens_to=previous_tokenizer_of_downstream_model,
    )[0]

    target_surface_form_matrix, n_truncated = get_surface_form_matrix(
        tokenizer, maxlen=config.hn_surface_maxlen, tokenizer_to_use=hn_tokenizer
    )

    print(f"Truncated {n_truncated} tokens.")

    if hasattr(tokenizer._tokenizer.model, "get_scores"):
        target_priors = tokenizer._tokenizer.model.get_scores()
    else:
        print("WARNING: using uniform priors, get_scores() not available.")
        target_priors = [0.0] * len(tokenizer)

    target_priors += [0.0] * (
        len(tokenizer) - len(target_priors)
    )  # for added special tokens
    target_priors = np.array(target_priors)

    @jax.jit
    def predict(target_surface_form_matrix, target_priors):
        (
            predicted_embeddings_in,
            predicted_embeddings_out,
            predicted_bias,
        ) = hypernet.apply(
            {"params": hypernet_params},
            target_surface_form_matrix,
            target_priors,
            source_embeddings=source_embeddings_stacked,
            lang_index=lang_index,
        )
        return predicted_embeddings_in, predicted_embeddings_out, predicted_bias

    original_length = len(target_surface_form_matrix)

    if args.do_batching:
        predicted_embeddings_in, predicted_embeddings_out, predicted_bias = batched_inference(
            target_surface_form_matrix, target_priors, config, args
        )
    else:
        # pad to multiple of 128
        pad_to = 128

        n_pad = pad_to - (original_length % pad_to)
        target_surface_form_matrix = np.pad(
            target_surface_form_matrix, ((0, n_pad), (0, 0)), constant_values=0
        )
        target_priors = np.pad(
            target_priors, (0, n_pad), constant_values=NEGATIVE_INF_FILL_VALUE
        )

        target_surface_form_matrix = jax.device_put(
            target_surface_form_matrix, SHARDING.reshape((-1, 1))
        )
        target_priors = jax.device_put(target_priors, SHARDING.reshape((-1,)))

        predicted_embeddings_in, predicted_embeddings_out, predicted_bias = predict(
            target_surface_form_matrix,
            target_priors,
        )

    predicted_embeddings_in = predicted_embeddings_in[:original_length]
    if predicted_embeddings_out is not None:
        predicted_embeddings_out = predicted_embeddings_out[:original_length]
    if predicted_bias is not None:
        predicted_bias = predicted_bias[:original_length]
    target_surface_form_matrix = target_surface_form_matrix[:original_length]
    target_priors = target_priors[:original_length]

    downstream_model.config.vocab_size = len(tokenizer)

    special_token_embeddings_in = source_embeddings_in[
        np.array(previous_tokenizer_of_downstream_model.all_special_ids)
    ]
    special_token_ids = np.array(
        [
            tokenizer.get_vocab()[x]
            for x in previous_tokenizer_of_downstream_model.all_special_tokens
        ]
    )

    previous_tokenizer_of_downstream_model.save_pretrained(
        args.output
    )  # to get tokenizer_config.json and other metadata
    tokenizer.save_pretrained(args.output)

    predicted_embeddings_in = np.array(predicted_embeddings_in).astype(np.float32)
    predicted_embeddings_in[special_token_ids] = special_token_embeddings_in

    flat_downstream_params[embedding_path_in] = predicted_embeddings_in

    if embedding_path_out is not None:
        special_token_embeddings_out = source_embeddings_out[
            np.array(previous_tokenizer_of_downstream_model.all_special_ids)
        ]

        predicted_embeddings_out = np.array(predicted_embeddings_out).astype(np.float32)
        predicted_embeddings_out[special_token_ids] = special_token_embeddings_out

        flat_downstream_params[embedding_path_out] = predicted_embeddings_out.T

    if bias_path is not None:
        flat_downstream_params[bias_path] = predicted_bias
    else:
        open(os.path.join(args.output, "bias.msgpack"), "wb").write(
            serialization.msgpack_serialize(predicted_bias)
        )

    downstream_model.save_pretrained(
        args.output,
        params=traverse_util.unflatten_dict(flat_downstream_params),
        max_shard_size="20GB",  # needed for 7B PT conversion https://github.com/huggingface/transformers/issues/20248
    )

    del downstream_model
    del flat_downstream_params
    del predicted_embeddings_in
    del predicted_embeddings_out

    gc.collect()

    if args.save_pt:
        pt_model = getattr(transformers, args.model_class).from_pretrained(
            args.output, from_flax=True
        )
        pt_model.save_pretrained(args.output)
