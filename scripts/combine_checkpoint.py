from dataclasses import dataclass
from transformers import HfArgumentParser
import jax
import numpy as np
from flax.serialization import msgpack_restore, msgpack_serialize
from flax import traverse_util
from jax.sharding import PartitionSpec as P, NamedSharding
import regex as re
import os
from jax.experimental.multihost_utils import (
    host_local_array_to_global_array,
    global_array_to_host_local_array,
    process_allgather,
)
from functools import partial
from transformers import AutoConfig
from tqdm.auto import tqdm
import math

from zett.utils import keystr
from zett.model import HYPERNET_MODEL_PARALLEL_MAP, MODEL_PARALLEL_MAPS


@dataclass
class Args:
    name: str
    batch_size: int = 16


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    MESH = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(jax.process_count() * jax.local_device_count()),
        ["model"],
    )

    config = AutoConfig.from_pretrained(args.name)

    local_param_path_hypernet = os.path.join(args.name, "flax_model.local.msgpack")
    global_param_path_hypernet = os.path.join(args.name, "flax_model.msgpack")

    local_param_path_inner = os.path.join(
        args.name, "full_model", "flax_model.local.msgpack"
    )
    global_param_path_inner = os.path.join(
        args.name, "full_model", "flax_model.msgpack"
    )

    if not os.path.exists(local_param_path_hypernet):
        os.rename(global_param_path_hypernet, local_param_path_hypernet)

    if not os.path.exists(local_param_path_inner) and os.path.exists(
        global_param_path_inner
    ):
        os.rename(global_param_path_inner, local_param_path_inner)

    params = {
        "params": {
            "hypernet": msgpack_restore(open(local_param_path_hypernet, "rb").read())
        }
    }

    if os.path.exists(local_param_path_inner):
        params["params"]["inner"] = msgpack_restore(
            open(local_param_path_inner, "rb").read()
        )

    def get_pspec(path, v, map=None):
        path_tuple = tuple(str(keystr(x)) for x in path)
        path = ".".join(path_tuple)

        if map is not None:
            for key, value in map.items():
                if re.match(key, path):
                    pspec = value
                    print(f"Sharding {path} with {pspec}.")
                    return P(*pspec)

        return P()

    model_parallel_keys = HYPERNET_MODEL_PARALLEL_MAP
    model_parallel_keys.update(MODEL_PARALLEL_MAPS.get(config.model_type, {}))

    pspecs = jax.tree_util.tree_map_with_path(
        partial(get_pspec, map=HYPERNET_MODEL_PARALLEL_MAP), params
    )
    none_pspecs = jax.tree_util.tree_map_with_path(get_pspec, params)

    shardings = jax.tree_map(lambda x: NamedSharding(MESH, x), pspecs)
    none_shardings = jax.tree_map(lambda x: NamedSharding(MESH, x), none_pspecs)

    global_params = host_local_array_to_global_array(params, MESH, pspecs)

    def params_to_host(params):
        return params

    flat_params = traverse_util.flatten_dict(global_params)
    flat_shardings = traverse_util.flatten_dict(shardings)
    flat_out_shardings = traverse_util.flatten_dict(none_shardings)

    keys = list(flat_params.keys())
    n_batches = math.ceil(len(keys) / args.batch_size)

    flat_out_params = {}

    for i in tqdm(range(n_batches), disable=jax.process_index() != 0):
        batch_keys = keys[i * args.batch_size : (i + 1) * args.batch_size]

        batch_params = jax.jit(
            params_to_host,
            in_shardings=([flat_shardings[k] for k in batch_keys],),
            out_shardings=[flat_out_shardings[k] for k in batch_keys],
        )([flat_params[k] for k in batch_keys])

        for key, param in zip(batch_keys, batch_params):
            flat_out_params[key] = param

    out_params = traverse_util.unflatten_dict(flat_out_params)

    open(global_param_path_hypernet, "wb").write(
        msgpack_serialize(out_params["params"]["hypernet"])
    )
    if "inner" in out_params["params"]:
        open(global_param_path_inner, "wb").write(
            msgpack_serialize(out_params["params"]["inner"])
        )
