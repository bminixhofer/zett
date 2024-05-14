from transformers import HfArgumentParser
from dataclasses import dataclass
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model

from flax import serialization, traverse_util
import os
import regex as re
import torch

from zett.model import Hypernet as FlaxHypernet

from hf_hypernet.configuration_hypernet import ZettHypernetConfig
from hf_hypernet.modeling_hypernet import ZettHypernet

@dataclass
class Args:
    checkpoint_path: str
    langs_path: str = None
    save_name: str = None
    push_to_hub_name: str = None
    push_to_hub_private: bool = True

if __name__ == "__main__":
    (args,) = HfArgumentParser(Args).parse_args_into_dataclasses()

    ZettHypernetConfig.register_for_auto_class()
    ZettHypernet.register_for_auto_class("AutoModel")

    config = ZettHypernetConfig.from_pretrained(args.checkpoint_path)

    if args.langs_path is not None:
        langs = [x.strip() for x in open(args.langs_path)]
        config.langs = langs
    
    flax_weights = serialization.msgpack_restore(open(os.path.join(args.checkpoint_path, "flax_model.msgpack"), "rb").read())

    flax_weights_for_pt = traverse_util.flatten_dict(flax_weights)
    flax_weights_for_pt = {tuple(re.sub(r"^layers_(\d+)$", "\\1", name) for name in key): value for key, value in flax_weights_for_pt.items()}
    flax_weights_for_pt = traverse_util.unflatten_dict(flax_weights_for_pt)

    flax_hypernet = FlaxHypernet(config)

    hypernet = ZettHypernet(config)
    hypernet = load_flax_weights_in_pytorch_model(hypernet, flax_weights_for_pt)
    if config.hn_embed_lang_id:
        hypernet.lang_embeddings.weight.data = torch.from_numpy(flax_weights_for_pt["model"]["embeddings"]["lang_embedding"]["embedding"])

    if args.save_name is not None:
        hypernet.save_pretrained(args.save_name)

    if args.push_to_hub_name is not None:
        hypernet.push_to_hub(args.push_to_hub_name, private=args.push_to_hub_private)