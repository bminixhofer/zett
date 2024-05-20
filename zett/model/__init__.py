from transformers import FlaxAutoModel
from flax import linen as nn
from flax.core import frozen_dict, unfreeze, freeze
from dataclasses import dataclass
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from typing import Tuple
from functools import partial
import jax

from zett.model.hyper_roberta import HyperRobertaConfig, FlaxHyperRobertaModule
from zett.model.hyper_t5 import HyperT5Config, FlaxHyperT5EncoderModule
from zett.utils import Rescaler

IN_EMBEDDING_PATHS = {
    "gpt2": ("transformer", "wte", "embedding"),
    "roberta": ("roberta", "embeddings", "word_embeddings", "embedding"),
    "xlm-roberta": ("roberta", "embeddings", "word_embeddings", "embedding"),
    "xglm": ("model", "embed_tokens", "embedding"),
    "mistral": ("model", "embed_tokens", "embedding"),
    "llama": ("model", "embed_tokens", "embedding"),
    "gemma": ("model", "embed_tokens", "embedding"),
}
OUT_EMBEDDING_PATHS = {
    "gpt2": ("lm_head", "kernel"),
    "roberta": None,
    "xlm-roberta": None,
    "xglm": None,
    "mistral": ("lm_head", "kernel"),
    "llama": ("lm_head", "kernel"),
    "gemma": None,
}
BIAS_PATHS = {
    "gpt2": None,
    "roberta": ("lm_head", "bias"),
    "xlm-roberta": ("lm_head", "bias"),
    "xglm": None,
    "mistral": None,
    "llama": None,
    "gemma": None,
}
HYPERNET_MODEL_PARALLEL_MAP = {
    # don't shard adafactor
    "opt_state.*?\.(v|v_row|v_col)\..*": P(),
    # source embeddings
    "source_embeddings.*": P("model", None),
    # projections
    "(params|opt_state).*?hypernet.*projection.*dense1.kernel": P(None, "model"),
    "(params|opt_state).*?hypernet.*projection.*dense2.kernel": P("model", None),
    "(params|opt_state).*?hypernet.*projection.*layers_\\d+.kernel": P("model", None),
    # passthrough
    "(params|opt_state).*?hypernet.input_embeddings.embedding": P("model", None),
    "(params|opt_state).*?hypernet.output_embeddings.embedding": P("model", None),
    # roberta
    "(params|opt_state).*?hypernet.*.attention.self.(query|key|value).kernel": P(
        None, "model"
    ),
    "(params|opt_state).*?hypernet.*.attention.output.dense.kernel": P("model", None),
    "(params|opt_state).*?hypernet.*.intermediate.dense.kernel": P(None, "model"),
    "(params|opt_state).*?hypernet.*.output.dense.kernel": P("model", None),
    # t5
    "(params|opt_state).*?hypernet.*.SelfAttention.(q|k|v).kernel": P(None, "model"),
    "(params|opt_state).*?hypernet.*.SelfAttention.o.kernel": P("model", None),
    "(params|opt_state).*?(params|opt_state).hypernet.*.DenseReluDense.wi.*.kernel": P(
        None, "model"
    ),
    "(params|opt_state).*?hypernet.*.DenseReluDense.wo.kernel": P("model", None),
}
MODEL_PARALLEL_MAPS = {
    "llama": {
        # ".*embed_tokens.*embedding": P("model", None),
        "(params|opt_state).*?inner.*self_attn.(q_proj|k_proj|v_proj).kernel.a": P(
            "model", None
        ),  # lora
        "(params|opt_state).*?inner.*self_attn.(q_proj|k_proj|v_proj).kernel.b": P(
            None, "model"
        ),  # lora
        "(params|opt_state).*?inner.*self_attn.(q_proj|k_proj|v_proj).kernel.w": P(
            None, "model"
        ),
        "(params|opt_state).*?inner.*self_attn.(q_proj|k_proj|v_proj).kernel": P(
            None, "model"
        ),
        "(params|opt_state).*?inner.*self_attn.o_proj.kernel": P("model", None),
        # ".*lm_head.kernel": P(None, "model"),
        "(params|opt_state).*?inner.*mlp.down_proj.kernel": P("model", None),
        "(params|opt_state).*?inner.*mlp.up_proj.kernel": P(None, "model"),
        "(params|opt_state).*?inner.*mlp.gate_proj.kernel": P(None, "model"),
    },
    "mistral": {
        # ".*embed_tokens.*embedding": P("model", None),
        "(params|opt_state).*?inner.*self_attn.(q_proj|k_proj|v_proj).kernel": P(
            None, "model"
        ),
        "(params|opt_state).*?inner.*self_attn.o_proj.kernel": P("model", None),
        # ".*lm_head.kernel": P(None, "model"),
        "(params|opt_state).*?inner.*mlp.down_proj.kernel": P("model", None),
        "(params|opt_state).*?inner.*mlp.up_proj.kernel": P(None, "model"),
        "(params|opt_state).*?inner.*mlp.gate_proj.kernel": P(None, "model"),
    },
    "gemma": {
        # ".*embed_tokens.*embedding": P("model", None),
        "(params|opt_state).*?inner.*self_attn.(q_proj|k_proj|v_proj).kernel": P(
            None, "model"
        ),
        "(params|opt_state).*?inner.*self_attn.o_proj.kernel": P("model", None),
        # ".*lm_head.kernel": P(None, "model"),
        "(params|opt_state).*?inner.*mlp.down_proj.kernel": P("model", None),
        "(params|opt_state).*?inner.*mlp.up_proj.kernel": P(None, "model"),
        "(params|opt_state).*?inner.*mlp.gate_proj.kernel": P(None, "model"),
    },
    "xlm-roberta": {
        "(params|opt_state).*?inner.*self.(query|key|value).kernel": P(None, "model"),
        "(params|opt_state).*?inner.*output.dense.kernel": P("model", None),
        "(params|opt_state).*?inner.*intermediate.dense.kernel": P(None, "model"),
        "(params|opt_state).*?inner.*output.dense.kernel": P("model", None),
    },
}


class ProjectorBlock(nn.Module):
    dim: int
    intermediate_dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense1 = nn.Dense(self.intermediate_dim)
        self.dense2 = nn.Dense(self.dim)

        self.ln = nn.LayerNorm()

    def __call__(self, x):
        h = nn.gelu(self.dense2(nn.gelu(self.dense1(x))))
        return self.ln(h + x)


@dataclass
class HypernetArgs:
    hn_model_name_or_path: str = "output_hypernetwork"
    hn_surface_maxlen: int = 16
    hn_n_layers: int = 3
    n_embd: int = 768
    hn_hidden_size: int = None
    hn_intermediate_size: int = None
    # whether to rescale the mean and standard deviation of the predicted embeddings
    # to equal those of the original embeddings
    hn_rescale_embeddings: bool = False
    # whether to embed log unigram probability as an extra item in the input sequence
    hn_embed_target_priors: bool = False
    # whether to add sparse inter-token attention as described in the paper
    hn_add_inter_token_attention: bool = False
    hn_inter_token_attention_bias_by_priors: bool = False
    hn_inter_token_attention_bias_scaler: float = 1.0
    hn_n_inter_token_blocks: int = 16
    hn_embed_using_source_embeddings: bool = True
    hn_concat_last_hidden_state: bool = False
    hn_single_head: bool = False
    hn_predict_bias: bool = True
    hn_num_attention_heads: int = None
    hn_embed_lang_id: bool = False
    hn_model_type: str = "roberta"
    n_langs: int = None  # set in train.py


class PassthroughHypernet(nn.Module):
    """
    A hypernet that embeds the first item in the input sequence using a learned embedding.
    Equivalent to not using a hypernetwork, just learning embeddings directly.
    """
    config: HypernetArgs
    vocab_size: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.has_separate_out_embeddings = getattr(
            self.config, "separate_out_embeddings", False
        )

        self.input_embeddings = nn.Embed(
            self.vocab_size, self.config.n_embd, dtype=self.dtype
        )

        if self.has_separate_out_embeddings:
            self.output_embeddings = nn.Embed(
                self.vocab_size, self.config.n_embd, dtype=self.dtype
            )

        if getattr(self.config, "hn_predict_bias", False):
            self.bias = nn.Embed(self.vocab_size, 1, dtype=self.dtype)

    def __call__(
        self,
        target_surface_forms,
        target_priors,
        source_embeddings=None,
        lang_index=None,
        deterministic: bool = True,
    ):
        predicted_embeddings_in = self.input_embeddings(target_surface_forms[..., 0])

        if self.has_separate_out_embeddings:
            predicted_embeddings_out = self.output_embeddings(
                target_surface_forms[..., 0]
            )
        else:
            predicted_embeddings_out = None

        if getattr(self.config, "hn_predict_bias", False):
            predicted_bias = self.bias(target_surface_forms[..., 0])[..., 0]
        else:
            predicted_bias = jnp.zeros_like(
                target_surface_forms[..., 0], dtype=self.dtype
            )

        return predicted_embeddings_in, predicted_embeddings_out, predicted_bias


class Hypernet(nn.Module):
    config: HypernetArgs
    dtype: jnp.dtype = jnp.float32
    prefix_to_pretrained: Tuple[str] = ("model",)

    def setup(self):
        self.has_separate_out_embeddings = getattr(
            self.config, "separate_out_embeddings", False
        )

        if self.has_separate_out_embeddings:
            n_in_embd = self.config.n_embd * 2
            n_out_embd = self.config.n_embd
        else:
            n_in_embd = self.config.n_embd
            n_out_embd = self.config.n_embd

        if self.config.hn_model_type == "roberta":
            config = HyperRobertaConfig.from_pretrained(
                self.config.hn_model_name_or_path
            )
            config.num_hidden_layers = self.config.hn_n_layers
            config.hidden_size = self.config.hn_hidden_size
            config.intermediate_size = self.config.hn_intermediate_size
            if getattr(self.config, "hn_num_attention_heads", None) is None:
                self.config.hn_num_attention_heads = self.config.hn_hidden_size // 64
            config.num_attention_heads = self.config.hn_num_attention_heads
            self.embed_init_range = config.initializer_range
            module_class = partial(FlaxHyperRobertaModule, add_pooling_layer=False)
        elif self.config.hn_model_type == "t5":
            config = HyperT5Config.from_pretrained(self.config.hn_model_name_or_path)
            config.num_layers = self.config.hn_n_layers

            if self.config.hn_hidden_size is not None:
                config.d_model = self.config.hn_hidden_size
            else:
                self.config.hn_hidden_size = config.d_model

            if self.config.hn_intermediate_size is not None:
                config.d_ff = self.config.hn_intermediate_size
            else:
                self.config.hn_intermediate_size = config.d_ff

            self.embed_init_range = config.initializer_factor * 1.0
            module_class = FlaxHyperT5EncoderModule

        if self.config.hn_embed_using_source_embeddings:
            # do not need to alloc embeddings since inputs_embeds is always used
            config.vocab_size = 1

        config.embed_target_priors = self.config.hn_embed_target_priors
        config.add_inter_token_attention = self.config.hn_add_inter_token_attention
        config.inter_token_attention_bias_by_priors = (
            self.config.hn_inter_token_attention_bias_by_priors
        )
        config.inter_token_attention_bias_scaler = (
            self.config.hn_inter_token_attention_bias_scaler
        )
        config.n_inter_token_blocks = self.config.hn_n_inter_token_blocks
        config.embed_using_source_embeddings = (
            self.config.hn_embed_using_source_embeddings
        )
        config.embed_lang_id = self.config.hn_embed_lang_id
        config.n_langs = self.config.n_langs

        self.pad_token_id = self.config.pad_token_id
        assert self.pad_token_id is not None
        self.model = module_class(config, dtype=self.dtype)

        # need at least one embedding
        self.fallback_embeddings = nn.Embed(
            max(self.config.hn_n_extra_tokens, 1),
            n_in_embd,
            dtype=self.dtype,
            embedding_init=jax.nn.initializers.normal(self.embed_init_range),
        )

        if self.config.hn_embed_using_source_embeddings:
            self.input_projection = nn.Sequential(
                [
                    nn.Dense(self.config.hn_hidden_size, dtype=self.dtype),
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                        dtype=self.dtype,
                    ),
                ]
            )

        if self.config.hn_single_head:
            self.output_projection = nn.Sequential(
                [
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                        dtype=self.dtype,
                    ),
                    nn.Dense(n_in_embd, dtype=self.dtype),
                ]
            )
        else:
            self.output_projection = nn.Sequential(
                [
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                        dtype=self.dtype,
                    ),
                    nn.Dense(n_out_embd, dtype=self.dtype),
                ]
            )
            if self.has_separate_out_embeddings:
                self.output_projection_out = nn.Sequential(
                    [
                        ProjectorBlock(
                            self.config.hn_hidden_size,
                            self.config.hn_intermediate_size,
                            dtype=self.dtype,
                        ),
                        nn.Dense(self.config.n_embd, dtype=self.dtype),
                    ]
                )

        if self.config.hn_rescale_embeddings:
            self.in_scaler = Rescaler(n_in_embd, dtype=self.dtype)
            self.scaler = Rescaler(n_out_embd, dtype=self.dtype)

            if self.has_separate_out_embeddings:
                self.out_scaler = Rescaler(self.config.n_embd, dtype=self.dtype)

        if getattr(self.config, "hn_predict_bias", False):
            self.bias_projection = nn.Dense(1, dtype=self.dtype)

    def init_rescaler(
        self,
        target_surface_forms,
        target_priors,
        source_embeddings,
        lang_index,
        target_embeddings_in,
        target_embeddings_out,
    ):
        if not self.config.hn_rescale_embeddings:
            return self.variables

        params = unfreeze(freeze(self.variables).copy({}))  # awkward..

        win, bin = self.in_scaler.scale_to(
            source_embeddings,
            target_stds=jnp.full(
                source_embeddings.shape[1], fill_value=self.embed_init_range
            ),
            target_means=jnp.zeros(source_embeddings.shape[1]),
        )
        params["params"]["in_scaler"]["w"] = win
        params["params"]["in_scaler"]["b"] = bin

        pred_in, pred_out, _ = self.apply(
            params, target_surface_forms, target_priors, source_embeddings, lang_index
        )

        w, b = self.scaler.scale_to(pred_in, target_embeddings_in)
        params["params"]["scaler"]["w"] = w
        params["params"]["scaler"]["b"] = b

        if target_embeddings_out is not None:
            w_out, b_out = self.out_scaler.scale_to(pred_out, target_embeddings_out)
            params["params"]["out_scaler"]["w"] = w_out
            params["params"]["out_scaler"]["b"] = b_out

        return params

    def __call__(
        self,
        target_surface_forms,
        target_priors=None,
        source_embeddings=None,
        lang_index=None,
        deterministic: bool = True,
    ):
        if self.config.hn_embed_using_source_embeddings:
            use_fallback = target_surface_forms >= self.config.original_vocab_size

            main_ids = jnp.minimum(
                target_surface_forms, self.config.original_vocab_size - 1
            )
            fallback_ids = jnp.maximum(
                target_surface_forms - self.config.original_vocab_size, 0
            )

            source_embeds = jnp.take(source_embeddings, main_ids, axis=0)

            if self.config.hn_rescale_embeddings:
                source_embeds = self.in_scaler(source_embeds)

            inputs_embeds = jnp.where(
                use_fallback[..., None],
                self.fallback_embeddings(fallback_ids),
                source_embeds,
            )

            kwargs = {
                "input_ids": target_surface_forms,
                "inputs_embeds": self.input_projection(inputs_embeds),
            }
        else:
            kwargs = {
                "input_ids": target_surface_forms,
            }

        hidden_states = self.model(
            attention_mask=target_surface_forms != self.pad_token_id,
            target_priors=target_priors,
            lang_index=lang_index,
            deterministic=deterministic,
            **kwargs,
        ).last_hidden_state

        if self.config.hn_concat_last_hidden_state:
            hidden_states = hidden_states.reshape(target_surface_forms.shape[0], -1)
        else:
            hidden_states = hidden_states[:, 0]

        predicted_embeddings = self.output_projection(hidden_states)

        if self.config.hn_single_head:
            predicted_embeddings_in = predicted_embeddings[..., : self.config.n_embd]

            if self.has_separate_out_embeddings:
                predicted_embeddings_out = predicted_embeddings[
                    ..., self.config.n_embd :
                ]
            else:
                predicted_embeddings_out = None
        else:
            predicted_embeddings_in = predicted_embeddings
            if self.has_separate_out_embeddings:
                predicted_embeddings_out = self.output_projection_out(hidden_states)
            else:
                predicted_embeddings_out = None

        if self.config.hn_rescale_embeddings:
            predicted_embeddings_in = self.scaler(predicted_embeddings_in)

            if predicted_embeddings_out is not None:
                predicted_embeddings_out = self.out_scaler(predicted_embeddings_out)

        if getattr(self.config, "hn_predict_bias", False):
            predicted_bias = self.bias_projection(hidden_states)[..., 0]
        else:
            predicted_bias = jnp.zeros_like(
                target_surface_forms[..., 0], dtype=self.dtype
            )

        return predicted_embeddings_in, predicted_embeddings_out, predicted_bias
