from transformers.models.t5.modeling_flax_t5 import *
import math
from zett.utils import NEGATIVE_INF_FILL_VALUE


class HyperT5Config(T5Config):
    def __init__(
        self,
        embed_target_priors=False,
        add_inter_token_attention=False,
        inter_token_attention_bias_by_priors=False,
        n_inter_token_blocks=16,
        **kwargs
    ):
        self.embed_target_priors = embed_target_priors
        self.add_inter_token_attention = add_inter_token_attention
        self.inter_token_attention_bias_by_priors = inter_token_attention_bias_by_priors
        self.n_inter_token_blocks = n_inter_token_blocks

        super().__init__(**kwargs)


class FlaxHyperT5Block(nn.Module):
    config: HyperT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.causal = self.config.causal
        self.layer = (
            FlaxT5LayerSelfAttention(
                self.config,
                has_relative_attention_bias=self.has_relative_attention_bias,
                name=str(0),
                dtype=self.dtype,
            ),
        )
        feed_forward_index = 1
        if self.causal:
            self.layer += (
                FlaxT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),
            )
            feed_forward_index += 1

        self.layer += (
            FlaxT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),
        )

        if self.config.add_inter_token_attention:
            self.inter_token_layer_norm1 = FlaxT5LayerNorm(
                self.config.d_model,
                eps=self.config.layer_norm_epsilon,
                dtype=self.dtype,
            )
            self.inter_token_layer_norm2 = FlaxT5LayerNorm(
                self.config.d_model,
                eps=self.config.layer_norm_epsilon,
                dtype=self.dtype,
            )
            self.inter_token_attention_down = FlaxT5Attention(
                self.config, causal=False, dtype=self.dtype
            )
            self.inter_token_attention_up = FlaxT5Attention(
                self.config, causal=False, dtype=self.dtype
            )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        target_priors=None,
        inter_token_blocks=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        return_dict=True,
        deterministic=True,
        init_cache=False,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[
            1:
        ]  # Keep self-attention outputs and relative position weights

        do_cross_attention = self.causal and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            hidden_states = cross_attention_outputs[0]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Inter-Token Attention Block
        if self.config.add_inter_token_attention:
            if self.config.inter_token_attention_bias_by_priors:
                attention_bias = (
                    target_priors[None] * self.config.inter_token_attention_bias_scaler
                )
            else:
                attention_bias = None

            normed_hidden_states = self.inter_token_layer_norm1(hidden_states)

            blocks = self.inter_token_attention_down(
                inter_token_blocks[None],
                attention_mask=None,
                position_bias=attention_bias,
                key_value_states=normed_hidden_states[
                    None, :, 0
                ],  # (1 x n_tokens x hidden_size)
                deterministic=deterministic,
            )[0][0]
            blocks = self.inter_token_layer_norm2(inter_token_blocks + blocks)

            inter_token_attn_out = self.inter_token_attention_up(
                normed_hidden_states[None, :, 0],
                attention_mask=None,
                key_value_states=blocks[None],
                deterministic=deterministic,
            )
            #  inter_token_attn_out[0][0] shape: n_batch x hidden_size
            hidden_states = hidden_states + inter_token_attn_out[0][0][:, None, :]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        outputs = outputs + attention_outputs

        # returns hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        return outputs


class FlaxHyperT5LayerCollection(FlaxT5LayerCollection):
    config: HyperT5Config
    has_relative_attention_bias: bool
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxHyperT5Block(
            self.config,
            has_relative_attention_bias=self.has_relative_attention_bias,
            dtype=self.dtype,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        target_priors=None,
        inter_token_blocks=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        deterministic=True,
        init_cache=False,
    ):
        return self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            target_priors=target_priors,
            inter_token_blocks=inter_token_blocks,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
        )


class FlaxHyperT5BlockCollection(FlaxT5BlockCollection):
    config: HyperT5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        if self.config.add_inter_token_attention:
            self.inter_token_blocks = self.param(
                "inter_token_blocks",
                jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
                (self.config.n_inter_token_blocks, self.config.d_model),
                self.dtype,
            )

        self.causal = self.config.causal
        if self.gradient_checkpointing:
            FlaxT5CheckpointLayer = remat(
                FlaxHyperT5LayerCollection, static_argnums=(8, 9, 10)
            )
            self.blocks = [
                FlaxT5CheckpointLayer(
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]
        else:
            self.blocks = [
                FlaxHyperT5LayerCollection(
                    self.config,
                    has_relative_attention_bias=(i == 0),
                    dtype=self.dtype,
                    name=str(i),
                )
                for i in range(self.config.num_layers)
            ]

    def __call__(
        self,
        hidden_states=None,
        attention_mask=None,
        target_priors=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        # Prepare head mask if needed
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.causal) else None
        position_bias = None
        encoder_decoder_position_bias = None

        inter_token_blocks = (
            self.inter_token_blocks if self.config.add_inter_token_attention else None
        )

        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                target_priors,
                inter_token_blocks,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                output_attentions,
                deterministic,
                init_cache,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]

            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    3 if output_attentions else 2
                ]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class FlaxHyperT5Stack(FlaxT5Stack):
    config: HyperT5Config
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False
    prior_normalization_constant: float = math.log(1e-12)

    def setup(self):
        self.causal = self.config.causal

        self.block = FlaxHyperT5BlockCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.final_layer_norm = FlaxT5LayerNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.target_priors_projection = nn.Dense(self.config.d_model, dtype=self.dtype)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        target_priors=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)

        if self.config.embed_target_priors:
            # append an extra token embedding the target priors
            target_priors_embeddings = self.target_priors_projection(
                target_priors[..., None] / self.prior_normalization_constant
            )
            hidden_states = jnp.concatenate(
                [hidden_states, target_priors_embeddings[:, None, :]], axis=1
            )
            attention_mask = jnp.concatenate(
                [
                    attention_mask,
                    jnp.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype),
                ],
                axis=1,
            )

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            target_priors=target_priors,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )

        hidden_states = outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # Add last layer
        all_hidden_states = None

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxHyperT5EncoderModule(FlaxT5EncoderModule):
    config: HyperT5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(
                self.config.initializer_factor * 1.0
            ),
            dtype=self.dtype,
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.causal = False
        self.encoder = FlaxHyperT5Stack(
            encoder_config,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        target_priors=None,
        lang_index=None,  # not implemented
        output_attentions=False,
        output_hidden_states=False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        target_priors = target_priors.astype("f4")

        # Encode if needed (training, first prediction pass)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_priors=target_priors,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        return encoder_outputs


class FlaxHyperT5EncoderModel(FlaxT5PreTrainedModel):
    module_class = FlaxHyperT5EncoderModule
