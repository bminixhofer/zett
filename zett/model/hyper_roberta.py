from transformers.models.roberta.modeling_flax_roberta import *
import math
from flax.linen import initializers
from zett.utils import NEGATIVE_INF_FILL_VALUE


class HyperRobertaConfig(RobertaConfig):
    def __init__(
        self,
        embed_target_priors=False,
        add_inter_token_attention=False,
        inter_token_attention_bias_by_priors=False,
        n_inter_token_blocks=16,
        **kwargs,
    ):
        self.embed_target_priors = embed_target_priors
        self.add_inter_token_attention = add_inter_token_attention
        self.inter_token_attention_bias_by_priors = inter_token_attention_bias_by_priors
        self.n_inter_token_blocks = n_inter_token_blocks

        super().__init__(**kwargs)


class FlaxHyperRobertaEmbeddings(FlaxRobertaEmbeddings):
    config: HyperRobertaConfig
    prior_normalization_constant: float = 1.0

    def setup(self):
        super().setup()

        self.target_priors_projection = nn.Dense(
            self.config.hidden_size, dtype=self.dtype
        )
        self.lang_embedding = nn.Embed(
            self.config.n_langs,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        target_priors,
        lang_index,
        attention_mask,
        inputs_embeds=None,
        deterministic: bool = True,
    ):
        # Embed
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

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

        if self.config.embed_lang_id:
            lang_embeddings = jnp.squeeze(self.lang_embedding(lang_index))
            hidden_states = jnp.concatenate(
                [
                    hidden_states,
                    lang_embeddings[None, None, :].repeat(
                        hidden_states.shape[0], axis=0
                    ),
                ],
                axis=1,
            )
            attention_mask = jnp.concatenate(
                [
                    attention_mask,
                    jnp.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype),
                ],
                axis=1,
            )

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states, attention_mask


class FlaxHyperRobertaSelfAttention(FlaxRobertaSelfAttention):
    config: HyperRobertaConfig

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        attention_bias=None,
        key_value_states: Optional[jnp.array] = None,
        init_cache: bool = False,
        deterministic=True,
        output_attentions: bool = False,
    ):
        assert (attention_bias is None) or (attention_mask is None)  # can not have both

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        # get query proj
        query_states = self.query(hidden_states)
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self.key(key_value_states)
            value_states = self.value(key_value_states)
        else:
            # self_attention
            key_states = self.key(hidden_states)
            value_states = self.value(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # handle cache prepare causal attention mask
        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(
                causal_mask, (batch_size,) + causal_mask.shape[1:]
            )

        # combine masks if needed
        if self.causal:
            raise NotImplementedError()

        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if attention_bias is not None:
            attention_bias = jnp.expand_dims(attention_bias, axis=(-3, -2))

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                    self.dtype
                ),
            )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxHyperRobertaAttention(FlaxRobertaAttention):
    config: HyperRobertaConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32
    with_output: bool = True

    def setup(self):
        self.self = FlaxHyperRobertaSelfAttention(
            self.config, causal=self.causal, dtype=self.dtype
        )
        if self.with_output:
            self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        attention_bias=None,
        key_value_states=None,
        init_cache=False,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            attention_bias=attention_bias,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        if self.with_output:
            hidden_states = self.output(
                attn_output, hidden_states, deterministic=deterministic
            )
        else:
            hidden_states = attn_output

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class FlaxHyperRobertaOutput(nn.Module):
    config: HyperRobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype
        )
        self.activation = ACT2FN[self.config.hidden_act]

        if self.config.language_adapter_bottleneck_dim > 0:
            self.lang_adapter_down_weights = self.param(
                "lang_adapter_down_weights",
                jax.nn.initializers.lecun_normal(),
                (
                    self.config.n_langs,
                    self.config.hidden_size,
                    self.config.language_adapter_bottleneck_dim,
                ),
                self.dtype,
            )
            self.lang_adapter_down_bias = self.param(
                "lang_adapter_down_bias",
                jax.nn.initializers.zeros,
                (self.config.n_langs, self.config.language_adapter_bottleneck_dim),
            )
            self.lang_adapter_up_weights = self.param(
                "lang_adapter_up_weights",
                jax.nn.initializers.lecun_normal(),
                (
                    self.config.n_langs,
                    self.config.language_adapter_bottleneck_dim,
                    self.config.hidden_size,
                ),
                self.dtype,
            )
            self.lang_adapter_up_bias = self.param(
                "lang_adapter_up_bias",
                jax.nn.initializers.zeros,
                (self.config.n_langs, self.config.hidden_size),
            )

    def __call__(
        self, hidden_states, attention_output, lang_index, deterministic: bool = True
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)

        if self.config.language_adapter_bottleneck_dim > 0:
            # not very elegant but oh well
            la_down_weights = jnp.take(
                self.lang_adapter_down_weights, lang_index, axis=0
            )
            la_down_bias = jnp.take(self.lang_adapter_down_bias, lang_index, axis=0)
            la_up_weights = jnp.take(self.lang_adapter_up_weights, lang_index, axis=0)
            la_up_bias = jnp.take(self.lang_adapter_up_bias, lang_index, axis=0)

            la_down_output = (
                jnp.einsum("...i,io->...o", hidden_states, la_down_weights)
                + la_down_bias
            )
            la_down_output = self.activation(la_down_output)

            la_up_output = (
                jnp.einsum("...i,io->...o", la_down_output, la_up_weights) + la_up_bias
            )
            hidden_states = self.LayerNorm(la_up_output + hidden_states)

        return hidden_states


class FlaxHyperRobertaLayer(FlaxRobertaLayer):
    config: HyperRobertaConfig

    def setup(self):
        self.attention = FlaxHyperRobertaAttention(
            self.config, causal=self.config.is_decoder, dtype=self.dtype
        )
        self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxHyperRobertaOutput(self.config, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxHyperRobertaAttention(
                self.config, causal=False, dtype=self.dtype
            )
        if self.config.add_inter_token_attention:
            self.inter_token_attention_down = FlaxHyperRobertaAttention(
                self.config, causal=False, dtype=self.dtype
            )
            self.inter_token_attention_up = FlaxHyperRobertaAttention(
                self.config, causal=False, dtype=self.dtype
            )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        target_priors,
        inter_token_blocks,
        lang_index,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # Self Attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]  # (n_tokens x seq_length x hidden_size)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]

        # Inter-Token Attention Block
        if self.config.add_inter_token_attention:
            if self.config.inter_token_attention_bias_by_priors:
                attention_bias = (
                    target_priors[None] * self.config.inter_token_attention_bias_scaler
                )
            else:
                attention_bias = None

            blocks = self.inter_token_attention_down(
                inter_token_blocks[None],
                attention_mask=None,
                layer_head_mask=None,
                attention_bias=attention_bias,
                key_value_states=attention_output[
                    None, :, 0
                ],  # (1 x n_tokens x hidden_size)
                deterministic=deterministic,
            )[0][0]
            attention_output = self.inter_token_attention_up(
                attention_output.reshape(1, -1, self.config.hidden_size),
                attention_mask=None,
                layer_head_mask=None,
                key_value_states=blocks[None],
                deterministic=deterministic,
            )[0].reshape(attention_output.shape)

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(
            hidden_states, attention_output, lang_index, deterministic=deterministic
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)

        outputs += (inter_token_blocks,)

        return outputs


class FlaxHyperRobertaLayerCollection(FlaxRobertaLayerCollection):
    config: HyperRobertaConfig

    def setup(self):
        self.inter_token_blocks = self.param(
            "inter_token_blocks",
            jax.nn.initializers.normal(stddev=self.config.initializer_range),
            (self.config.n_inter_token_blocks, self.config.hidden_size),
            self.dtype,
        )

        if self.gradient_checkpointing:
            FlaxRobertaCheckpointLayer = remat(
                FlaxHyperRobertaLayer, static_argnums=(8, 9, 10)
            )
            self.layers = [
                FlaxRobertaCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            self.layers = [
                FlaxHyperRobertaLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        target_priors,
        lang_index,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )

        # Check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                    f"       {head_mask.shape[0]}."
                )

        inter_token_blocks = (
            self.inter_token_blocks if self.config.add_inter_token_attention else None
        )

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                target_priors,
                inter_token_blocks,
                lang_index,
                encoder_hidden_states,
                encoder_attention_mask,
                init_cache,
                deterministic,
                output_attentions,
            )
            if self.config.add_inter_token_attention:
                inter_token_blocks = layer_outputs[-1]

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (
            hidden_states,
            all_hidden_states,
            all_attentions,
            all_cross_attentions,
        )

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class FlaxHyperRobertaEncoder(FlaxRobertaEncoder):
    config: HyperRobertaConfig

    def setup(self):
        self.layer = FlaxHyperRobertaLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        target_priors,
        lang_index,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            target_priors=target_priors,
            lang_index=lang_index,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxHyperRobertaModule(nn.Module):
    config: HyperRobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = FlaxHyperRobertaEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxHyperRobertaEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.pooler = FlaxRobertaPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: Optional[jnp.ndarray] = None,
        lang_index: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        target_priors: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if target_priors is not None:
            target_priors = target_priors.astype("f4")

        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
            )

        hidden_states, attention_mask = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            target_priors=target_priors,
            lang_index=lang_index,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            deterministic=deterministic,
        )
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            target_priors=target_priors,
            lang_index=lang_index,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
