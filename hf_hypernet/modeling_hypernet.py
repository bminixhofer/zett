from .configuration_hypernet import ZettHypernetConfig
from transformers import PreTrainedModel, RobertaConfig, RobertaModel
from functools import partial

from torch import nn as nn
import torch
from torch.nn import functional as F

class Rescaler(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim

        self.w = nn.Parameter(torch.ones((1, self.dim)), requires_grad=False)
        self.b = nn.Parameter(torch.ones((1, self.dim)), requires_grad=False)

    def __call__(self, x):
        return self.w * x + self.b


class ProjectorBlock(nn.Module):
    def __init__(self, input_dim: int, dim: int, intermediate_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.dim = dim
        self.intermediate_dim = intermediate_dim

        self.dense1 = nn.Linear(self.input_dim, self.intermediate_dim)
        self.dense2 = nn.Linear(self.intermediate_dim, self.dim)

        self.ln = nn.LayerNorm(self.dim, eps=1e-6)

    def __call__(self, x):
        h = F.gelu(
            self.dense2(F.gelu(self.dense1(x), approximate="tanh")),
            approximate="tanh",
        )
        return self.ln(h + x)


class ZettHypernet(PreTrainedModel):
    config_class = ZettHypernetConfig

    def __init__(self, config: ZettHypernetConfig):
        super().__init__(config)

        self.config = config
        self.has_separate_out_embeddings = getattr(
            self.config, "separate_out_embeddings", False
        )

        if self.config.hn_embed_lang_id:
            self.lang_embeddings = nn.Embedding(
                self.config.n_langs, self.config.hn_hidden_size
            )

        if self.has_separate_out_embeddings:
            n_in_embd = self.config.n_embd * 2
            n_out_embd = self.config.n_embd
        else:
            n_in_embd = self.config.n_embd
            n_out_embd = self.config.n_embd

        if self.config.hn_model_type == "roberta":
            config = RobertaConfig.from_pretrained(
                self.config.hn_model_name_or_path
            )
            config.num_hidden_layers = self.config.hn_n_layers
            config.hidden_size = self.config.hn_hidden_size
            config.intermediate_size = self.config.hn_intermediate_size
            if getattr(self.config, "hn_num_attention_heads", None) is None:
                self.config.hn_num_attention_heads = self.config.hn_hidden_size // 64
            config.num_attention_heads = self.config.hn_num_attention_heads
            self.embed_init_range = config.initializer_range
            module_class = partial(RobertaModel, add_pooling_layer=False)
        elif self.config.hn_model_type == "t5":
            raise NotImplementedError()

        if self.config.hn_embed_using_source_embeddings:
            # do not need to alloc embeddings since inputs_embeds is always used
            config.vocab_size = self.config.pad_token_id + 1

        if (
            self.config.hn_add_inter_token_attention
            or self.config.hn_embed_target_priors
        ):
            raise NotImplementedError()

        self.pad_token_id = self.config.pad_token_id
        assert self.pad_token_id is not None
        self.model = module_class(config)

        # need at least one embedding
        self.fallback_embeddings = nn.Embedding(
            max(self.config.hn_n_extra_tokens, 1), n_in_embd
        )

        if self.config.hn_embed_using_source_embeddings:
            self.input_projection = nn.Sequential(
                *[
                    nn.Linear(n_in_embd, self.config.hn_hidden_size),
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                    ),
                ]
            )

        if self.config.hn_single_head:
            self.output_projection = nn.Sequential(
                *[
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                    ),
                    nn.Linear(self.config.hn_hidden_size, n_in_embd),
                ]
            )
        else:
            self.output_projection = nn.Sequential(
                *[
                    ProjectorBlock(
                        self.config.hn_hidden_size,
                        self.config.hn_hidden_size,
                        self.config.hn_intermediate_size,
                    ),
                    nn.Linear(self.config.hn_hidden_size, n_out_embd),
                ]
            )
            if self.has_separate_out_embeddings:
                self.output_projection_out = nn.Sequential(
                    *[
                        ProjectorBlock(
                            self.config.hn_hidden_size,
                            self.config.hn_hidden_size,
                            self.config.hn_intermediate_size,
                        ),
                        nn.Linear(self.config.hn_hidden_size, self.config.n_embd),
                    ]
                )

        if self.config.hn_rescale_embeddings:
            self.in_scaler = Rescaler(n_in_embd)
            self.scaler = Rescaler(n_out_embd)

            if self.has_separate_out_embeddings:
                self.out_scaler = Rescaler(self.config.n_embd)

        if getattr(self.config, "hn_predict_bias", False):
            self.bias_projection = nn.Linear(self.config.hn_hidden_size, 1)

    def __call__(
        self,
        target_surface_forms,
        target_priors=None,
        source_embeddings=None,
        lang_index=None,
        deterministic: bool = True,
    ):
        if target_priors is not None:
            raise NotImplementedError()

        if not self.config.hn_embed_using_source_embeddings:
            raise NotImplementedError()

        use_fallback = target_surface_forms >= self.config.original_vocab_size

        main_ids = torch.minimum(
            target_surface_forms, torch.tensor(self.config.original_vocab_size - 1, device=self.device)
        )
        fallback_ids = torch.maximum(
            target_surface_forms - self.config.original_vocab_size, torch.tensor(0, device=self.device)
        )

        source_embeds = F.embedding(main_ids, weight=source_embeddings)

        if self.config.hn_rescale_embeddings:
            source_embeds = self.in_scaler(source_embeds)

        inputs_embeds = torch.where(
            use_fallback[..., None],
            self.fallback_embeddings(fallback_ids),
            source_embeds,
        )
        inputs_embeds = self.input_projection(inputs_embeds)
        attention_mask = target_surface_forms != self.pad_token_id

        if self.config.hn_embed_lang_id:
            lang_embedding = self.lang_embeddings(lang_index).squeeze()
            # position embed and type embed are added afterwards only in PT version so we need to subtract them here
            lang_embedding -= self.model.embeddings.token_type_embeddings(
                torch.tensor(0, device=self.device)
            ) + self.model.embeddings.position_embeddings(
                torch.tensor(attention_mask.shape[1], device=self.device)
            )

            lang_embedding = lang_embedding[None, None, :].expand(
                inputs_embeds.shape[0], -1, -1
            )

            inputs_embeds = torch.cat(
                [
                    inputs_embeds,
                    lang_embedding,
                ],
                axis=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(lang_embedding.shape[:-1], dtype=torch.bool, device=self.device),
                ],
                axis=1,
            )

        position_ids = torch.broadcast_to(
            torch.arange(torch.atleast_2d(attention_mask).shape[-1], device=self.device),
            attention_mask.shape,
        )

        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            predicted_bias = torch.zeros_like(
                target_surface_forms[..., 0], dtype=self.dtype
            )

        return predicted_embeddings_in, predicted_embeddings_out, predicted_bias
