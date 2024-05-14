from transformers import PretrainedConfig

class ZettHypernetConfig(PretrainedConfig):
    def __init__(
        self,
        hn_model_name_or_path: str = "roberta-base",
        hn_surface_maxlen: int = 16,
        hn_n_layers: int = 3,
        n_embd: int = 768,
        hn_hidden_size: int = None,
        hn_intermediate_size: int = None,
        hn_rescale_embeddings: bool = False,
        use_unigram_bias: bool = False,
        hn_embed_target_priors: bool = False,
        hn_add_inter_token_attention: bool = False,
        hn_inter_token_attention_bias_by_priors: bool = False,
        hn_inter_token_attention_bias_scaler: float = 1.0,
        hn_n_inter_token_blocks: int = 16,
        hn_language_adapter_bottleneck_dim: int = 0,
        hn_embed_using_source_embeddings: bool = False,
        hn_concat_last_hidden_state: bool = False,
        hn_single_head: bool = False,
        hn_predict_bias: bool = True,
        hn_num_attention_heads: int = None,
        hn_embed_lang_id: bool = False,
        hn_model_type: str = "roberta",
        n_langs: int = None,  # set in train.py
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model_type = "zett_hypernetwork"
        self.hn_model_name_or_path = hn_model_name_or_path
        self.hn_surface_maxlen = hn_surface_maxlen
        self.hn_n_layers = hn_n_layers
        self.n_embd = n_embd
        self.hn_hidden_size = hn_hidden_size
        self.hn_intermediate_size = hn_intermediate_size
        self.hn_rescale_embeddings = hn_rescale_embeddings
        self.use_unigram_bias = use_unigram_bias
        self.hn_embed_target_priors = hn_embed_target_priors
        self.hn_add_inter_token_attention = hn_add_inter_token_attention
        self.hn_inter_token_attention_bias_by_priors = (
            hn_inter_token_attention_bias_by_priors
        )
        self.hn_inter_token_attention_bias_scaler = hn_inter_token_attention_bias_scaler
        self.hn_n_inter_token_blocks = hn_n_inter_token_blocks
        self.hn_language_adapter_bottleneck_dim = hn_language_adapter_bottleneck_dim
        self.hn_embed_using_source_embeddings = hn_embed_using_source_embeddings
        self.hn_concat_last_hidden_state = hn_concat_last_hidden_state
        self.hn_single_head = hn_single_head
        self.hn_predict_bias = hn_predict_bias
        self.hn_num_attention_heads = hn_num_attention_heads
        self.hn_embed_lang_id = hn_embed_lang_id
        self.hn_model_type = hn_model_type
        self.n_langs = n_langs
