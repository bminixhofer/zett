from typing import List
import numpy as np
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import tokenizers
from tokenizers import models, pre_tokenizers, decoders, normalizers, Tokenizer
from collections import Counter

import rust_utils

from zett.utils import (
    SPLIT_REGEX,
    MAX_CHARS_PER_TOKEN,
    NEGATIVE_INF_FILL_VALUE,
    get_surface_form_matrix,
    unset_tokenizer_special_tokens,
    default_pretokenize,
    CHARS_TO_BYTES,
    BYTES_TO_CHARS,
)
from zett.tokenizer_converters import convert_to_byte_level

from tqdm.auto import tqdm


class Collator:
    def __init__(
        self,
        reference,
        hn_tokenizer,
        data_args,
        batch_size=None,
        tokenizer_name=None,
        initial_texts=None,
        lang_code=None,
        inner_collator=None,
        is_validation=False,
        with_consistent_whitespace=True,
    ):
        self.tokenizer_name = tokenizer_name
        self.reference = reference
        self.hn_tokenizer = hn_tokenizer
        self.data_args = data_args
        self.batch_size = batch_size
        self.lang_code = lang_code
        self.inner_collator = inner_collator
        self.is_validation = is_validation
        self.with_consistent_whitespace = with_consistent_whitespace

        assert (tokenizer_name is None) == self.data_args.do_tokenizer_sampling

        if not data_args.do_tokenizer_sampling:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.original_length = len(tokenizer)

            # keep original in case of passthrough hypernet
            if not self.data_args.use_passthrough_hypernet:
                tokenizer = convert_to_byte_level(
                    tokenizer,
                    match_special_tokens_to=reference,
                    make_whitespace_consistent=self.with_consistent_whitespace,
                )[0]

                # assume consistent pretokenizer
                tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                    use_regex=True, add_prefix_space=self.data_args.add_prefix_space
                )
                tokenizer._tokenizer.decoder = decoders.ByteLevel()
            else:
                # gpt2 / llama
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

            if (
                hn_tokenizer is None
                or hn_tokenizer.get_vocab() == tokenizer.get_vocab()
            ):
                # in this case, we do not actually compute surface_forms - just use np.arange
                # note that this is not exactly equivalent to actually computing the surface form matrix
                # since tokenizer(token) need not be one token (even if the token is in the vocab)
                self.surface_forms = np.arange(len(tokenizer))[:, None]
            else:
                self.surface_forms, n_truncated = get_surface_form_matrix(
                    tokenizer,
                    data_args.hn_surface_maxlen,
                    hn_tokenizer,
                )
                print(
                    f"Truncated {n_truncated} surface forms (length={data_args.hn_surface_maxlen})."
                )

            self.tokenizer = tokenizer

            if hasattr(tokenizer._tokenizer.model, "get_scores"):
                scores = tokenizer._tokenizer.model.get_scores()
                while len(scores) < len(tokenizer):
                    scores.append(0.0)
                self.scores = np.array(scores)
            else:
                self.scores = np.zeros(len(tokenizer))

            # careful: assumes byte pretokenization
            all_tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
            self.byte_lengths = np.array([len(x) for x in all_tokens])

            self.inv_ids_to_embed = (
                np.zeros(len(tokenizer), dtype=np.int32)
                if self.data_args.n_token_subsample is not None
                else None
            )
        else:
            self.inv_ids_to_embed = (
                np.zeros(self.data_args.tokenizer_sample_max + 256, dtype=np.int32)
                if self.data_args.n_token_subsample is not None
                else None
            )

        self.samplers = {}

        if initial_texts is not None:
            for lang in initial_texts.keys():
                texts = []
                max_length = MAX_CHARS_PER_TOKEN * self.data_args.block_size

                for text in initial_texts[lang]:
                    if self.data_args.sample_text_span:
                        start = np.random.randint(0, max(len(text) - max_length, 0) + 1)
                    else:
                        start = 0

                    end = start + max_length
                    texts.append(text[start:end])

                samplers = []

                for _ in range(self.data_args.n_pools):
                    sampler = rust_utils.TokenizerSampler()

                    for start in tqdm(range(0, len(texts), self.batch_size)):
                        end = start + self.batch_size

                        sampler.sample_tokenizer(
                            {text: 1 for text in texts[start:end]},
                            30_000,
                            16,
                            4,
                            0.0,
                            False,
                        )

                    samplers.append(sampler)

                self.samplers[lang] = samplers

    def encode(
        self,
        tokenizer,
        texts,
        target_surface_form_matrix_to_use,
        target_priors_to_use,
        special_ids_map=None,
        metrics_data=None,
    ):
        assert len(target_priors_to_use) == len(target_surface_form_matrix_to_use)

        encodings = dict(
            tokenizer(
                texts,
                max_length=self.data_args.block_size,
                truncation=True,
                padding="max_length",
                return_tensors="np",
                add_special_tokens=True,
            )
        )

        for key, value in (special_ids_map or {}).items():
            encodings["input_ids"][encodings["input_ids"] == key] = value

        if self.inner_collator is not None:
            updates = self.inner_collator(tokenizer, return_tensors="np")(
                encodings["input_ids"]
            )
            encodings.update(updates)
        else:
            # clm
            encodings["labels"] = encodings["input_ids"].copy()

        input_ids = encodings["input_ids"]

        positive_indices, positive_counts = np.unique(input_ids, return_counts=True)

        if metrics_data is not None:
            byte_lengths_per_token = metrics_data[0]

            non_special_tokens_mask = np.isin(
                input_ids, tokenizer.all_special_ids, invert=True
            )
            byte_lengths = byte_lengths_per_token[input_ids]

            encodings["metrics"] = {
                "avg_byte_length": byte_lengths[non_special_tokens_mask].mean(),
                "unk_ratio": (input_ids == tokenizer.unk_token_id).mean(),
            }
            encodings["byte_lengths"] = byte_lengths

        if self.data_args.n_token_subsample is not None:
            assert (
                self.data_args.n_token_subsample % self.data_args.pad_to_multiple_of
                == 0
            )

            tokens_in_batch = np.concatenate(
                [
                    np.array(tokenizer.all_special_ids),
                    np.setdiff1d(
                        np.unique(np.concatenate([input_ids, encodings["labels"]])),
                        np.array(tokenizer.all_special_ids),
                    ),
                ]
            )
            assert len(tokens_in_batch) <= self.data_args.n_token_subsample

            if self.data_args.subsample_mode == "positives_only":
                negatives_to_embed = np.zeros(
                    self.data_args.n_token_subsample - len(tokens_in_batch),
                    dtype=np.int32,
                )
            elif self.data_args.subsample_mode == "random":
                # random sampling makes sense because the tokenizer is already sampled according to unigram probabilities
                # so if we do unigram sampling again here we would have a sort of "squared sampling"
                negatives_to_embed = np.setdiff1d(
                    np.arange(len(tokenizer)), positive_indices
                )
                assert len(
                    negatives_to_embed
                ) >= self.data_args.n_token_subsample - len(tokens_in_batch)
                np.random.shuffle(negatives_to_embed)
                negatives_to_embed = negatives_to_embed[
                    : self.data_args.n_token_subsample - len(tokens_in_batch)
                ]
            elif self.data_args.subsample_mode == "highest_scores":
                raise NotImplementedError()

            ids_to_embed = np.concatenate([tokens_in_batch, negatives_to_embed])
            ids_to_embed_list = list(ids_to_embed)

            # try to preserve special token indices
            # because e.g. the model might have a hardcoded padding id for which the embedding is ignored
            # we can't always preserve it because special tokens may be at the end of the vocabulary (e.g. GPT2 <endoftext> token)
            for special_token in sorted(tokenizer.all_special_ids):
                del ids_to_embed_list[ids_to_embed_list.index(special_token)]

                ids_to_embed_list.insert(special_token, special_token)

            ids_to_embed = np.array(ids_to_embed_list)

            self.inv_ids_to_embed[ids_to_embed] = np.arange(len(ids_to_embed))
            encodings["input_ids"] = self.inv_ids_to_embed[encodings["input_ids"]]

            active_labels = encodings["labels"] != -100
            encodings["labels"] = np.where(
                active_labels, self.inv_ids_to_embed[encodings["labels"]], -100
            )

            encodings["target_priors"] = target_priors_to_use[ids_to_embed]
            encodings["target_surface_forms"] = target_surface_form_matrix_to_use[
                ids_to_embed
            ]
            encodings["mask"] = np.ones(len(ids_to_embed), dtype=bool)
            encodings["ids_to_embed"] = ids_to_embed

            assert tokenizer.all_special_tokens == self.reference.all_special_tokens
            encodings["special_indices"] = np.array(
                [ids_to_embed_list.index(x) for x in tokenizer.all_special_ids]
            )
            encodings["special_indices_in_reference"] = np.array(
                [
                    self.reference.convert_tokens_to_ids(token)
                    for token in tokenizer.all_special_tokens
                ]
            )
        else:
            length = len(target_priors_to_use)
            if self.data_args.do_tokenizer_sampling:
                # need consistent size
                # + pad_to_multiple_of to take into account potential special tokens
                assert (
                    self.data_args.tokenizer_sample_max
                    % self.data_args.pad_to_multiple_of
                    == 0
                )
                n_pad = (
                    self.data_args.tokenizer_sample_max
                    + self.data_args.pad_to_multiple_of
                    - length
                )
            elif length % self.data_args.pad_to_multiple_of != 0:
                n_pad = self.data_args.pad_to_multiple_of - (
                    length % self.data_args.pad_to_multiple_of
                )
            else:
                n_pad = 0

            target_priors_to_use = np.pad(
                target_priors_to_use,
                (0, n_pad),
                constant_values=NEGATIVE_INF_FILL_VALUE,
            )
            target_surface_form_matrix_to_use = np.pad(
                target_surface_form_matrix_to_use,
                ((0, n_pad), (0, 0)),
                constant_values=0,
            )

            encodings["target_priors"] = target_priors_to_use
            encodings["target_surface_forms"] = target_surface_form_matrix_to_use
            encodings["mask"] = np.concatenate(
                [
                    np.ones(length, dtype=bool),
                    np.zeros(n_pad, dtype=bool),
                ]
            )
            encodings["ids_to_embed"] = np.concatenate(
                [
                    np.arange(length),
                    np.zeros(n_pad, dtype=np.int32),
                ]
            )
            assert tokenizer.all_special_tokens == self.reference.all_special_tokens
            encodings["special_indices"] = np.array(tokenizer.all_special_ids)
            encodings["special_indices_in_reference"] = np.array(
                [
                    self.reference.convert_tokens_to_ids(token)
                    for token in tokenizer.all_special_tokens
                ]
            )

        return encodings

    def sample_tokenizer(self, texts, sampler):
        n_total = int(
            np.random.normal(
                self.data_args.tokenizer_sample_mean,
                self.data_args.tokenizer_sample_std,
            )
        )
        n_total = max(self.data_args.tokenizer_sample_min, n_total)
        n_total = min(self.data_args.tokenizer_sample_max, n_total)

        pretoken_counts = {}
        for text in texts:
            pretoken_counts[text] = 1

        if self.data_args.tokenizer_noise_mean > 0:
            noise_std = np.random.lognormal(
                mean=np.log(self.data_args.tokenizer_noise_mean),
                sigma=self.data_args.tokenizer_noise_std,
            )
        else:
            noise_std = 0

        pieces, scores = zip(
            *sampler.sample_tokenizer(
                pretoken_counts, n_total, 16, 4, noise_std, True, not self.is_validation
            )
        )
        pieces = list(pieces)
        scores = list(scores)

        piece_set = set(pieces)

        unknown_chars = set(CHARS_TO_BYTES.keys()) - piece_set
        min_score = min(scores)
        pieces = sorted(unknown_chars) + pieces
        scores = [min_score] * len(unknown_chars) + scores

        special_tokens_to_remove = set(self.reference.all_special_tokens).intersection(
            piece_set
        )
        for token in special_tokens_to_remove:
            idx = pieces.index(token)
            pieces.remove(token)
            scores.pop(idx)

        special_ids_map = {}

        for i in np.argsort(self.reference.all_special_ids):
            pieces.insert(
                self.reference.all_special_ids[i], self.reference.all_special_tokens[i]
            )
            scores.insert(self.reference.all_special_ids[i], 0.0)

            if (
                pieces.index(self.reference.all_special_tokens[i])
                != self.reference.all_special_ids[i]
            ):
                special_ids_map[self.reference.all_special_ids[i]] = pieces.index(
                    self.reference.all_special_tokens[i]
                )

        scores = np.array(scores)

        tokenizer = Tokenizer(models.Unigram([(piece, score) for piece, score in zip(pieces, scores)]))
        if self.data_args.add_prefix_space:
            tokenizer.normalizer = normalizers.Prepend(" ")

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(tokenizers.Regex(SPLIT_REGEX), "removed", invert=True),
            pre_tokenizers.ByteLevel(False, False),
        ])
        tokenizer.decoder = decoders.ByteLevel()

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, clean_up_tokenization_spaces=False
        )

        if self.reference._tokenizer.post_processor is not None:
            tokenizer._tokenizer.post_processor = (
                self.reference._tokenizer.post_processor
            )

        tokenizer.eos_token = self.reference.eos_token
        tokenizer.pad_token = self.reference.pad_token
        tokenizer.sep_token = self.reference.sep_token
        tokenizer.unk_token = self.reference.unk_token
        tokenizer.bos_token = self.reference.bos_token
        tokenizer.cls_token = self.reference.cls_token
        tokenizer.mask_token = self.reference.mask_token
        tokenizer.unk_token = self.reference.unk_token

        tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
        byte_lengths = np.array([len(token) for token in tokens])

        if self.hn_tokenizer is not None:
            target_surface_form_matrix_to_use = get_surface_form_matrix(
                tokens,
                self.data_args.hn_surface_maxlen,
                self.hn_tokenizer,
                verbose=False,
            )[0]
        else:
            target_surface_form_matrix_to_use = None
        target_priors_to_use = scores

        return (
            tokenizer,
            special_ids_map,
            target_surface_form_matrix_to_use,
            target_priors_to_use,
            byte_lengths,
        )

    def __call__(self, data, for_identity_step=False):
        if for_identity_step:
            lang_code = self.lang_code

            # choose random uniform
            indices = np.random.choice(
                self.original_length,
                size=self.data_args.n_token_subsample,
                replace=False,
            )
            target_surface_form_matrix_to_use = self.surface_forms[indices]

            return {
                "target_surface_forms": target_surface_form_matrix_to_use,
                "target_priors": np.zeros(len(indices), dtype=np.float32),
                "ids_to_embed": indices,
                "lang_code": lang_code,
                "lang_index": np.array(
                    self.data_args.langs.index(lang_code)
                    if lang_code is not None
                    else 0
                ),
            }

        # `data` is either:
        #  - a list of dicts (e.g. in case of load_dataset("parquet", ..))
        # - a list containing a single dict of {"texts": [..], "lang_code": "xx"} in case of TrainDataset and ValidDataset
        if "texts" in data[0]:
            examples = [{"text": text} for text in data[0]["texts"]]
            lang_code = data[0]["lang_code"]
        else:
            examples = data
            lang_code = None

        if self.lang_code is not None:
            lang_code = self.lang_code  # overrides

        texts = []
        max_length = MAX_CHARS_PER_TOKEN * self.data_args.block_size

        for e in examples:
            if self.data_args.sample_text_span:
                start = np.random.randint(0, max(len(e["text"]) - max_length, 0) + 1)
            else:
                start = 0

            end = start + max_length
            texts.append(e["text"][start:end])

        if self.data_args.do_tokenizer_sampling:  # <-> one tokenizer for each language
            samplers = self.samplers[lang_code]
            sampler_idx = np.random.randint(0, len(samplers))
            sampler = samplers[sampler_idx]

            (
                tokenizer,
                special_ids_map,
                target_surface_form_matrix_to_use,
                target_priors_to_use,
                byte_lengths,
            ) = self.sample_tokenizer(texts, sampler)
        else:
            tokenizer = self.tokenizer
            special_ids_map = {}

            target_surface_form_matrix_to_use = self.surface_forms
            target_priors_to_use = self.scores
            byte_lengths = self.byte_lengths

        encodings = self.encode(
            tokenizer,
            texts,
            target_surface_form_matrix_to_use,
            target_priors_to_use,
            metrics_data=(byte_lengths,),
            special_ids_map=special_ids_map,
        )

        encodings["lang_code"] = lang_code
        encodings["lang_index"] = np.array(
            self.data_args.langs.index(lang_code) if lang_code is not None else 0
        )

        return encodings
