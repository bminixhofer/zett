import numpy as np
import copy
from tempfile import NamedTemporaryFile
import json
from transformers import AutoTokenizer

from zett.utils import SPLIT_REGEX, BYTES_TO_CHARS, CHARS_TO_BYTES, NEGATIVE_INF_FILL_VALUE
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
)


def fix_postprocessor_data(data, surface_forms):
    if data["type"] == "TemplateProcessing":
        for k in data["special_tokens"].keys():
            tokens = data["special_tokens"][k]["tokens"]
            ids = [surface_forms.index(t) for t in tokens]
            data["special_tokens"][k]["ids"] = ids
    elif data["type"] == "RobertaProcessing":
        data["sep"][1] = surface_forms.index(data["sep"][0])
        data["cls"][1] = surface_forms.index(data["cls"][0])
    elif data["type"] == "Sequence":
        for postprocessor in data["processors"]:
            fix_postprocessor_data(postprocessor, surface_forms)


def is_byte_level(tokenizer, tokenizer_data):
    return isinstance(tokenizer._tokenizer.pre_tokenizer, pre_tokenizers.ByteLevel) or (
        (tokenizer_data.get("pre_tokenizer") or {}).get("type") == "Sequence"
        and any(
            pretok["type"] == "ByteLevel"
            for pretok in tokenizer_data["pre_tokenizer"]["pretokenizers"]
        )
    )


def get_byte_fn(tokenizer, tokenizer_data):
    # needed e.g. for bigcode/starcoder
    # which has [Digits, ByteLevel] sequence of pretokenizers
    if is_byte_level(tokenizer, tokenizer_data):
        assert len(tokenizer_data["model"].get("continuing_subword_prefix") or "") == 0
        return lambda x: x, None

    def normalize_function(x):
        if tokenizer._tokenizer.normalizer is not None:
            x = tokenizer._tokenizer.normalizer.normalize_str(x)

        if tokenizer._tokenizer.pre_tokenizer is not None:
            x = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(x)[0][0]

        return x

    normalized = normalize_function(" test")
    if normalized[0] != " " and normalized != "test":
        meta_char = normalized[0]
    else:
        meta_char = None

    continuing_subword_prefix = tokenizer_data["model"].get("continuing_subword_prefix")

    def to_byte_fn(token):
        if meta_char is not None:
            token = token.replace(meta_char, " ")
        if continuing_subword_prefix is not None:
            if token.startswith(continuing_subword_prefix):
                token = token[len(continuing_subword_prefix) :]
            else:
                token = " " + token

        return "".join(BYTES_TO_CHARS[b] for b in token.encode("utf-8"))

    return to_byte_fn, continuing_subword_prefix


def convert_to_byte_level(
    tokenizer,
    keep_normalizer=False,
    keep_pretokenizer=False,
    make_whitespace_consistent=False,
    match_special_tokens_to=None,
):
    f = NamedTemporaryFile()

    if match_special_tokens_to is not None:
        match_special_tokens_to._tokenizer.save(f.name)
        match_special_tokens_to_data = json.load(open(f.name))
    else:
        match_special_tokens_to_data = {}

    tokenizer._tokenizer.save(f.name)
    tokenizer_data = json.load(open(f.name))
    if "added_tokens" in tokenizer_data:
        # will be added to the vocab
        del tokenizer_data["added_tokens"]

    original_tokenizer_data = copy.deepcopy(tokenizer_data)
    preserved_original_token_indices = True

    original_length = len(tokenizer)

    to_byte_fn, continuing_subword_prefix = get_byte_fn(tokenizer, tokenizer_data)
    is_already_byte_level = is_byte_level(tokenizer, tokenizer_data)

    if continuing_subword_prefix is not None:
        tokenizer_data["model"]["continuing_subword_prefix"] = ""

    raw_tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
    surface_forms = [
        to_byte_fn(token) if token not in tokenizer.all_special_tokens else token
        for token in raw_tokens
    ]

    if tokenizer_data["model"].get("byte_fallback"):
        byte_fallback_to_byte = {f"<0x{i:02X}>": BYTES_TO_CHARS[i] for i in range(255)}

        surface_form_set = set(surface_forms)
        for i in range(len(surface_forms)):
            if surface_forms[i] in byte_fallback_to_byte:
                byte = byte_fallback_to_byte[surface_forms[i]]

                if byte not in surface_form_set:
                    surface_forms[i] = byte_fallback_to_byte[surface_forms[i]]
    else:
        byte_fallback_to_byte = {}

    fill_bytes = [x for x in CHARS_TO_BYTES.keys() if x not in surface_forms]

    if len(fill_bytes) > 0:
        print(f"WARNING: {len(fill_bytes)} bytes not in surface forms.")
        surface_forms += fill_bytes

    if make_whitespace_consistent:
        extra_whitespace = ["Ġ", "Ċ", "ĉ"]
        allowed_whitespace_tokens = []

        for c1 in extra_whitespace:
            for i in range(1, 16):
                for c2 in extra_whitespace:
                    allowed_whitespace_tokens.append(c2 + c1 * i)

        for i in range(len(surface_forms)):
            if surface_forms[i] in allowed_whitespace_tokens:
                # already present, no need to change
                allowed_whitespace_tokens.remove(surface_forms[i])
            elif sum(c in extra_whitespace for c in surface_forms[i]) > 1 or len(surface_forms[i].strip()) == 0:
                 # ^ len(x.strip()) in case there are added tokens consisting of whitespace (e.g. GPT-NeoX)
                surface_forms[i] = f"<unused_whitespace__{i}>"

        for token in allowed_whitespace_tokens:
            surface_forms.append(token)

    if match_special_tokens_to is not None:
        surface_forms = [
            s
            for s in surface_forms
            if s not in tokenizer.all_special_tokens
            and s not in match_special_tokens_to.all_special_tokens
        ]

        for i in np.argsort(match_special_tokens_to.all_special_ids):
            surface_forms.insert(
                match_special_tokens_to.all_special_ids[i],
                match_special_tokens_to.all_special_tokens[i],
            )

        special_tokens = match_special_tokens_to.all_special_tokens
        preserved_original_token_indices = False
    else:
        special_tokens = tokenizer.all_special_tokens

    new_normalizer_data = {
        "type": "Prepend",
        "prepend": " "
    }

    new_pretokenizer_data = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {
                    "Regex": SPLIT_REGEX
                },
                "behavior": "Removed",
                "invert": True,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": False,
            }
        ]
    }

    if not keep_normalizer:
        # this is problematic if it was doing e.g. NFKC but we can't keep the metaspace normalizer
        tokenizer_data["normalizer"] = new_normalizer_data
    else:
        previous_normalizer = tokenizer_data.get("normalizer")

        tokenizer_data["normalizer"] = {
            "type": "Sequence",
            "normalizers": [
                new_normalizer_data
            ]
        }

        if previous_normalizer is not None:
            tokenizer_data["normalizer"]["normalizers"].insert(0, previous_normalizer)

    if not keep_pretokenizer:
        tokenizer_data["pre_tokenizer"] = new_pretokenizer_data
    else:
        if not is_already_byte_level:
            previous_pretokenizer = tokenizer_data.get("pre_tokenizer")
            new_pretokenizer_data[
                "use_regex"
            ] = False  # TODO: clarify this / make an argument?

            tokenizer_data["pre_tokenizer"] = {
                "type": "Sequence",
                "pretokenizers": [
                    new_pretokenizer_data,
                ],
            }

            if previous_pretokenizer is not None:
                tokenizer_data["pre_tokenizer"]["pretokenizers"].insert(
                    0, previous_pretokenizer
                )

    if isinstance(tokenizer._tokenizer.model, models.Unigram):
        score_dict = {
            to_byte_fn(s): score
            for s, score in original_tokenizer_data["model"]["vocab"]
        }
        for char in CHARS_TO_BYTES.keys():
            if char not in score_dict:
                # avoid fill bytes
                score_dict[char] = NEGATIVE_INF_FILL_VALUE

        if make_whitespace_consistent:
            for key in list(score_dict.keys()):
                if sum(c in extra_whitespace for c in key) > 1:
                    del score_dict[key]

        tokenizer_data["model"]["vocab"] = [
            # TODO: 0.0 might be problematic for whitespace tokens
            (surface_forms[i], score_dict.get(surface_forms[i], 0.0))
            for i in range(len(surface_forms))
        ]
    elif isinstance(tokenizer._tokenizer.model, models.BPE):
        surface_forms_set = set(surface_forms)

        inv_merges = {}
        merges = []

        for merge in tokenizer_data["model"]["merges"]:
            if isinstance(merge, str):
                x, y = merge.split(" ")
            else:
                x, y = merge
            x = to_byte_fn(x)
            y = to_byte_fn(y)

            z = x + y

            if make_whitespace_consistent and sum(c in extra_whitespace for c in z) > 1:
                continue

            if z not in inv_merges:
                inv_merges[z] = []

            inv_merges[z].append((x, y))
            merges.append(f"{x} {y}")

        def decompose(token):
            decompositions = {token}

            done = False
            while not done:
                done = True

                for d in copy.deepcopy(decompositions):
                    merges = inv_merges.get(d)
                    if merges is not None:
                        for merge in merges:
                            decompositions.update(merge)

                        decompositions.remove(d)
                        done = False
                        break

            return decompositions

        def get_merges(token):
            merges = []
            vocab = set()
            atoms = [c for c in token]

            while len(atoms) > 1:
                frozen_atoms = copy.deepcopy(atoms)
                for c1, c2 in zip(frozen_atoms, frozen_atoms[1:]):
                    # apply merge
                    applied_merge = False
                    i = 0
                    while i < len(atoms) - 1:
                        if atoms[i] == c1 and atoms[i + 1] == c2:
                            atoms[i] = c1 + c2
                            del atoms[i + 1]
                            applied_merge = True
                        i += 1

                    if applied_merge:
                        merges.append(f"{c1} {c2}")
                        if c1 + c2 not in surface_forms_set:
                            vocab.add(c1 + c2)

            return merges, vocab

        all_extra_merges_set = set()
        all_extra_pre_merges = []
        all_extra_post_merges = []
        all_extra_vocab = set()

        problematic_decompostions = set()

        if is_already_byte_level:
            surface_forms_to_check = surface_forms[original_length:]
        else:
            surface_forms_to_check = surface_forms

        for token in surface_forms_to_check:
            if (
                token in special_tokens
                or token in byte_fallback_to_byte.keys()
                or token.startswith("<unused_whitespace__")
            ):
                continue

            problematic_decompostions.update(x for x in decompose(token) if len(x) > 1)

        for token in problematic_decompostions:
            extra_merges, extra_vocab = get_merges(token)
            all_extra_vocab |= extra_vocab

            for merge in extra_merges:
                if merge not in all_extra_merges_set:
                    all_extra_merges_set.add(merge)

                    if (
                        make_whitespace_consistent
                        and sum(c in extra_whitespace for c in token) > 1
                    ):
                        all_extra_post_merges.append(merge)
                    else:
                        all_extra_pre_merges.append(merge)

        surface_forms += sorted(all_extra_vocab)
        merges = all_extra_pre_merges + merges + all_extra_post_merges

        tokenizer_data["model"]["vocab"] = {
            surface_form: i for i, surface_form in enumerate(surface_forms)
        }
        tokenizer_data["model"]["merges"] = merges
    elif isinstance(tokenizer._tokenizer.model, models.WordPiece):
        tokenizer_data["model"]["vocab"] = {
            surface_form: i for i, surface_form in enumerate(surface_forms)
        }
    else:
        raise ValueError(f"Unknown model type: {type(tokenizer._tokenizer.model)}")

    if (
        match_special_tokens_to is not None
        and match_special_tokens_to_data.get("post_processor") is not None
    ):
        fix_postprocessor_data(
            match_special_tokens_to_data["post_processor"], surface_forms
        )

        tokenizer_data["post_processor"] = match_special_tokens_to_data[
            "post_processor"
        ]

    json.dump(tokenizer_data, open(f.name, "w"))
    tokenizer._tokenizer = Tokenizer.from_file(f.name)

    tokenizer._tokenizer.decoder = decoders.ByteLevel()
    if match_special_tokens_to is not None:
        tokenizer.eos_token = match_special_tokens_to.eos_token
        tokenizer.pad_token = match_special_tokens_to.pad_token
        tokenizer.sep_token = match_special_tokens_to.sep_token
        tokenizer.unk_token = match_special_tokens_to.unk_token
        tokenizer.bos_token = match_special_tokens_to.bos_token
        tokenizer.cls_token = match_special_tokens_to.cls_token
        tokenizer.mask_token = match_special_tokens_to.mask_token
        tokenizer.unk_token = match_special_tokens_to.unk_token

    return (
        tokenizer,
        len(tokenizer) - original_length if preserved_original_token_indices else None,
    )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer, _ = convert_to_byte_level(
        tokenizer,
        match_special_tokens_to=AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B"
        ),
    )
    # tokenizer, n_added = convert_to_byte_level(tokenizer, keep_normalizer=True, keep_pretokenizer=True)
    tokenizer.save_pretrained("output_debug")
