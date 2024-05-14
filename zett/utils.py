import numpy as np
from itertools import chain
from flax import linen as nn
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import optax
from tqdm.auto import tqdm
from jax.sharding import PositionalSharding
from jax.sharding import PartitionSpec as P
import copy
from tokenizers import pre_tokenizers, Tokenizer, models
from transformers import ByT5Tokenizer, AutoTokenizer, PreTrainedTokenizerFast
from pathlib import Path
import json
from tempfile import NamedTemporaryFile
from transformers.utils.hub import cached_file
from flax import traverse_util
from flax.serialization import msgpack_restore

PRIOR_ESTIMATION_SUBSAMPLE = 1_000_000
NEGATIVE_INF_FILL_VALUE = -100_000
MAX_CHARS_PER_TOKEN = 16
EPSILON = 1e-8
SHARDING = PositionalSharding(np.array(jax.local_devices()))
CACHE_DIR = Path(__file__).parent / ".." / ".cache"
MADLAD_METADATA = pd.read_csv("data/madlad400_metadata.csv", index_col="lang_code")
SPLIT_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    x = x / x.sum(axis=axis)
    return x


def cosine_similarity(x, y):
    # x, y: (batch, dim)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)

    return (x * y).sum(axis=-1)


def huber_loss(predictions, targets=None, delta: float = 1.0):
    errors = (predictions - targets) if (targets is not None) else predictions
    # 0.5 * err^2                  if |err| <= d
    # 0.5 * d^2 + d * (|err| - d)  if |err| > d
    abs_errors = jnp.abs(errors)
    quadratic = jnp.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def default_pretokenize(text):
    # very conservative default pre-tokenization, split such that there is never whitespace at the end of a token (but allowed at the start)
    # we need this because e.g. Llama tokenizer does not specify any pre-tokenization (in the HF version)

    tokens = []
    token = ""
    has_non_whitespace = False

    for c in text:
        if c.isspace():
            if has_non_whitespace:
                tokens.append((token, None))
                token = ""
                has_non_whitespace = False
        else:
            has_non_whitespace = True

        token += c

    if len(token) > 0:
        tokens.append((token, None))

    return tokens


def create_learning_rate_fn(args):
    """Returns a linear warmup, linear_decay learning rate function."""

    args = copy.deepcopy(args)

    if not hasattr(args, "random_warmup_steps"):
        args.random_warmup_steps = 0

    if not getattr(args, "random_learning_rate", None):
        args.random_learning_rate = args.learning_rate

    if isinstance(args.warmup_steps, int):
        args.warmup_steps = [args.warmup_steps]

    if isinstance(args.learning_rate, float):
        args.learning_rate = [args.learning_rate] * len(args.warmup_steps)

    random_warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=args.random_learning_rate,
        transition_steps=args.random_warmup_steps,
    )

    warmup_fns = []
    warmup_boundaries = [args.random_warmup_steps]

    for i, warmup_boundary in enumerate(args.warmup_steps):
        n_steps = warmup_boundary - warmup_boundaries[-1]
        warmup_boundaries.append(warmup_boundary)

        warmup_fns.append(
            optax.linear_schedule(
                init_value=0.0,
                end_value=args.learning_rate[i],
                transition_steps=n_steps,
            )
        )

    decay_fn = optax.cosine_decay_schedule(
        init_value=args.learning_rate[-1],
        decay_steps=args.steps - warmup_boundary,
        alpha=args.learning_rate_alpha,
    )
    random_schedule_fn = optax.join_schedules(
        schedules=[random_warmup_fn, *warmup_fns, decay_fn],
        boundaries=[
            args.random_warmup_steps,
            *args.warmup_steps,
        ],
    )
    pretrained_schedule_fn = optax.join_schedules(
        schedules=[optax.constant_schedule(0.0), *warmup_fns, decay_fn],
        boundaries=[
            args.random_warmup_steps,
            *args.warmup_steps,
        ],
    )

    return random_schedule_fn, pretrained_schedule_fn


class Rescaler(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.w = self.param(
            "w",
            jax.nn.initializers.constant(1),
            (1, self.dim),
            self.dtype,
        )
        self.b = self.param(
            "b",
            jax.nn.initializers.constant(0),
            (1, self.dim),
            self.dtype,
        )

    def __call__(self, x):
        return self.w * x + self.b

    @staticmethod
    def scale_to(x, target=None, target_stds=None, target_means=None):
        if target_stds is None:
            target_stds = target.std(axis=0)
        if target_means is None:
            target_means = target.mean(axis=0)

        w = (target_stds / (x.std(axis=0) + EPSILON))[None]
        b = (target_means - (x * w).mean(axis=0))[None]

        return w, b


def has_same_special_tokens(source, target):
    # check if all special tokens are the same
    return all(
        getattr(source, attr) == getattr(target, attr)
        for attr in [
            "eos_token",
            "pad_token",
            "sep_token",
            "unk_token",
            "bos_token",
            "cls_token",
            "mask_token",
        ]
    )


def pad_to_multiple_of(x, n, constant_values=0):
    remainder = x.shape[0] % n
    if remainder == 0:
        return x

    pad_width = [(0, n - remainder)] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad_width, mode="constant", constant_values=constant_values)


def make_whitespace_consistent(tokenizer, maxlen):
    extra_whitespace = ["Ġ", "Ċ", "ĉ"]
    consistent_tokenizer = copy.deepcopy(tokenizer)

    pieces = consistent_tokenizer._tokenizer.model.get_pieces()

    for i in range(len(pieces)):
        if sum(c in extra_whitespace for c in pieces[i][0]) > 1:
            pieces[i] = (f"<unused_whitespace__{i}>", NEGATIVE_INF_FILL_VALUE)

    for c1 in extra_whitespace:
        for i in range(1, maxlen):
            for c2 in extra_whitespace:
                pieces.append((c2 + c1 * i, 0.0))

    consistent_tokenizer._tokenizer.model = models.Unigram(
        pieces, unk_id=tokenizer.unk_token_id
    )

    return consistent_tokenizer


def copy_tokenizer_auxiliaries(source, target):
    # add special tokens
    # make sure these keep positions as in the original vocab
    # because models may depend on special token positions (e.g. model.config.pad_token_id)
    if source.get_vocab() == target.get_vocab() and has_same_special_tokens(
        source, target
    ):
        return target

    needs_piece_update = False
    for source_token in source.all_special_tokens:
        if (
            source_token not in target.all_special_tokens
            or source.convert_tokens_to_ids(source_token)
            != target.convert_tokens_to_ids(source_token)
        ):
            needs_piece_update = True

    if needs_piece_update:  # only implemented for Unigram
        pieces = target._tokenizer.model.get_pieces()
        pieces = [
            piece for piece in pieces if piece[0] not in source.all_special_tokens
        ]

        for i in np.argsort(source.all_special_ids):
            pieces.insert(
                source.all_special_ids[i], (source.all_special_tokens[i], 0.0)
            )

        target._tokenizer.model.set_pieces(pieces)

    if source._tokenizer.post_processor is not None:
        target._tokenizer.post_processor = source._tokenizer.post_processor

    # make sure length etc. is updated and cache is cleared
    with NamedTemporaryFile() as f:
        original_name_or_path = target.name_or_path
        target._tokenizer.save(f.name)

        target = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(f.name),
            clean_up_tokenization_spaces=False,
        )
        target.name_or_path = original_name_or_path

    target.eos_token = source.eos_token
    target.pad_token = source.pad_token
    target.sep_token = source.sep_token
    target.unk_token = source.unk_token
    target.bos_token = source.bos_token
    target.cls_token = source.cls_token
    target.mask_token = source.mask_token
    target.unk_token = source.unk_token
    return target


def unset_tokenizer_special_tokens(tokenizer):
    tokenizer.eos_token = None
    tokenizer.pad_token = None
    tokenizer.sep_token = None
    tokenizer.unk_token = None
    tokenizer.bos_token = None
    tokenizer.cls_token = None
    tokenizer.mask_token = None


def get_prior(mode, input_ids, tokenizer, padding=0):
    if mode == "keep":
        return None
    elif mode == "reestimate":
        uniq, counts = np.unique(
            input_ids,
            return_counts=True,
        )
        target_priors = np.ones(len(tokenizer) + padding)  # laplace smoothing
        target_priors[uniq] += counts
        target_priors /= target_priors.sum()
        target_priors = jnp.array(np.log(target_priors))
    elif mode == "use_tokenizer":
        target_priors = jnp.array(
            np.pad(
                tokenizer._tokenizer.model.get_scores(),
                (0, padding),
                constant_values=NEGATIVE_INF_FILL_VALUE,
            )
        )

    # special tokens bias are 0 by convention
    if len(tokenizer.all_special_ids) > 0:
        target_priors = target_priors.at[jnp.array(tokenizer.all_special_ids)].set(0.0)
    return target_priors


def tokenize_function(examples, block_size, tokenizer):
    # NULL characters cause problems in HDF5 VLEN encoding
    texts = [text.replace("\x00", "").strip() for text in examples["text"]]
    encodings = tokenizer(
        texts,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    output = {
        "input_ids": encodings["input_ids"],
        "offset_mapping": encodings["offset_mapping"],
    }

    # group texts
    # does not allow grouping across batches, but that's fine
    concatenated_examples = {k: list(chain(*output[k])) for k in output.keys()}
    total_length = len(concatenated_examples[list(output.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

    return result


# assumes byte pretokenization
CHARS_TO_BYTES = {
    "Ā": 0,
    "ā": 1,
    "Ă": 2,
    "ă": 3,
    "Ą": 4,
    "ą": 5,
    "Ć": 6,
    "ć": 7,
    "Ĉ": 8,
    "ĉ": 9,
    "Ċ": 10,
    "ċ": 11,
    "Č": 12,
    "č": 13,
    "Ď": 14,
    "ď": 15,
    "Đ": 16,
    "đ": 17,
    "Ē": 18,
    "ē": 19,
    "Ĕ": 20,
    "ĕ": 21,
    "Ė": 22,
    "ė": 23,
    "Ę": 24,
    "ę": 25,
    "Ě": 26,
    "ě": 27,
    "Ĝ": 28,
    "ĝ": 29,
    "Ğ": 30,
    "ğ": 31,
    "Ġ": 32,
    "!": 33,
    '"': 34,
    "#": 35,
    "$": 36,
    "%": 37,
    "&": 38,
    "'": 39,
    "(": 40,
    ")": 41,
    "*": 42,
    "+": 43,
    ",": 44,
    "-": 45,
    ".": 46,
    "/": 47,
    "0": 48,
    "1": 49,
    "2": 50,
    "3": 51,
    "4": 52,
    "5": 53,
    "6": 54,
    "7": 55,
    "8": 56,
    "9": 57,
    ":": 58,
    ";": 59,
    "<": 60,
    "=": 61,
    ">": 62,
    "?": 63,
    "@": 64,
    "A": 65,
    "B": 66,
    "C": 67,
    "D": 68,
    "E": 69,
    "F": 70,
    "G": 71,
    "H": 72,
    "I": 73,
    "J": 74,
    "K": 75,
    "L": 76,
    "M": 77,
    "N": 78,
    "O": 79,
    "P": 80,
    "Q": 81,
    "R": 82,
    "S": 83,
    "T": 84,
    "U": 85,
    "V": 86,
    "W": 87,
    "X": 88,
    "Y": 89,
    "Z": 90,
    "[": 91,
    "\\": 92,
    "]": 93,
    "^": 94,
    "_": 95,
    "`": 96,
    "a": 97,
    "b": 98,
    "c": 99,
    "d": 100,
    "e": 101,
    "f": 102,
    "g": 103,
    "h": 104,
    "i": 105,
    "j": 106,
    "k": 107,
    "l": 108,
    "m": 109,
    "n": 110,
    "o": 111,
    "p": 112,
    "q": 113,
    "r": 114,
    "s": 115,
    "t": 116,
    "u": 117,
    "v": 118,
    "w": 119,
    "x": 120,
    "y": 121,
    "z": 122,
    "{": 123,
    "|": 124,
    "}": 125,
    "~": 126,
    "ġ": 127,
    "Ģ": 128,
    "ģ": 129,
    "Ĥ": 130,
    "ĥ": 131,
    "Ħ": 132,
    "ħ": 133,
    "Ĩ": 134,
    "ĩ": 135,
    "Ī": 136,
    "ī": 137,
    "Ĭ": 138,
    "ĭ": 139,
    "Į": 140,
    "į": 141,
    "İ": 142,
    "ı": 143,
    "Ĳ": 144,
    "ĳ": 145,
    "Ĵ": 146,
    "ĵ": 147,
    "Ķ": 148,
    "ķ": 149,
    "ĸ": 150,
    "Ĺ": 151,
    "ĺ": 152,
    "Ļ": 153,
    "ļ": 154,
    "Ľ": 155,
    "ľ": 156,
    "Ŀ": 157,
    "ŀ": 158,
    "Ł": 159,
    "ł": 160,
    "¡": 161,
    "¢": 162,
    "£": 163,
    "¤": 164,
    "¥": 165,
    "¦": 166,
    "§": 167,
    "¨": 168,
    "©": 169,
    "ª": 170,
    "«": 171,
    "¬": 172,
    "Ń": 173,
    "®": 174,
    "¯": 175,
    "°": 176,
    "±": 177,
    "²": 178,
    "³": 179,
    "´": 180,
    "µ": 181,
    "¶": 182,
    "·": 183,
    "¸": 184,
    "¹": 185,
    "º": 186,
    "»": 187,
    "¼": 188,
    "½": 189,
    "¾": 190,
    "¿": 191,
    "À": 192,
    "Á": 193,
    "Â": 194,
    "Ã": 195,
    "Ä": 196,
    "Å": 197,
    "Æ": 198,
    "Ç": 199,
    "È": 200,
    "É": 201,
    "Ê": 202,
    "Ë": 203,
    "Ì": 204,
    "Í": 205,
    "Î": 206,
    "Ï": 207,
    "Ð": 208,
    "Ñ": 209,
    "Ò": 210,
    "Ó": 211,
    "Ô": 212,
    "Õ": 213,
    "Ö": 214,
    "×": 215,
    "Ø": 216,
    "Ù": 217,
    "Ú": 218,
    "Û": 219,
    "Ü": 220,
    "Ý": 221,
    "Þ": 222,
    "ß": 223,
    "à": 224,
    "á": 225,
    "â": 226,
    "ã": 227,
    "ä": 228,
    "å": 229,
    "æ": 230,
    "ç": 231,
    "è": 232,
    "é": 233,
    "ê": 234,
    "ë": 235,
    "ì": 236,
    "í": 237,
    "î": 238,
    "ï": 239,
    "ð": 240,
    "ñ": 241,
    "ò": 242,
    "ó": 243,
    "ô": 244,
    "õ": 245,
    "ö": 246,
    "÷": 247,
    "ø": 248,
    "ù": 249,
    "ú": 250,
    "û": 251,
    "ü": 252,
    "ý": 253,
    "þ": 254,
    "ÿ": 255,
}
BYTES_TO_CHARS = {v: k for k, v in CHARS_TO_BYTES.items()}


def get_sample_indices(n, p, batch_size, min_k, n_samples):
    p = np.where(p > NEGATIVE_INF_FILL_VALUE, p, -np.inf)
    p = np.exp(p)

    indices = np.empty((n_samples, batch_size), dtype=np.int32)

    random_offset = 0
    random_indices = np.arange(n)
    np.random.shuffle(random_indices)

    n_samples_per_k = n_samples // min_k
    assert n_samples_per_k * min_k == n_samples

    for i in range(n_samples):
        if (i + 1) % n_samples_per_k == 0:
            num_random = len(random_indices) - random_offset
        else:
            num_random = len(random_indices) // n_samples_per_k

        indices[i, :num_random] = random_indices[
            random_offset : random_offset + num_random
        ]

        if (i + 1) % n_samples_per_k == 0:
            random_offset = 0
            np.random.shuffle(random_indices)
        else:
            random_offset += num_random

        sample_p = p.copy()
        sample_p[indices[i, :num_random]] = 0
        sample_p /= sample_p.sum()
        indices[i, num_random:] = np.random.choice(
            n, size=batch_size - num_random, p=sample_p, replace=False
        )

    return indices


def get_surface_form_matrix(
    tokenizer_or_tokens, maxlen, tokenizer_to_use=None, padding=0, verbose=False
):
    # tokens are expected to be byte encoded
    if isinstance(tokenizer_or_tokens, list):
        tokens = tokenizer_or_tokens
    else:
        tokenizer = tokenizer_or_tokens
        tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))

    vocab_size = len(tokens)
    surface_form_matrix = np.full(
        (vocab_size + padding, maxlen),
        tokenizer_to_use.pad_token_id if tokenizer_to_use is not None else 0,
        dtype=np.int32,
    )

    n_truncated = 0

    for i, token in tqdm(enumerate(tokens), total=vocab_size, disable=not verbose):
        if token in tokenizer_to_use.all_special_tokens:
            surface_form_matrix[i, 0] = tokenizer_to_use.convert_tokens_to_ids(token)
            continue

        token_bytes = bytes([CHARS_TO_BYTES[c] for c in token])

        if isinstance(tokenizer_to_use, ByT5Tokenizer):
            ids = tokenizer_to_use.convert_tokens_to_ids([chr(i) for i in token_bytes])
        else:
            # assume hn tokenizer uses byte pretokenization
            ids = [x.id for x in tokenizer_to_use._tokenizer.model.tokenize(token)]

        if len(ids) > maxlen:
            ids = ids[:maxlen]
            n_truncated += 1

        surface_form_matrix[i, : len(ids)] = ids

    return surface_form_matrix, n_truncated


def convert_ids_to_tokens(ids, surface_forms):
    tokens = []
    for i in ids:
        s = "".join([BYTES_TO_CHARS[x] for x in surface_forms[i] if x != 0])
        if s == "<|endoftext|>":
            continue

        tokens.append(s)

    return tokens


def get_subtree(tree, path):
    for p in path:
        tree = tree[p]

    return tree


def needs_output_mapping(model_class):
    return model_class in {"FlaxAutoModelForMaskedLM", "FlaxAutoModelForCausalLM"}


# from optax: https://github.com/google-deepmind/optax/blob/4eeef48f17cc2d9a5f9e4c5404ef9c766e44fbc9/optax/losses/_classification.py#L215
def kl_divergence_with_log_targets(log_predictions, log_targets):
    loss = jnp.exp(log_targets) * (log_targets - log_predictions)
    return jnp.sum(loss, axis=-1)


def load_params(model_name_or_path, **kwargs):
    try:
        index = cached_file(
            model_name_or_path, "flax_model.msgpack.index.json", **kwargs
        )
    except OSError:
        index = None

    if index is not None:
        index = json.load(open(index))
        files = [
            cached_file(model_name_or_path, x, **kwargs)
            for x in set(index["weight_map"].values())
        ]
    else:
        files = [cached_file(model_name_or_path, "flax_model.msgpack", **kwargs)]

    params = {}
    for x in files:
        params.update(traverse_util.flatten_dict(msgpack_restore(open(x, "rb").read())))

    return traverse_util.unflatten_dict(params)


def keystr(x):
    if hasattr(x, "name"):
        return x.name
    elif hasattr(x, "key"):
        return x.key
    elif hasattr(x, "idx"):
        return x.idx

    assert isinstance(x, str)
    return x


keys_to_model_shard = {
    "target_surface_forms",
    "target_priors",
    "mask",
    "ids_to_embed",
    "input_ids",
    "attention_mask",
}


def get_batch_pspecs(batch):
    pspecs = {}
    keys_to_ignore = {"lang_code", "metrics"}

    for key in batch.keys():
        if key in keys_to_ignore:
            continue

        pspec = [None] * (batch[key].ndim)

        if key in keys_to_model_shard:
            pspec[0] = "model"

        pspecs[key] = P(*pspec)

    return pspecs


def to_global_batch(batch, shardings):
    if shardings is not None:
        for key in keys_to_model_shard:
            if key not in batch:
                continue

            data = batch.pop(key)

            def cb(index):
                return data[index]

            batch[key] = jax.make_array_from_callback(data.shape, shardings[key], cb)

    return batch
