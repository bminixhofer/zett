from tokenizers import (
    SentencePieceUnigramTokenizer,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)
from transformers import HfArgumentParser
from dataclasses import dataclass
from datasets import load_from_disk
from pathlib import Path
from transformers import PreTrainedTokenizerFast
import os
import sentencepiece as spm
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

from zett.tokenizer_converters import convert_to_byte_level
from zett.utils import BYTES_TO_CHARS


@dataclass
class Args:
    output: str = "/mnt/disks/persist/tokenizers_spm/cpp"
    dataset_path: str = "/mnt/disks/persist/valid/cpp"
    vocab_size: int = 50000
    max_length: int = 16
    for_code: bool = True


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset_path + ".txt"

    if not os.path.exists(dataset_path):
        dset = load_from_disk(args.dataset_path)
        open(dataset_path, "w").write("\n".join(dset["text"]))

    spm_path = str(output / "spm.model")
    spm.SentencePieceTrainer.train(
        input=dataset_path,
        model_prefix=spm_path[: -len(".model")],
        vocab_size=args.vocab_size,
        input_sentence_size=1_000_000,
        normalization_rule_name="identity",
        max_sentencepiece_length=args.max_length,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=SentencePieceUnigramTokenizer.from_spm(spm_path),
        clean_up_tokenization_spaces=False,
    )
    tokenizer.save_pretrained(output / "original")
    tokenizer = convert_to_byte_level(tokenizer)[0]

    if args.for_code:
        # add whitespace pieces to tokenizer
        with NamedTemporaryFile() as f:
            tokenizer._tokenizer.save(f.name)
            tokenizer_data = json.load(open(f.name))

            scores = [score for _, score in tokenizer_data["model"]["vocab"]]
            pieces = [piece for piece, _ in tokenizer_data["model"]["vocab"]]
            pieces_set = set(pieces)

            to_add = []

            for x in ["\n", "\t", " "]:
                x = "".join(BYTES_TO_CHARS[b] for b in x.encode("utf-8"))

                for i in range(1, args.max_length + 1):
                    piece = x * i
                    if piece not in pieces_set:
                        to_add.append(piece)

            tokenizer_data["model"]["vocab"] = tokenizer_data["model"]["vocab"] + [
                (piece, -1.0) for piece in to_add
            ]

            json.dump(tokenizer_data, open(f.name, "w"))
            tokenizer._tokenizer = Tokenizer.from_file(f.name)

    tokenizer.save_pretrained(output)
