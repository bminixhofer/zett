from transformers import AutoTokenizer
from dataclasses import dataclass
import h5py
from transformers import HfArgumentParser
from pathlib import Path
from tokenizers import models


@dataclass
class Args:
    output: str = "output_ko_debug_tokenizer"
    tokenizer_data: str = "artifacts/tokenizer_data_10l_xlmr.hdf5"
    lang_code: str = "ko"
    tokenizer_directory: str = "/mnt/disks/persist/large_tokenizers/"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    indices = h5py.File(args.tokenizer_data)[f"{args.lang_code}_indices"][0]
    tokenizer = AutoTokenizer.from_pretrained(
        Path(args.tokenizer_directory) / args.lang_code
    )
    pieces = tokenizer._tokenizer.model.get_pieces()
    tokenizer._tokenizer.model = models.Unigram([pieces[i] for i in indices])
    tokenizer.save_pretrained(args.output)
