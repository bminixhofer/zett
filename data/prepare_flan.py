from dataclasses import dataclass
from transformers import HfArgumentParser
from datasets import load_dataset
from pathlib import Path


@dataclass
class Args:
    out_train_dir: str = "/mnt/disks/persist/train"
    out_valid_dir: str = "/mnt/disks/persist/valid"

if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    dset = load_dataset("benjamin/flanv2_subsample")
    dset["train"].save_to_disk(Path(args.out_train_dir) / "flan")
    dset["valid"].save_to_disk(Path(args.out_valid_dir) / "flan")