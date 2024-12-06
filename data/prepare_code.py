from dataclasses import dataclass, field
from typing import List, Optional
from transformers import HfArgumentParser
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path


@dataclass
class Args:
    max_train_pages_per_language: int = 2_000_000
    valid_percent: float = 1.0
    out_train_dir: str = "/mnt/disks/persist/train"
    out_valid_dir: str = "/mnt/disks/persist/valid"
    include_langs: Optional[List[str]] = field(
        default_factory=lambda: [
            "cpp",
            "go",
            "java",
            "javascript",
            "python",
            "github-issues-filtered-structured",
        ]
    )


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    out_train_dir = Path(args.out_train_dir)
    out_valid_dir = Path(args.out_valid_dir)

    out_train_dir.mkdir(exist_ok=True, parents=True)
    out_valid_dir.mkdir(exist_ok=True, parents=True)

    total_pages = args.max_train_pages_per_language * len(args.include_langs)
    bar = tqdm(total=total_pages)

    for lang in args.include_langs:
        # assume we have enough - need to fix
        n_valid_pages = int(
            args.max_train_pages_per_language * (args.valid_percent / 100.0)
        )
        n_train_pages = args.max_train_pages_per_language

        print(f"Downloading {lang} ({n_train_pages=}, {n_valid_pages=})")

        dset = load_dataset(
            "bigcode/starcoderdata", data_dir=lang, split="train", streaming=True
        )

        iterator = iter(dset)

        n_valid = 0
        n_train = 0

        train_data = []
        valid_data = []

        i = 0
        while i < n_valid_pages:
            try:
                text = next(iterator)["content"]
            except Exception as e:
                print("Warning: Got error.")
                print(e)
                continue
            valid_data.append((text,))

            i += 1
            bar.update(1)

        i = 0
        while i < n_train_pages:
            try:
                try:
                    text = next(iterator)["content"]
                except Exception as e:
                    print("Warning: Got error.")
                    print(e)
                    continue
                train_data.append((text,))
            except StopIteration:
                print(
                    "Warning: Reached end of dataset before expected number of pages."
                )
                break

            i += 1
            bar.update(1)

        train_df = pd.DataFrame(train_data)
        train_df.columns = ["text"]
        train_df.to_parquet(out_train_dir / f"{lang}.parquet")

        valid_df = pd.DataFrame(valid_data)
        valid_df.columns = ["text"]
        valid_df.to_parquet(out_valid_dir / f"{lang}.parquet")
