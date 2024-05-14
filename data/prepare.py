from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from transformers import HfArgumentParser
import os
import math
from typing import List, Optional

METADATA_PATH = "data/madlad400_metadata.csv"


@dataclass
class Args:
    max_train_pages_per_language: int = 2_000_000
    valid_percent: float = 1.0
    out_train_dir: str = "/mnt/disks/persist/train"
    out_valid_dir: str = "/mnt/disks/persist/valid"
    include_langs: Optional[List[str]] = None


def process_text(text):
    return text.replace("\\t", "\t").replace("\\n", "\n").replace("\\_", "\\")


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    out_train_dir = Path(args.out_train_dir)
    out_valid_dir = Path(args.out_valid_dir)

    out_train_dir.mkdir(exist_ok=True, parents=True)
    out_valid_dir.mkdir(exist_ok=True, parents=True)

    metadata = pd.read_csv(METADATA_PATH).set_index("lang_code")
    metadata = metadata.loc[
        [
            lang
            for lang in metadata.index
            if lang != "-"
            and (args.include_langs is None or lang in args.include_langs)
        ]
    ]  # '-' indicates romanization

    total_pages = 0

    for lang in metadata.index:
        n_pages = math.ceil(
            min(
                metadata.loc[lang]["n_pages"],
                args.max_train_pages_per_language * (1.0 + args.valid_percent / 100.0),
            )
        )
        total_pages += n_pages

    print(f"Total pages: {total_pages}")
    bar = tqdm(total=total_pages)

    for lang in metadata.index:
        n_pages = math.ceil(
            min(
                metadata.loc[lang]["n_pages"],
                args.max_train_pages_per_language * (1.0 + args.valid_percent / 100.0),
            )
        )
        n_valid_pages = math.ceil(
            n_pages / (1.0 + args.valid_percent / 100.0) * (args.valid_percent / 100.0)
        )
        n_train_pages = n_pages - n_valid_pages

        print(f"Downloading {lang} ({n_train_pages=}, {n_valid_pages=})")

        dset = load_dataset(
            "allenai/madlad-400",
            lang,
            streaming=True,
            split="clean",
        )

        iterator = iter(dset)

        n_valid = 0
        n_train = 0

        train_data = []
        valid_data = []

        for i in range(n_valid_pages):
            text = next(iterator)["text"]
            text = process_text(text)
            valid_data.append((text,))

            bar.update(1)

        for i in range(n_train_pages):
            try:
                text = next(iterator)["text"]
                text = process_text(text)
                train_data.append((text,))
            except StopIteration:
                print(
                    "Warning: Reached end of dataset before expected number of pages."
                )
                break

            bar.update(1)

        train_df = pd.DataFrame(train_data)
        train_df.columns = ["text"]
        train_df.to_parquet(out_train_dir / f"{lang}.parquet")

        valid_df = pd.DataFrame(valid_data)
        valid_df.columns = ["text"]
        valid_df.to_parquet(out_valid_dir / f"{lang}.parquet")
