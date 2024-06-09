from datasets import load_dataset, load_from_disk
import math
import os
import numpy as np
import pandas as pd
from collections import Counter
import jax
from datasets import DatasetDict

from zett.utils import MAX_CHARS_PER_TOKEN
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class TrainDataset(IterableDataset):
    def __init__(
        self,
        langs,
        train_directory,
        language_probs,
        batch_size,
        block_size,
        do_sequence_packing=True,
    ):
        self.langs = langs
        self.train_directory = train_directory
        self.language_probs = language_probs / language_probs.sum()

        self.batch_size = batch_size
        self.block_size = block_size
        self.do_sequence_packing = do_sequence_packing
        self.char_length = block_size * MAX_CHARS_PER_TOKEN

        self.dataset = {}
        for lang in list(self.langs):
            if os.path.exists(os.path.join(train_directory, lang)):
                dset = load_from_disk(os.path.join(train_directory, lang))
                if isinstance(dset, DatasetDict):
                    dset = dset["train"]

                self.dataset[lang] = dset
            elif os.path.exists(os.path.join(train_directory, f"{lang}.parquet")):
                self.dataset[lang] = load_dataset(
                    "parquet",
                    data_files=os.path.join(train_directory, f"{lang}.parquet"),
                    split="train",
                )
            else:
                raise ValueError(
                    f"Could not find training data for language {lang} in {train_directory}"
                )

    def get_texts_in_each_language(self, n):
        return {lang: self.dataset[lang][:n]["text"] for lang in self.langs}

    def get_texts(self, n):
        texts = []

        for batch in self:
            texts.extend(batch["texts"])
            if len(texts) >= n:
                break

        return texts[:n]

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_idx = worker_info.id if worker_info is not None else 0

        global_random_states = {lang: np.random.RandomState(0) for lang in self.langs}
        local_random_state = np.random.RandomState(worker_idx)

        self.language_orders = {
            lang: global_random_states[lang].permutation(len(self.dataset[lang]))[
                worker_idx::num_workers
            ]
            for lang in self.langs
        }
        examples_this_epoch = {lang: 0 for lang in self.langs}

        while True:
            texts = []

            for _ in range(self.batch_size):
                language = local_random_state.choice(self.langs, p=self.language_probs)

                text = []
                n_chars = 0

                # try to avoid padding during training if do_packing is True
                while n_chars < self.char_length:
                    index = int(
                        self.language_orders[language][examples_this_epoch[language]]
                    )
                    current_text = self.dataset[language][index]["text"].strip()

                    # sample text spans from the text
                    max_length = self.char_length - n_chars
                    start = 0#np.random.randint(0, max(len(current_text) - max_length, 0) + 1)
                    end = start + min(max_length, len(current_text) - start)

                    is_truncated = len(current_text) > end
                    current_text = current_text[start:end]

                    examples_this_epoch[language] += 1
                    if examples_this_epoch[language] == len(
                        self.language_orders[language]
                    ):
                        # reshuffle
                        self.language_orders[language] = global_random_states[
                            language
                        ].permutation(len(self.dataset[language]))[
                            worker_idx::num_workers
                        ]
                        examples_this_epoch[language] = 0

                    if len(current_text) == 0:
                        continue

                    text.append((current_text, is_truncated))
                    n_chars += len(current_text)

                    if not self.do_sequence_packing:
                        break

                texts.append(tuple(text))

            yield {
                "texts": texts,
                "lang_code": "all" if len(self.langs) > 1 else language,
            }


class ValidDataset(Dataset):
    def __init__(self, langs, valid_directory, n_subsample, batch_size, block_size):
        self.batch_size = batch_size
        self.langs = langs
        self.n_subsample = n_subsample
        self.char_length = block_size * MAX_CHARS_PER_TOKEN

        self.dataset = {}

        for lang in self.langs:
            if lang == "flan":
                lang_to_load = "en"  # TEMP use English for FLAN validation
            else:
                lang_to_load = lang

            if os.path.exists(os.path.join(valid_directory, lang_to_load)):
                dset = load_from_disk(os.path.join(valid_directory, lang_to_load))
                if isinstance(dset, DatasetDict):
                    dset = dset["train"]

                if n_subsample is not None:
                    dset = dset.select(range(min(len(dset), n_subsample)))

                self.dataset[lang] = dset
            elif os.path.exists(
                os.path.join(valid_directory, f"{lang_to_load}.parquet")
            ):
                self.dataset[lang] = load_dataset(
                    "parquet",
                    data_files=os.path.join(valid_directory, f"{lang_to_load}.parquet"),
                    split=f"train[:{n_subsample}]"
                    if n_subsample is not None
                    else "train",
                )
            else:
                raise ValueError(
                    f"Could not find validation data for language {lang_to_load} in {valid_directory}"
                )

    def __len__(self):
        return sum(
            # math.floor here is equivalent to drop_last
            math.floor(len(self.dataset[lang]) / self.batch_size)
            for lang in self.langs
        )

    def __getitem__(self, idx):
        lang_idx = 0
        while idx >= math.floor(
            len(self.dataset[self.langs[lang_idx]]) / self.batch_size
        ):
            idx -= math.floor(len(self.dataset[self.langs[lang_idx]]) / self.batch_size)
            lang_idx += 1

        texts = self.dataset[self.langs[lang_idx]][idx * self.batch_size:(idx + 1) * self.batch_size]["text"]
        texts = [((text[:self.char_length], len(text) > self.char_length),) for text in texts]

        return {
            "texts": texts,
            "lang_code": self.langs[lang_idx],
        }
