from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)
from dataclasses import dataclass
from datasets import load_dataset
import torch
import numpy as np
from tqdm.auto import tqdm


@dataclass
class Args:
    model_name_or_path: str = "../../../output_t/xlm-roberta-base-fi"
    dataset_path: str = "../../../datasets/valid/fi.parquet"
    max_length: int = 128
    batch_size: int = 128
    device: str = "cuda"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    dset = load_dataset(
        "parquet", data_files={"train": args.dataset_path}, split="train"
    )

    model = (
        AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
        .eval()
        .to(args.device)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dset = dset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        ),
        batched=True,
    )
    dset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        ),
    )

    losses = []
    accs = []

    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            acc = (
                (batch["labels"] == outputs.logits.argmax(-1))[batch["labels"] != -100]
                .float()
                .mean()
                .item()
            )
            accs.append(acc)
            losses.append(outputs.loss.item())

        bar.update(1)
        bar.set_postfix(loss=np.mean(losses), acc=np.mean(accs))
