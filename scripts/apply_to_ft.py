from dataclasses import dataclass, field
from typing import List
from transformers import HfArgumentParser, AutoTokenizer
import transformers
from tqdm.auto import tqdm


@dataclass
class Args:
    output: str
    base_model_path: str
    ft_model_path: str
    tokenizer_swapped_base_model_path: str
    output_ft_with_base_embeddings: str = None
    lambdas: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7])
    model_class = "AutoModelForCausalLM"


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    print(args)

    model_class = getattr(transformers, args.model_class)

    tokenizer_swapped_base_model = model_class.from_pretrained(
        args.tokenizer_swapped_base_model_path
    )
    base_model = model_class.from_pretrained(args.base_model_path)
    ft_model = model_class.from_pretrained(args.ft_model_path)

    swap_diffs = []

    embedding_params = {
        base_model.get_input_embeddings().weight,
        base_model.get_output_embeddings().weight,
    }

    for (swapped_name, swapped_param), (orig_name, orig_param) in zip(
        tokenizer_swapped_base_model.named_parameters(), base_model.named_parameters()
    ):
        assert swapped_name == orig_name

        if orig_param in embedding_params:
            swap_diffs.append(0.0)
        else:
            swap_diffs.append(swapped_param - orig_param)

    tokenizer_swapped = AutoTokenizer.from_pretrained(
        args.tokenizer_swapped_base_model_path
    )
    tokenizer_base = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer_ft = AutoTokenizer.from_pretrained(args.ft_model_path)
    tokenizer_swapped.chat_template = tokenizer_ft.chat_template

    assert tokenizer_base.get_vocab() == tokenizer_ft.get_vocab()

    for lmd in tqdm(args.lambdas):
        for param, swap_diff in zip(ft_model.parameters(), swap_diffs):
            param.data += swap_diff * lmd

        ft_model.get_input_embeddings().weight.data = (
            tokenizer_swapped_base_model.get_input_embeddings().weight.data
        )
        ft_model.get_output_embeddings().weight.data = (
            tokenizer_swapped_base_model.get_output_embeddings().weight.data
        )
        ft_model.config.vocab_size = len(tokenizer_swapped)
        ft_model.save_pretrained(args.output + "_lambda" + str(lmd).replace(".", ""))
        tokenizer_swapped.save_pretrained(
            args.output + "_lambda" + str(lmd).replace(".", "")
        )

        for param, swap_diff in zip(ft_model.parameters(), swap_diffs):
            param.data -= swap_diff * lmd

    if args.output_ft_with_base_embeddings is not None:
        ft_model.get_input_embeddings().weight.data = (
            base_model.get_input_embeddings().weight.data
        )
        ft_model.get_output_embeddings().weight.data = (
            base_model.get_output_embeddings().weight.data
        )
        ft_model.config.vocab_size = len(tokenizer_base)
        ft_model.save_pretrained(args.output_ft_with_base_embeddings)
        tokenizer_ft.save_pretrained(args.output_ft_with_base_embeddings)
