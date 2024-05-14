<h1 align="center">Zero-Shot Tokenizer Transfer</h1>
<p align="center">
<img src=https://github.com/bminixhofer/zett/assets/13353204/f64dbdf4-da05-4586-8eb6-bf5a622b2160 width=300px>
</P>

This repository contains the code for the paper [Zero-Shot Tokenizer Transfer](https://arxiv.org/abs/2405.07883). ZeTT frees language models from their tokenizer, allowing you to use any model with any tokenizer, with little or no extra training⚡

## Available pretrained hypernetworks

| Hypernetwork                                                                                                                                                        | ..for Model                                                                                                                       | Comments                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| [benjamin/zett-hypernetwork-xlm-roberta-base](https://huggingface.co/benjamin/zett-hypernetwork-xlm-roberta-base)                                                   | [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)                                                            | multilingual, 26 languages |
| [benjamin/zett-hypernetwork-Mistral-7B-v0.1](https://huggingface.co/benjamin/zett-hypernetwork-Mistral-7B-v0.1)                                                     | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)                                                     | English + Code             |
| [benjamin/zett-hypernetwork-multilingual-Mistral-7B-v0.1](https://huggingface.co/benjamin/zett-hypernetwork-multilingual-Mistral-7B-v0.1)                           | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)                                                     | multilingual, 26 languages |
| [benjamin/zett-hypernetwork-TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/benjamin/zett-hypernetwork-TinyLlama-1.1B-intermediate-step-1431k-3T) | [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) | English + Code             |

## Using a pretrained hypernetwork


### Environment Setup

Requirements are in `requirements.txt`, This, for example, creates a working environment:

```
conda create -n zett Python=3.11
conda activate zett

pip install -r requirements.txt
pip install -U "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # adjust based on your CUDA version
pip install -e .
```

### Transferring to a new tokenizer

<details open>
    <summary>Let's transfer XLM-RoBERTa to the GPT2 tokenizer.</summary>

```bash
git clone https://huggingface.co/benjamin/zett-hypernetwork-xlm-roberta-base

python3 scripts/transfer.py \
    --target_model=FacebookAI/xlm-roberta-base \
    --tokenizer_name=gpt2 \
    --output=my-new-fancy-xlm-r \
    --model_class=AutoModelForMaskedLM \
    --lang_code=en \
    --checkpoint_path=zett-hypernetwork-xlm-roberta-base \
    --save_pt # otherwise saves only Flax weights
```

Tada!

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my-new-fancy-xlm-r")
model = AutoModelForMaskedLM.from_pretrained("my-new-fancy-xlm-r")

out = model(**tokenizer("Hello world!", return_tensors="pt"))
```
</details>

<details>
<summary>..or Mistral-7B to the GPT-NeoX tokenizer:</summary>

```bash
git clone https://huggingface.co/benjamin/zett-hypernetwork-Mistral-7B-v0.1

# because Flax weights are not merged in the main branch, we need to specify the revision of a PR containing Flax weights
python3 scripts/transfer.py \
    --target_model=mistralai/Mistral-7B-v0.1 \
    --revision=refs/pr/95 \
    --tokenizer_name=EleutherAI/gpt-neox-20b \
    --output=my-new-fancy-mistral \
    --model_class=AutoModelForCausalLM \
    --checkpoint_path=zett-hypernetwork-Mistral-7B-v0.1 \
    --save_pt # otherwise saves only Flax weights
```

```python
from transformers import AutoModelForCaualLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my-new-fancy-mistral")
model = AutoModelForCaualLM.from_pretrained("my-new-fancy-mistral")

out = model(**tokenizer("Hello world!", return_tensors="pt"))
```
</details>

Although the codebase is in Jax/Flax, there are Pytorch bindings for the model in `./hf_hypernet`. You can use them as follows:

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from zett.utils import get_surface_form_matrix

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
hypernet = AutoModel.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1", trust_remote_code=True)

source_embeddings = torch.concatenate([
    base_model.get_input_embeddings().weight.data,
    base_model.get_output_embeddings().weight.data,
], axis=1)

hn_tokenizer = AutoTokenizer.from_pretrained("benjamin/zett-hypernetwork-Mistral-7B-v0.1")

target_surface_forms = get_surface_form_matrix(
    ["Ġhello", "Ġworld"], # byte representation of the tokens to predict
    maxlen=hypernet.config.hn_surface_maxlen,
    tokenizer_to_use=hn_tokenizer,
)[0]

# the last output is the predicted bias in case the model uses a bias (e.g. XLM-R)
predicted_input_embeddings, predicted_output_embeddings, _ = hypernet(
    torch.from_numpy(target_surface_forms),
    source_embeddings=source_embeddings
)

```

but `transfer.py` is currently not ported to PyTorch (PRs welcome!).

## Advanced usage

### Training a Hypernetwork

The script used to train the hypernetwork is `train.py`. 

But first, you'll need to download and prepare the data via `data/prepare.py` and `data/prepare_code.py`.

You'll also need to install the Rust module in `rust_utils` (used to quickly sample tokenizers) via e.g. `cd rust_utils && maturin develop --release`.

Once finished, you can run training using the configs in `configs/`. For example:

```bash
python3 train.py configs/zeroshot/v7:tinyllama_en+code:lw=0.5_long.json
```

to train a hypernetwork for TinyLlama on English and Code.

### Transferring fine-tuned models to a new tokenizer using a base model hypernetwork

Use `scripts/apply_to_ft.py` to transfer the tokenizers of a fine-tuned model, given a base model with already transferred tokenizer. For example:

```bash
python3 scripts/apply_to_ft.py \
    --output=transferred-chat-mistral \
    --base_model_path=mistralai/Mistral-7B-v0.1 \
    --ft_model_path=mistralai/Mistral-7B-Instruct-v0.1 \
    --tokenizer_swapped_base_model_path=path-to-base-model-with-new-tokenizer \
    --lambdas 0.5 \
```

### Reproducing the experiments from the paper

There are bash scripts in `experiments/` to allow reproducing the main results from the paper.

Evaluation on code is still missing because we are using a fork of `bigcode-evaluation-harness` to fix some issues we encountered. They will be added soon.

### Unigramifying, using n-shot transferred models, reproducing the tokenizers from the paper, etc.

Guide coming soon... (but feel free to dig into `scripts/` in the meantime)

### 

## Disclaimer

I prioritized releasing the code quickly instead of making it perfectly clean. There may still be remnants of my personal environment used to train the models and other non-niceties. I am in the process of cleaning this up. If you run into any problems or have any questions, please open an issue.
