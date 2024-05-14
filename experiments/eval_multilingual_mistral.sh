target_model=mistralai/Mistral-7B-v0.1
checkpoint_path=zett-hypernetwork-multilingual-Mistral-7B-v0.1
EXTRA_ARGS="--revision=refs/pr/95"

xcopa_langs=(
    "et"
    "ht"
    "id"
    "it"
    "qu"
    "sw"
    "ta"
    "tr"
    "vi"
)
mmlu_langs=(
    "de"
    "es"
    "fr"
    "it"
    "ru"
)

for lang in ${xcopa_langs[@]}
do
    echo $lang
    tokenizer_path="datasets/tokenizers_spm/$lang-50k"

    our_target_path=output_t/$target_model-$(basename $tokenizer_path)
    focus_target_path=output_t/$target_model-$(basename $tokenizer_path)-from-focus

    ipython --pdb scripts/transfer.py -- \
        --target_model=$target_model \
        --checkpoint_path=$checkpoint_path \
        --output=$our_target_path \
        --model_class=AutoModelForCausalLM \
        --do_batching=True \
        --n_samples=4 \
        --min_k=1 \
        --lang_code=$lang \
        --lang_path=artifacts/26l.txt \
        --tokenizer_name=$tokenizer_path \
        --save_pt \
        $EXTRA_ARGS

    ipython --pdb scripts/transfer_focus.py -- \
        --target_model=$target_model \
        --model_class=AutoModelForCausalLM \
        --output=$focus_target_path \
        --tokenizer_name=$tokenizer_path \
        --save_pt \
        --lang_code=$lang \
        $EXTRA_ARGS

    ### XCOPA

    # baseline
    lm_eval --model hf \
        --model_args pretrained=$target_model,dtype="bfloat16" \
        --tasks xcopa_$lang \
        --device cuda:0 \
        --batch_size 8

    # ours
    lm_eval --model hf \
        --model_args pretrained=$our_target_path,dtype="bfloat16" \
        --tasks xcopa_$lang \
        --device cuda:0 \
        --batch_size 8

    # focus
    lm_eval --model hf \
        --model_args pretrained=$focus_target_path,dtype="bfloat16" \
        --tasks xcopa_$lang \
        --device cuda:0 \
        --batch_size 8

    remove to keep disk space in check
    rm -r $our_target_path
    rm -r $focus_target_path
done

for lang in ${mmlu_langs[@]}
do
    echo $lang
    tokenizer_path="datasets/tokenizers_spm/$lang-50k"

    our_target_path=output_t/$target_model-$(basename $tokenizer_path)
    focus_target_path=output_t/$target_model-$(basename $tokenizer_path)-from-focus

    ipython --pdb scripts/transfer.py -- \
        --target_model=$target_model \
        --checkpoint_path=$checkpoint_path \
        --output=$our_target_path \
        --model_class=AutoModelForCausalLM \
        --do_batching=True \
        --n_samples=4 \
        --min_k=1 \
        --lang_code=$lang \
        --lang_path=artifacts/26l.txt \
        --tokenizer_name=$tokenizer_path \
        --save_pt \
        $EXTRA_ARGS

    ipython --pdb scripts/transfer_focus.py -- \
        --target_model=$target_model \
        --model_class=AutoModelForCausalLM \
        --output=$focus_target_path \
        --tokenizer_name=$tokenizer_path \
        --save_pt \
        --lang_code=$lang \
        $EXTRA_ARGS

    # baseline
    lm_eval --model hf \
        --model_args pretrained=$target_model,dtype="bfloat16" \
        --tasks m_mmlu_$lang \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size 4

    # ours
    lm_eval --model hf \
        --model_args pretrained=$our_target_path,dtype="bfloat16" \
        --tasks m_mmlu_$lang \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size 4

    # focus
    lm_eval --model hf \
        --model_args pretrained=$focus_target_path,dtype="bfloat16" \
        --tasks m_mmlu_$lang \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size 4

    # # remove to keep disk space in check
    rm -r $our_target_path
    rm -r $focus_target_path
done