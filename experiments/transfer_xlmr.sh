langs=(
    "ar"
    "bg"
    "de"
    "el"
    "en"
    "es"
    "fr"
    "hi"
    "ru"
    "sw"
    "tr"
    "ur"
    "vi"
)
checkpoint_path=output_models/v7_xlmr_multilingual_26l

for lang in ${langs[@]}
do
    echo $lang
    tokenizer_size="50k"
    tokenizer_name="datasets/tokenizers_spm/$lang-$tokenizer_size"

    ipython --pdb zett/transfer.py -- \
        --checkpoint_path=$checkpoint_path \
        --tokenizer_name=$tokenizer_name \
        --model_class=AutoModelForMaskedLM \
        --target_model=xlm-roberta-base \
        --output=output_t/xlm-roberta-base-$lang \
        --save_pt \
        --lang_path=artifacts/26l.txt \
        --lang_code=$lang \

    python3 scripts/transfer_focus.py --target_model xlm-roberta-base --output output_t/xlm-roberta-base-$lang-from-focus --tokenizer_name $tokenizer_name --lang_code $lang --save_pt
    python3 scripts/transfer_lexical.py --output output_t/xlm-roberta-base-$lang-from-lexical --tokenizer_name=$tokenizer_name
    python3 scripts/transfer_lexical.py --output output_t/xlm-roberta-base-$lang-from-lexical-with-fvt --tokenizer_name=$tokenizer_name --fvt_mode=fvt
    python3 scripts/transfer_ofa.py --target_model xlm-roberta-base --output output_t/xlm-roberta-base-$lang-from-ofa  --tokenizer_name $tokenizer_name
done