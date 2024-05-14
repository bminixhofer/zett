eval() {
  local pattern=$1
  local adapter_path=$2
  local name=$3

  out_dir=output_eval_results

  # iterate over xnli languages
  for lang in ar bg de el en es fr hi ru sw tr ur vi
  do
    echo $lang
    result_file=$out_dir/eval_results_${name}_${lang}.json

    if [ -f $result_file ]; then
      echo "Skipping $name for $lang"
      continue
    fi

    ipython --pdb run_glue.py -- \
      --model_name_or_path $(printf $pattern $lang) \
      --dataset_name xnli \
      --dataset_config_name $lang \
      --load_adapter $adapter_path \
      --output_dir $out_dir \
      --do_eval \
      --do_predict \
      --evaluation_strategy=epoch \
      --save_strategy=epoch \
      --per_device_train_batch_size 32 \
      --overwrite_output_dir \
      --train_adapter \
      --report_to=none
    mv output_eval_results/eval_results.json $result_file
  done
}

eval "xlm-roberta-base" "output_peft_first/xnli_en/glue" "baseline"
eval "output_t/xlm-roberta-base-%s" "output_peft_first/xnli_en/glue" "ours"
eval "output_t/xlm-roberta-base-%s-from-focus" "output_peft_first/xnli_en/glue" "focus"
eval "output_t/xlm-roberta-base-%s-from-lexical" "output_peft_first/xnli_en/glue" "lexical"
eval "output_t/xlm-roberta-base-%s-from-lexical-with-fvt" "output_peft_first/xnli_en/glue" "fvt"
eval "output_t/xlm-roberta-base-%s-from-ofa" "output_peft_first/xnli_en/glue" "ofa"
