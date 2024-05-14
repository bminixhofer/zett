train() {
  local pattern=$1
  local out_prefix=$2

  mkdir -p $out_prefix

  python run_glue.py \
    --model_name_or_path $(printf $pattern "en") \
    --dataset_name=xnli \
    --dataset_config_name=en \
    --do_train \
    --do_eval \
    --log_level=info \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 5.0 \
    --output_dir $out_prefix/xnli_en \
    --overwrite_output_dir \
    --bf16 \
    --train_adapter \
    --report_to=none \
    --adapter_config seq_bn
}

train "xlm-roberta-base" "output_peft_first"