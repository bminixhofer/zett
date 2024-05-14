lm_eval --model hf \
    --model_args pretrained=output_t/TinyLlama-1.1B-intermediate-step-1431k-3T-gpt2,dtype="bfloat16" \
    --tasks piqa,hellaswag,arc_easy,boolq,mmlu \
    --device cuda:0 \
    --batch_size 8