export HF_DATASETS_CACHE=/mnt/disks/persist/cache

unigramify() {
    local tokenizer_name=$1
    local name=$2
    local extra_args=$3

    ipython --pdb -i scripts/unigramify.py -- \
        --output=artifacts/madlad/${name}0001 \
        --tokenizer_name=${tokenizer_name} \
        --train_dataset_names /mnt/disks/persist/train/en /mnt/disks/persist/train/java /mnt/disks/persist/train/javascript /mnt/disks/persist/train/github-issues-filtered-structured /mnt/disks/persist/train/go /mnt/disks/persist/train/cpp /mnt/disks/persist/train/python \
        --valid_dataset_names datasets/valid/en.parquet datasets/valid/java.parquet datasets/valid/javascript.parquet datasets/valid/github-issues-filtered-structured.parquet datasets/valid/go.parquet datasets/valid/cpp.parquet datasets/valid/python.parquet \
        --regularization_strength=0.001 \
        --max_n_train_pretokens=1000000 \
        $extra_args
}

# unigramify meta-llama/Llama-2-7b-hf llama
# unigramify mistralai/Mistral-7B-v0.1 mistral
unigramify gpt2 gpt2 "--keep_pretokenizer --keep_normalizer"
# unigramify roberta-base roberta
# unigramify bert-base-cased bert "--keep_pretokenizer --keep_normalizer"
# unigramify bigcode/starcoder starcoder

# bert
# Original accuracy: 99.6123%
# Unigramified accuracy: 99.44%
# Avg. logp diff: 0.2511

# roberta
# Original accuracy: 100.0000%                                                                                                                                                                   
# Unigramified accuracy: 99.0204%                                                                                                                                                                
# Avg. logp diff: 0.2776

# gpt2
# Original accuracy: 100.0000%                                                                                                                                                                   
# Unigramified accuracy: 99.0180%                                                                                                                                                                
# Avg. logp diff: 0.277

# mistral
# Original accuracy: 99.9767%                                                                                                                                                                    
# Unigramified accuracy: 99.81%                                                                                                                                                                
# Avg. logp diff: 0.5758

# llama
# Original accuracy: 99.9261%                                                                                                                                                                    
# Unigramified accuracy: 99.8%                                                                                                                                                                
# Avg. logp diff: 0.5918