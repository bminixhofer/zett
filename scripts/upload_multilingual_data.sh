#!/bin/bash

read_languages_from_file() {
    local file=$1
    local languages=()
    while IFS= read -r line; do
        languages+=("$line")
    done < "$file"
    echo "${languages[@]}"
}

if [[ $1 == *.txt ]]; then
    # If the argument is a .txt file, read the languages from the file
    languages=$(read_languages_from_file "$1")
else
    # If the argument is not a .txt file, treat it as the list of languages
    languages=$1
fi

for lang in $languages
do
    gsutil -m cp /mnt/disks/persist/train/$lang.parquet gs://trc-transfer-data/hypertoken/datasets/multilingual/train/
    gsutil -m cp /mnt/disks/persist/valid/$lang.parquet gs://trc-transfer-data/hypertoken/datasets/multilingual/valid/
    gsutil -m cp -r /mnt/disks/persist/large_tokenizers/$lang gs://trc-transfer-data/hypertoken/datasets/multilingual/large_tokenizers/
done