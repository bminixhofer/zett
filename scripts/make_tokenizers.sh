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

mkdir -p /mnt/disks/persist/tokenizers/

for lang in $languages
do
    echo $lang
    if [[ $lang = @(yue|zh|ja|th|lo|kjg|mnw|my|shn|ksw|rki|km|bo|dz) ]]; # languages without whitespace need subsample
    then
        EXTRA_ARGS="--n_subsample=1000000"
    else
        EXTRA_ARGS=""
    fi
    python3 scripts/make_spm.py --dataset /mnt/disks/persist/train/$lang.parquet --output /mnt/disks/persist/tokenizers/$lang $EXTRA_ARGS
    rm -r ~/.cache
done