for lang in `cat data/madlad400_metadata.csv | tail -n +2 | cut -d, -f1`
do
    echo $lang
    if [[ $lang = @(yue|zh|ja|th|lo|kjg|mnw|my|shn|ksw|rki|km|bo|dz) ]]; # languages without whitespace need subsample
    then
        EXTRA_ARGS="--n_subsample=1000000"
    else
        EXTRA_ARGS=""
    fi
    python3 scripts/make_large_spm.py --dataset /mnt/disks/persist/train/$lang.parquet --output /mnt/disks/persist/large_tokenizers/$lang $EXTRA_ARGS
    rm -r ~/.cache
done
