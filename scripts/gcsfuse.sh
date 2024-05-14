if ! command -v gcsfuse &> /dev/null; then
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

    sudo apt-get update
    sudo apt-get install gcsfuse
fi

mkdir -p bucket
gcsfuse --only-dir hypertoken --implicit-dirs trc-transfer-data bucket