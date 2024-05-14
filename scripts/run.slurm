#!/bin/bash

git pull

echo "Running on `hostname`..."
python3 $1 $2

exit 0