#!/bin/bash

OVERWRITE_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --overwrite)
            OVERWRITE_FLAG="--overwrite"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

python src/data_synthesis/sft_generator.py $OVERWRITE_FLAG > logs/sft_generator.log 2>&1