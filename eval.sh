#!/bin/bash

# Check if path argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    echo "Example: $0 /path/to/your/model"
    exit 1
fi

path="$1"


HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --mixed_precision=bf16 eval/harness.py \
    --model hf \
    --model_args "pretrained=$path,dtype=bfloat16,trust_remote_code=True" \
    --tasks ruler \
    --batch_size 16 \
    --device cuda \
    --metadata='{"max_seq_lengths":[4096]}' \
    --seed 0

export PYTHONPATH=./:../

