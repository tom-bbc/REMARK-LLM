#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --load_path  ./logs_text_wm/debug \
    --dataset "NicolaiSivesind/ChatGPT-Research-Abstracts" \
    --save_path  ./logs_text_wm/debug \
    --model_path t5-base \
    --input_max_length 80 \
    --target_max_length 80 \
    --mask_per 0.3 \
    --beam_width 5 \
    --repeat 10 \
    --message_max_length 16
