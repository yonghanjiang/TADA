#!/bin/bash
# CUDA_LAUNCH_BLOCKING=1 
set -e

python -u main.py \
    --config-file test/test_odinw13 \
    --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_tada.py \
    --model-checkpoint-path ./weights/groundingdino_swint_ogc.pth \
    --output-dir output/tada \
    --seed 0 --num-gpus 1