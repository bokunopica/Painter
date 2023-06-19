#!/bin/bash
python SegGPT/SegGPT_train/main_train.py \
    --batch_size 2 \
    --accum_iter 16  \
    --ckpt_path /home/pico/myCodes/Painter/SegGPT/SegGPT_inference/pretrained_seggpt/seggpt_vit_large.pth \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 1e-3 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --save_freq 1 \
    --data_path /mnt/c/data/breastCancer/processed \
    --json_path  \
    /mnt/c/data/breastCancer/processed/meta.json \
    --val_json_path \
    /mnt/c/data/breastCancer/processed/meta.json \
    --output_dir output_dir/seggpt-finetune \
    --log_dir output_dir/logs \
    # --log_wandb \

