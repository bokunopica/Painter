#!/bin/bash
python /home/qianq/mycodes/Painter/SegGPT/SegGPT_train/main_train.py \
    --batch_size 1 \
    --accum_iter 16  \
    --ckpt_path /home/qianq/mycodes/Painter/SegGPT/SegGPT_inference/pretrained_seggpt/seggpt_vit_large.pth \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 300 \
    --warmup_epochs 1 \
    --lr 1e-3 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --save_freq 100 \
    --data_path /run/media/breastCancer/processed \
    --json_path  \
    /run/media/breastCancer/processed/meta_train.json \
    --val_json_path \
    /run/media/breastCancer/processed/meta_test.json \
    --output_dir output_dir/seggpt-finetune-breastCancer \
    --log_dir output_dir/logs
    # --log_wandb \

