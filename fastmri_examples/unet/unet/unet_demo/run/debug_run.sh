#!/bin/bash

python fastmri_examples/unet/train_unet.py \
  --mode train \
  --challenge singlecoil \
  --data_path ~/scratch/fastmri_data/singlecoil_knee \
  --mask_type random \
  --max_epochs 1 \
  --limit_train_batches 2 \
  --limit_val_batches 2 \
  --loss_type psnr \
  --roi-weighting False \
  --roi-mask gaussian \
  --roi-margin 0.2 \
  --roi-strength 5.0 \
  --gpus 1 \
  --strategy ddp \
  --replace_sampler_ddp False