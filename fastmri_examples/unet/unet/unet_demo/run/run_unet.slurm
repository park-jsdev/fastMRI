#!/bin/bash
#SBATCH --job-name=fastmri_unet
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=run/unet_train_%j.out
#SBATCH --error=run/unet_train_%j.err

# Load conda + activate your scratch env
module load anaconda3/2022.05.0.1
source ~/.bashrc
conda activate ~/scratch/fastmri_env

# Run training script
python ~/scratch/fastMRI/fastmri_examples/unet/train_unet_demo.py \
  --challenge singlecoil \
  --data_path ~/scratch/fastmri_data/singlecoil_knee \
  --mask_type random \
  --gpus 1 \
  --default_root_dir /home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet/unet/unet_demo
