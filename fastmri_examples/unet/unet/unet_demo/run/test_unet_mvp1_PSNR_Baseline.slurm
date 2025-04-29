#!/bin/bash
#SBATCH -N 1                          # 1 node
#SBATCH -c 16                         # 16 CPU cores
#SBATCH --ntasks-per-node=1          # 1 task
#SBATCH -t 8:00:00                    # Max wall time
#SBATCH --gres=gpu:V100:2            # 2 GPUs (adjust as needed)
#SBATCH --mem=32G                     # Total memory
#SBATCH -J psnr-unet-baseline-test        # Job name
#SBATCH -o slurm%j.out               # Standard output
#SBATCH -e slurm%j.err               # Standard error

# Load env
module load anaconda3/2022.05.0.1
source ~/.bashrc
conda activate ~/scratch/fastmri_env

# Go to code directory
cd /home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet

# Diagnostics
echo "Running on $SLURM_NODELIST with $SLURM_CPUS_ON_NODE CPUs"
nvidia-smi

# Run training from epoch 0 with PSNR
python train_unet.py \
  --mode test \
  --test_split test \
  --challenge singlecoil \
  --data_path ~/scratch/fastmri_data/singlecoil_knee \
  --resume_from_checkpoint ~/scratch/fastMRI/fastmri_examples/unet/unet/unet_demo/checkpoints/PSNR Unet Baseline/epoch=40-step=1424422.ckpt

