#!/bin/bash
#SBATCH -N 1                           # 1 node
#SBATCH -c 16                          # 16 CPU cores
#SBATCH --ntasks-per-node=1             # 1 task
#SBATCH -t 8:00:00                      # Max wall time
#SBATCH --gres=gpu:V100:2               # 2 GPUs
#SBATCH --mem=32G                       # Total memory
#SBATCH -J fastmri-unet-binary2-psnr     # New Job name (binary2-psnr)
#SBATCH -o slurm%j.out                  # Standard output
#SBATCH -e slurm%j.err                  # Standard error

# Load required modules and activate conda env
module load anaconda3/2022.05.0.1
source ~/.bashrc
conda activate ~/scratch/fastmri_env

# Navigate to your code directory
cd /home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet

# Diagnostics
echo "Running on $SLURM_NODELIST with $SLURM_CPUS_ON_NODE CPUs"
nvidia-smi

# Set custom checkpoint directory (Binary 2)
CHECKPOINT_DIR=/home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet/unet/unet_demo/checkpoints/Binary\ 2
mkdir -p "$CHECKPOINT_DIR"

# Resume checkpoint from Binary 1
RESUME_CHECKPOINT=/home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet/unet/unet_demo/checkpoints/Binary\ 1/checkpoints/epoch=21-step=764324.ckpt

# Run the training script with args
python train_unet.py \
  --mode train \
  --challenge singlecoil \
  --data_path ~/scratch/fastmri_data/singlecoil_knee \
  --mask_type random \
  --gpus 2 \
  --strategy ddp \
  --replace_sampler_ddp False \
  --roi-weighting \
  --roi-mask binary \
  --roi-margin 0.2 \
  --roi-strength 5.0 \
  --loss-type psnr \
  --max_epochs 50 \
  --log_every_n_steps 25 \
  --default_root_dir "$CHECKPOINT_DIR" \
  --resume_from_checkpoint "$RESUME_CHECKPOINT"