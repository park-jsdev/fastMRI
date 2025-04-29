#!/bin/bash
#SBATCH -N 1                           # 1 node
#SBATCH -c 16                          # 16 CPU cores
#SBATCH --ntasks-per-node=1             # 1 task
#SBATCH -t 8:00:00                      # Max wall time
#SBATCH --gres=gpu:V100:2               # 2 GPUs
#SBATCH --mem=32G                       # Total memory
#SBATCH -J adaptive-varnet-acc4         # Job name
#SBATCH -o slurm%j.out                  # Standard output
#SBATCH -e slurm%j.err                  # Standard error

# Load environment
module load anaconda3/2022.05.0.1
source ~/.bashrc
conda activate ~/scratch/fastmri_env

# Navigate to your code directory
cd /home/hice1/lburk3/scratch/fastMRI/fastmri_examples/adaptive_varnet

# Diagnostics
echo "Running on $SLURM_NODELIST with $SLURM_CPUS_ON_NODE CPUs"
nvidia-smi

# Set output directory
OUTPUT_DIR=/home/hice1/lburk3/scratch/fastMRI/fastmri_examples/adaptive_varnet/checkpoints/adaptive_acc4_run1
mkdir -p $OUTPUT_DIR

# Run the training
python train_adaptive_varnet_demo.py \
  --mode train \
  --data_path ~/scratch/fastmri_data/singlecoil_knee \
  --default_root_dir $OUTPUT_DIR \
  --seed 42 \
  --batch_size 16 \
  --num_cascades 5 \
  --accelerations 4 \
  --center_fractions 0.08 \
  --num_sense_lines 10 \
  --budget 22 \
  --learn_acquisition True \
  --loupe_mask False \
  --use_softplus False \
  --slope 10
