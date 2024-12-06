#!/bin/bash
#SBATCH -J SSL_mask_50
#SBATCH --partition=ssrinath-gcondo --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH --account=ssrinath-gcondo
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -o ccv_log/mask_50.out
#SBATCH -e ccv_log/mask_50.err

module load miniconda3/23.11.0s cuda/12.2.0-4lgnkrh git-lfs ffmpeg/7.0-xny2fb2 ninja
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate ssl

cd /users/zli419/data/users/zli419/SSL/SSL_world_model
python train.py --bs 256 --num_workers 32 --mask_ratio 0.5 --exp_name mask_50