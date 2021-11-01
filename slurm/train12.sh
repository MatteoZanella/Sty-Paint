#!/bin/bash
#SBATCH -p chaos -A shared-mhug-staff
#SBATCH --gres gpu:2
#SBATCH --mem-per-cpu=5000
#SBATCH -c 8
#SBATCH -o /data/eperuzzo/train12.out
#SBATCH -e /data/eperuzzo/train12e.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/
python train.py --exp_name our+_bs64_lr0.0001_kl0.00025 --config /data/eperuzzo/brushstrokes-generation/configs/train/conf12.yaml & wait