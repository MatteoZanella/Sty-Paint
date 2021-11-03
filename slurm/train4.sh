#!/bin/bash
#SBATCH -p chaos -A shared-mhug-staff
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH -c 8
#SBATCH -o /data/eperuzzo/train4.out
#SBATCH -e /data/eperuzzo/train4e.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/
python train.py --exp_name our+_bs32_kl0.0025_pos1_col1_theta1_size1_ref0.1 --config /data/eperuzzo/brushstrokes-generation/configs/train/conf4.yaml & wait