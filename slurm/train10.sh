#!/bin/bash
#SBATCH -p chaos -A shared-mhug-staff
#SBATCH --gres gpu:2
#SBATCH --mem-per-cpu=5000
#SBATCH -c 8
#SBATCH -o /data/eperuzzo/train10.out
#SBATCH -e /data/eperuzzo/train10e.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/
python train.py --exp_name 10-our --config /data/eperuzzo/brushstrokes-generation/configs/train/conf10.yaml & wait