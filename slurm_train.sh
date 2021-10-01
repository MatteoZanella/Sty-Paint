#!/bin/bash
#SBATCH -p gpupart
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH -c 8
#SBATCH -o /data/eperuzzo/brushstrokes-generation/train_stout.out
#SBATCH -e /data/eperuzzo/brushstrokes-generation/train_errors.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/code/
python train.py --exp_name cat_1e-3_todi_seq_4_ctx_8 --config ./configs/train/todi_config.yaml & wait
