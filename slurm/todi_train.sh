#!/bin/bash
#SBATCH -p gpupart
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH -c 8
#SBATCH -o /data/eperuzzo/brush_std.out
#SBATCH -e /data/eperuzzo/brush_errors.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/
python train.py --exp_name small_v4 --config /data/eperuzzo/brushstrokes-generation/configs/train/todi_config.yaml & wait