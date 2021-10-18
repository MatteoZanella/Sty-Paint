#!/bin/bash
#SBATCH -p chaos -A shared-mhug-staff
#SBATCH --gres gpu:2
#SBATCH --mem-per-cpu=4000
#SBATCH -c 8
#SBATCH -o /data/eperuzzo/train_stout.out
#SBATCH -e /data/eperuzzo/train_errors.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/
python evaluate_metrics.py --checkpoint_path /data/eperuzzo/model_checkpoints/full_big_v2/latest.pth.tar --config /data/eperuzzo/brushstrokes-generation/configs/train/todi_config.yaml & wait