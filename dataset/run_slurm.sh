#!/bin/bash
#SBATCH -p gpupart
#SBATCH --gres gpu:1
#SBATCH -o /data/eperuzzo/stout.out
#SBATCH -e /data/eperuzzo/errors.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/code/dataset/
python main_render.py --dataset_path /data/eperuzzo/generative_brushstorkes_ade_v2 --index_path /data/eperuzzo/ade_strokes_v2/brushstrokes-sorting/ --strokes_path /data/eperuzzo/ade_strokes_v2/brushstrokes-decomposition & wait