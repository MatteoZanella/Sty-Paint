#!/bin/bash
#SBATCH -p gpupart
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH -o /data/eperuzzo/stout_sort.out
#SBATCH -e /data/eperuzzo/errors_sort.out
#SBATCH --signal=B:SIGTERM@120

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate brush

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT
cd /data/eperuzzo/brushstrokes-generation/code/dataset/
python main_sorting.py --csv_file /data/eperuzzo/todi3.csv --data_path /data/eperuzzo/decomposition_output_v1/ --output_path /data/eperuzzo/sorting_ade_v4/ & wait