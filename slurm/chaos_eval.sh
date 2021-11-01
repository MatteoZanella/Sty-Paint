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
<<<<<<< HEAD
python evaluate_metrics.py --checkpoint_path /data/eperuzzo/model_checkpoints/full_big_v2/latest.pth.tar --config /data/eperuzzo/brushstrokes-generation/configs/train/todi_config.yaml & wait
=======
python evaluate_metrics.py \
        --model_1_config /data/eperuzzo/model_checkpoints/full_clean_as_oxford/latest.pth.tar \
        --model_2_config /data/eperuzzo/model_checkpoints/2_steps_clean_as_oxford/latest.pth.tar \
        --output_path /data/eperuzzo/metrics_v2 \
        --checkpoint_baseline /data/eperuzzo/model_checkpoints/painttransformer/paint_best.pdparams \
        --n_iters_dataloader 3 \
        --config /data/eperuzzo/brushstrokes-generation/configs/train/todi_config.yaml & wait
>>>>>>> ablation
