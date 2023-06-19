#!/bin/sh
#BSUB -J 
#BSUB -o /work3/s194253/02514/project4/sh_logs/test_output_file_%J.out
#BSUB -e /work3/s194253/02514/project4/sh_errors%J.err
#BSUB -n 6
#BSUB -q gpuv100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 6:00
#BSUB -R 'rusage[mem=5GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N

nvidia-smi
module load cuda/11.8
source /work3/s194253/02514/DL-COMVIS/venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python src/models/project4/train_model.py \
    --model_name efficientnet_b4 \
    --experiment_name \
    --epochs 200 \
    --batch_size 1 \
    --use_super_categories True \
    --log_path '/work3/s194253/02514/project4_results/logs' \
    --save_path '/work3/s194253/02514/project4_results/models' \
    --seed 0 \


