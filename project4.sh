#!/bin/sh
#BSUB -J albertkjoller_efficientnet_v2
#BSUB -o /work3/s194253/02514/project4_results/sh_logs/albertkjoller_efficientnet_v2.out
#BSUB -e /work3/s194253/02514/project4_results/sh_errors/albertkjoller_efficientnet_v2.err
#BSUB -n 6
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 24:00
#BSUB -R 'rusage[mem=30GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N

nvidia-smi
module load cuda/11.8
source /work3/s194253/02514/DL-COMVIS/venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python src/models/project4/train_model.py \
    --model_name efficientnet_b4 \
    --experiment_name albertkjoller_efficientnet \
    --data_path '/work3/s194253/02514/project4_results/data_wastedetection' \
    --epochs 200 \
    --batch_size 1 \
    --use_super_categories True \
    --seed 0 \


