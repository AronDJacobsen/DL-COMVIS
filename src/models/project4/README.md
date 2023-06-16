# Waste detection training

For training a model, you can run (and modify) one of the following command:

```
python src/models/project4/train_model.py --dataset waste --model_name efficientnet_b4 --log_path /work3/s194235/02514/DL-COMVIS/logs/project4 --save_path /work3/s194235/02514/DL-COMVIS/models/project4 --seed 0 --experiment_name test1234 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24 --augmentation 0 0
```

```

CUDA_VISIBLE_DEVICES=1 python src/models/project4/predict_model.py --dataset waste --model_name efficientnet_b4 --path_model /work3/s194253/02514/DL-COMVIS/logs/project1/transfer_0.0/efficientnet_b4/version_0/checkpoints/epoch=46_val_loss=0.1741.ckpt --augmentation 0 0 --out 1
```