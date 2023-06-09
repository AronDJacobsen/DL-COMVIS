# Hotdog / not hotdog - model training

```
CUDA_VISIBLE_DEVICES=1 python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name test --log_path  /zhome/04/5/147207/DLinCV/DL-COMVIS/logs --save_path /zhome/04/5/147207/DLinCV/DL-COMVIS/models/project1 --seed 0 --experiment_name test1234 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24
```


saliency maps

```
CUDA_VISIBLE_DEVICES=1 python src/visualization/project1/saliency.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --model_path /work3/s194253/02514/DL-COMVIS/logs/project1/Transfer/efficientnet_b4/version_0/checkpoints/epoch=146_val_loss=0.2034.ckpt --norm none --percentage_to_freeze -1.0 --batch_size 32 
```

CUDA_VISIBLE_DEVICES=1 python src/visualization/project1/saliency.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --seed 0 --batch_size 128 --initial_lr_steps -1 --optimizer Adam --epochs 150 --num_workers 24 --model_path/work3/s194253/02514/DL-COMVIS/logs/project1/Transfer/efficientnet_b4/version_0/checkpoints/epoch=146_val_loss=0.2034.ckpt