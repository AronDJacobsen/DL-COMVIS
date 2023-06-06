# Hotdog / not hotdog - model training


For training a model, you can run (and modify) one of the following command:

##### Own CNN (currently dummy)
```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name test --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name test1234 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24
```

CUDA_VISIBLE_DEVICES=0,1 python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name initial --log_path /work3/s184984/repos/DL-COMVIS/logs/project1 --save_path /work3/s184984/repos/DL-COMVIS/models/project1 --seed 0 --experiment_name Adam_flip --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 50 --num_workers 24 --devices -1 --augmentation True False False --norm batchnorm

##### EfficientNet (B4)
```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name test1234 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24
```

```
CUDA_VISIBLE_DEVICES=0,1 python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/repos/DL-COMVIS/logs/project1 --save_path /work3/s194253/repos/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_None --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 50 --num_workers 24 --devices -1 --augmentation True True True
```

```
CUDA_VISIBLE_DEVICES=0,1 python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/repos/DL-COMVIS/logs/project1 --save_path /work3/s194253/repos/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.2 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 50 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.2
```



##### Predictions


```
CUDA_VISIBLE_DEVICES=1 python src/models/project1/predict_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name test --model_path logs/test1234/test/version_1/checkpoints/epoch=9_val_loss=0.7083.ckpt
```