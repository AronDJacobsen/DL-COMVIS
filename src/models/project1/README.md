# Hotdog / not hotdog - model training


For training a model, you can run (and modify) one of the following command:

##### Own CNN (currently dummy)
```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name test --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name test1234 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24
```

CUDA_VISIBLE_DEVICES=0,1 python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name initial --log_path /work3/s184984/repos/DL-COMVIS/logs/project1 --save_path /work3/s184984/repos/DL-COMVIS/models/project1 --seed 0 --experiment_name Adam_flip --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 50 --num_workers 24 --devices -1 --augmentation True False False --norm batchnorm


python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name initial --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name FINALLY --log_every_n 2 --batch_size 128 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 100 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation 0 0 0 --norm none --lr 1e-04


##### EfficientNet (B4)

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name Transfer_None --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation 1 0 0
```

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.0 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.0
```

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.2 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.2
```

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.4 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.4
```

``` 
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.6 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.6
```

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.8 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.8
```

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.9 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.9
```

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name transfer_0.95 --log_every_n 2 --batch_size 32 --min_lr 0.00000001 --max_lr 0.1 --initial_lr_steps 1000 --optimizer Adam --epochs 250 --num_workers 24 --devices -1 --augmentation True True True --percentage_to_freeze 0.95
```


##### Predictions


```
CUDA_VISIBLE_DEVICES=1 python src/models/project1/predict_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --model_path /work3/s194253/02514/DL-COMVIS/logs/project1/transfer_0.0/efficientnet_b4/version_0/checkpoints/epoch=46_val_loss=0.1741.ckpt 
```