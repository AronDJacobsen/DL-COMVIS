# Hotdog / not hotdog - model training


For training a model, you can run (and modify) one of the following command:

##### Own CNN (currently dummy)
```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name test --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name test1234 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24
```

```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name test --log_path /work3/s184984/repos/DL-COMVIS/logs/project1 --save_path /work3/s184984/repos/DL-COMVIS/models/project1 --seed 0 --experiment_name Phil1 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24
```
##### EfficientNet (B4)
```
python src/models/project1/train_model.py --data_path /dtu/datasets1/02514/hotdog_nothotdog/ --network_name efficientnet_b4 --log_path /work3/s194253/02514/DL-COMVIS/logs/project1 --save_path /work3/s194253/02514/DL-COMVIS/models/project1 --seed 0 --experiment_name test1234 --log_every_n 2 --batch_size 32 --lr 0.001 --optimizer Adam --epochs 10 --num_workers 24
```