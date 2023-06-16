from omegaconf import OmegaConf
import os

def dummy_args(args):
    # todo: oscwd, augmentation
    args_to_override = {
        'seed': 0,
        'dataset': 'waste', 
        'save_path': os.getcwd() +'/data/runs/',
        'batch_size': 2,
        'epochs': 10,
        'loss': 'CrossEntropy',
        'augmentation': '0 0 0',
        'experiment_name': 'dummy_run',
        'model_name': 'baseline',
    }
    for key, value in args_to_override.items():
        setattr(args, key, value)
    return args



