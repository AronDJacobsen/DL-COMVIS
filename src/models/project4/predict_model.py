import argparse
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import sys
import json
import os
#sys.path.append('../../')

from src.utils import set_seed, get_optimizer
from src.models.project2.models import get_model
from src.models.project2.losses import get_loss
from src.data.project2.dataloader import get_loaders#, get_normalization_constants


class BooleanListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Convert the string values to booleans
        bool_values = [bool(int(v)) for v in values]
        setattr(namespace, self.dest, bool_values)    

# CUDA_VISIBLE_DEVICES=1 python src/models/project2/predict_model.py --dataset DRIVE_TEST --model_name UNet  --log_path /work3/$USER/02514/DL-COMVIS/logs/project2/DRIVE --save_path /work3/$USER/02514/DL-COMVIS/models/project2/DRIVE  --batch_size 6 --lr 0.001 --initial_lr_steps -1 --optimizer Adam --epochs 1 --num_workers 24 --devices -1 --experiment_name DRIVE_TEST --augmentation 1 1 --loss BCE  --model UNet --reg none --reg_coef 0 --path_model /work3/s184984/02514/DL-COMVIS/model/project2/DRIVE_UNet_lr=0.001_BCE_AUG/UNet_fold4.ckpt --out 1

def parse_arguments():

    
    parser = argparse.ArgumentParser()

    # GENERAL ()
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo-randomness.")
    parser.add_argument("--dataset", type=str, default='PH2',
                        help="Data set either (PH2 or DRIVE, or DRIVE_TEST).")
    parser.add_argument("--log_path", type=str, default = 'lightning_logs',
                        help="Path determining where to store logs.")
    parser.add_argument("--log_every_n", type=int, default=1,
                        help="Logging interval.")
    parser.add_argument("--save_path", type=str,
                        help="Path determining where to store results.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="Determines console logging.")
    parser.add_argument("--devices", type=int, default=2, 
                        help="Number of devices"),
    
    parser.add_argument("--out", type=bool, default=False,
                        help="output individual predicted images")
    parser.add_argument("--path_model", type=str, default='None')
                        
    # TRAINING PARAMETERS
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training the model.")
    parser.add_argument("--lr", type=float, default=1e-04,
                        help="Learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-08,
                        help="Minimum allowed learning rater.")
    parser.add_argument("--max_lr", type=float, default=1,
                        help="Maximum allowed learning rate.")
    parser.add_argument("--initial_lr_steps", type=int, default=-1,
                        help="Number of initial steps for finding learning rate, -1 to deactivate.")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="The optimizer to be used.")
    parser.add_argument("--loss", type=str, default = 'BCE',
                        help="Loss function - one of: [BSE, FOCAL, DICE]")
    parser.add_argument("--reg", type=str, default = 'sparsity',
                        help="Regularization - one of: [none, centered, sparsity, tv")
    parser.add_argument("--reg_coef", type=float, default = 0.1,
                        help="Regularization coefficient")
    parser.add_argument('--augmentation', nargs='+', action=BooleanListAction, 
                        help='List of booleans, i.e. [flip, rotation]')
    
    # EXPERIMENT NAMING
    parser.add_argument("--experiment_name", type=str, default='test',
                        help="Sets the overall experiment name.")
    
    # MODEL BASED
    parser.add_argument("--model_name", type=str, default='SegCNN',
                        help="Model name - either 'SegCNN' or ...")
    #parser.add_argument("--norm", type=str, default = 'none',
    #                    help="Batch normalization - one of: [none, batchnorm, layernorm, instancenorm]")

    return parser.parse_args()
    


def test(args):
    # Set random seed
    set_seed(args.seed)

    # Get functions
    loss_fun = get_loss(args.loss, args.reg, args.reg_coef)
    optimizer = get_optimizer(args.optimizer)

    # Get data loaders with applied transformations
    loaders = get_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        seed=args.seed, 
        num_workers=args.num_workers,
        augmentations={'rotate': args.augmentation[0], 'flip': args.augmentation[1]}
    )

    # Load model
    model = get_model(args.model_name, args, loss_fun, optimizer, 0, out=args.out)

    model = model.load_from_checkpoint(args.path_model)
    model.eval()
    model.out = args.out

    # Set up logger
    tb_logger = None

    # Setup trainer
    trainer = pl.Trainer(
        devices=args.devices, 
        accelerator='gpu', 
        max_epochs = args.epochs,
        log_every_n_steps = args.log_every_n,
        callbacks=[model.model_checkpoint] if args.initial_lr_steps == -1 else [model.model_checkpoint, model.lr_finder],
        logger=tb_logger,
    )
    print(model.out)
    
    trainer.predict(model, dataloaders=loaders['test'] if args.dataset == 'PH2' else loaders['test'], ckpt_path = args.path_model)


if __name__ == '__main__':

    # Get input arguments
    args = parse_arguments()
    
    # Train model
    test(args)

