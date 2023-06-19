import argparse
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import json
import os

from src.utils import set_seed, get_optimizer
from src.models.project4.models import get_model
from src.models.project4.losses import get_loss
from src.data.project4.dataloader import get_loaders 
from dummy_args import dummy_args

class BooleanListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Convert the string values to booleans
        bool_values = [bool(int(v)) for v in values]
        setattr(namespace, self.dest, bool_values)    

# see how to call this script in the README.md

def parse_arguments():

    parser = argparse.ArgumentParser()

    # GENERAL ()
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo-randomness.")
    parser.add_argument("--dataset", type=str, default='waste',
                        help="Data set either")
    parser.add_argument("--use_super_categories", type=bool, default=True,
                        help="Whether to use 60 categories or 28 less fine-grained super categories")
    parser.add_argument("--log_path", type=str, default='/work3/s194253/02514/project4_results/logs',
                        help="Path determining where to store logs.")
    parser.add_argument("--log_every_n", type=int, default=1,
                        help="Logging interval.")
    parser.add_argument("--save_path", type=str, default='/work3/s194253/02514/project4_results/models',
                        help="Path determining where to store results.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="Determines console logging.")
    parser.add_argument("--devices", type=int, default=1, 
                        help="Number of devices"),
    parser.add_argument("--data_path", type=str, default="/dtu/datasets1/02514/data_wastedetection", 
                        help="Path to dataset"),
    
    parser.add_argument("--out", type=bool, default=False,
                        help="output individual predicted images")
                        
    # TRAINING PARAMETERS
    parser.add_argument("--region_size", type=int, default=224,
                        help="Size of bbox images for training.")
    parser.add_argument("--batch_size", type=int, default=8,
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
                        help="Loss function - one of: [CrossEntropy,BCE]")
    parser.add_argument('--augmentation', nargs='+', action=BooleanListAction, 
                        help='List of booleans, i.e. [flip, rotation]')
    
    # EXPERIMENT NAMING
    parser.add_argument("--experiment_name", type=str, default='test',
                        help="Sets the overall experiment name.")
    
    # MODEL BASED
    parser.add_argument("--model_name", type=str, default='testnet',
                        help="Model name - either 'efficientnet_b4' or ...")
    parser.add_argument("--percentage_to_freeze", type=float, default=None,
                        help="Percentage to freeze (transfer learning) in [0, 1]")

    return parser.parse_args()


def train(args):
    # Set random seed
    set_seed(args.seed)

    # Get functions
    loss_fun = get_loss(args.loss)
    optimizer = get_optimizer(args.optimizer)

    # Get normalization constants
    # TODO:
    #train_mean, train_std = get_normalization_constants(root=args.data_path, seed=args.seed)


    # Get data loaders with applied transformations
    loaders, num_classes = get_loaders(
        dataset='waste', 
        batch_size=args.batch_size, 
        seed=args.seed, 
        num_workers=args.num_workers,
        img_size = (512, 512),
        region_size = (args.region_size, args.region_size),
        use_super_categories=args.use_super_categories,
        root = args.data_path,
        #augmentations={'rotate': args.augmentation[0], 'flip': args.augmentation[1]},
    )

    # Load model
    model = get_model(args.model_name, args, loss_fun, optimizer, out=args.out, num_classes=num_classes, region_size=(args.region_size, args.region_size), id2cat=loaders['train'].dataset.id2cat)


    # Set up logger
    tb_logger = TensorBoardLogger(
        save_dir=f"{args.log_path}/{args.experiment_name}/{args.model_name}",
        version=None,
        name=args.model_name,
    )

    acc = "gpu" if torch.cuda.is_available() else "cpu"
    if acc != 'gpu':
        print('##### RUNNING ON CPU ####')
        
    # Setup trainer
    trainer = pl.Trainer(
        devices=args.devices, 
        accelerator=acc, 
        max_epochs = args.epochs,
        log_every_n_steps = args.log_every_n,
        callbacks=[model.model_checkpoint] if args.initial_lr_steps == -1 else [model.model_checkpoint, model.lr_finder],
        #logger=tb_logger,
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders = loaders['train'],
        val_dataloaders   = loaders['validation']
    )

    # manually you can save best checkpoints - 
    trainer.save_checkpoint(f"{args.save_path}/{args.experiment_name}/{args.model_name}.ckpt")

    # Testing the model
    trainer.test(model, dataloaders=loaders['test'], ckpt_path = 'best')

    # Prediction
    trainer.predict(model, dataloaders=loaders['test'], ckpt_path = 'best')
        
    # saving sweep plot if activated
    if args.initial_lr_steps != -1:
        fig = trainer.model.lr_finder.optimal_lr.plot(suggest=True, show=False);
        plt.savefig(f"{args.save_path}/{args.experiment_name}/lr_sweep.png")
        plt.close(fig)


if __name__ == '__main__':

    # Get input arguments
    args = parse_arguments()
    
    # Train model
    train(args)




