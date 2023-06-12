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

# CUDA_VISIBLE_DEVICES=1 python src/models/project2/train_model.py --dataset PH2 --model_name UNet  --log_path /work3/$USER/repos/DL-COMVIS/logs/project2 --save_path /work3/$USER/repos/DL-COMVIS/models/project2  --batch_size 2 --lr 0.001 --initial_lr_steps -1 --optimizer Adam --epochs 2 --num_workers 24 --devices -1 --experiment_name test --augmentation 1 1 --loss BCE --norm none

def parse_arguments():

    

    parser = argparse.ArgumentParser()

    # GENERAL ()
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo-randomness.")
    parser.add_argument("--dataset", type=str, default='PH2',
                        help="Data set either (PH2 or DRIVE).")
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
    parser.add_argument("--norm", type=str, default = 'none',
                        help="Batch normalization - one of: [none, batchnorm, layernorm, instancenorm]")


    return parser.parse_args()
    


def train(args):
    # Set random seed
    set_seed(args.seed)

    # Get functions
    loss_fun = get_loss(args.loss, args.reg, args.reg_coef)
    optimizer = get_optimizer(args.optimizer)

    # Get normalization constants
    # TODO:
    #train_mean, train_std = get_normalization_constants(root=args.data_path, seed=args.seed)


    # Get data loaders with applied transformations
    loaders = get_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        seed=args.seed, 
        num_workers=args.num_workers,
        augmentations={'rotate': args.augmentation[0], 'flip': args.augmentation[1]}
    )


    folds = 20 if args.dataset == 'DRIVE' else 1
    returns = []
    for fold in range(folds):

        # Load model
        model = get_model(args.model_name, args, loss_fun, optimizer, fold)

        # Set up logger
        tb_logger = TensorBoardLogger(
            save_dir=f"{args.log_path}/{args.experiment_name}/{args.model_name}_fold{fold}",
            version=None,
            name=args.model_name,
        )

        # Setup trainer
        trainer = pl.Trainer(
            devices=args.devices, 
            accelerator='gpu', 
            max_epochs = args.epochs,
            log_every_n_steps = args.log_every_n,
            callbacks=[model.model_checkpoint] if args.initial_lr_steps == -1 else [model.model_checkpoint, model.lr_finder],
            logger=tb_logger,
        )
        
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders = loaders['train'] if args.dataset == 'PH2' else loaders[fold]['train'],
            val_dataloaders   = loaders['validation'] if args.dataset == 'PH2' else loaders[fold]['validation'],
        ) 

        # manually you can save best checkpoints - 
        trainer.save_checkpoint(f"{args.save_path}/{args.experiment_name}/{args.model_name}_fold{fold}.pt")

        # Testing the model
        returns.append(trainer.test(model, dataloaders=loaders['test'] if args.dataset == 'PH2' else loaders[fold]['test']))
        
        trainer.predict(model, dataloaders=loaders['test'] if args.dataset == 'PH2' else loaders[fold]['test'])
		
    test_dict = {
        key: sum(dct[0][key] for dct in returns) / len(returns)
        for key in returns[0][0].keys()
    }

    if not os.path.exists(f"{args.log_path}/{args.experiment_name}/{args.model_name}"):
        os.makedirs(f"{args.log_path}/{args.experiment_name}/{args.model_name}")
    with open(f"{args.log_path}/{args.experiment_name}/{args.model_name}/test_dict.txt", "w+") as fp:
        json.dump(test_dict, fp)  # encode dict into JSON

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




