import argparse

import torch
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from src.utils import set_seed
from src.models.project1.models import get_model
from src.data.project1.dataloader import get_loaders, get_normalization_constants

def parse_arguments():

    parser = argparse.ArgumentParser()

    # EXPERIMENT PARAMETERS
    parser.add_argument("--data_path", type=str, default='/dtu/datasets1/02514/hotdog_nothotdog',
                        help="Path to data set.")
    parser.add_argument("--network_name", type=str,
                        help="Network name - either 'efficientnet_b4' or one of the self-implemented ones.")
    parser.add_argument("--log_path", type=str,
                        help="Path determining where to store logs.")
    parser.add_argument("--save_path", type=str,
                        help="Path determining where to store results.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo-randomness.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="Determines console logging.")
    parser.add_argument("--experiment_name", type=str,
                        help="Sets the overall experiment name.")
    parser.add_argument("--log_every_n", type=int, default=1,
                        help="Logging interval.")
    parser.add_argument("--devices", type=int, default=2, 
                        help="Number of devices")
    
    # TRAINING PARAMETERS
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="The optimizer to be used.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training the model.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers in the dataloader.")

    return parser.parse_args()

def train(args):
    # Set random seed
    set_seed(args.seed)

    # Load model
    model = get_model(network_name=args.network_name)(args)
    
    # Get normalization constants
    train_mean, train_std = get_normalization_constants(root=args.data_path, seed=args.seed)

    # Define transforms for training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),             # flips "left-right"
        # transforms.RandomVerticalFlip(p=1.0),             # flips "upside-down"
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(60, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=train_mean, 
            std=train_std, 
        )
    ])

    # Define transforms for test and validation
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=train_mean, 
            std=train_std, 
        )
    ])

    # Get data loaders with applied transformations
    loaders = get_loaders(
        root=args.data_path, 
        batch_size=args.batch_size, 
        seed=args.seed, 
        train_transforms=train_transforms, 
        test_transforms=test_transforms, 
        num_workers=args.num_workers,
    )

    # Set up logger
    tb_logger = TensorBoardLogger(
        save_dir=f"{args.log_path}/{args.experiment_name}",
        version=None,
        name=args.network_name,
    ),

    # Setup trainer
    trainer = pl.Trainer(
        devices=args.devices, 
        accelerator="gpu", 
        max_epochs = args.epochs,
        log_every_n_steps = args.log_every_n,
        callbacks=[model.model_checkpoint],
        logger=tb_logger,
    ) 

    # Train model
    trainer.fit(
        model=model,
        train_dataloaders = loaders['train'],
        val_dataloaders = loaders['validation'], 
    ) 

    # manually you can save best checkpoints - 
    trainer.save_checkpoint(f"{args.save_path}/{args.experiment_name}/{args.network_name}.pt")


if __name__ == '__main__':

    # Get input arguments
    args = parse_arguments()
    
    # Train model
    train(args)