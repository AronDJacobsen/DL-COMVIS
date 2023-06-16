import argparse

import pytorch_lightning as pl

from src.utils import set_seed, get_optimizer
from src.models.project4.models import get_model
from src.models.project4.losses import get_loss
from src.data.project4.dataloader import get_loaders

class BooleanListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Convert the string values to booleans
        bool_values = [bool(int(v)) for v in values]
        setattr(namespace, self.dest, bool_values)    

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    # GENERAL ()
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo-randomness.")
    parser.add_argument("--dataset", type=str, default='waste',
                        help="Data set either")
    parser.add_argument("--log_path", type=str, default = 'lightning_logs',
                        help="Path determining where to store logs.")
    parser.add_argument("--log_every_n", type=int, default=1,
                        help="Logging interval.")
    parser.add_argument("--save_path", type=str,
                        help="Path determining where to store results.")
    parser.add_argument("--verbose", type=bool, default=False,
                        help="Determines console logging.")
    parser.add_argument("--devices", type=int, default=1, 
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
                        help="Loss function - one of: [BCE]")
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
    parser.add_argument("--model_name", type=str, default='efficientnet_b4',
                        help="Model name - either 'efficientnet_b4' or ...")

    parser.add_argument("--percentage_to_freeze", type=float, default=None,
                    help="Percentage to freeze (transfer learning) in [0, 1]")

    return parser.parse_args()
    

def test(args):
    # Set random seed
    set_seed(args.seed)

    # Get functions
    loss_fun = get_loss(args.loss)
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
    model = get_model(args.model_name, args, loss_fun, optimizer, out=args.out)

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
    
    trainer.predict(model, dataloaders=loaders['test'], ckpt_path=args.path_model)


if __name__ == '__main__':

    # Get input arguments
    args = parse_arguments()
    
    # test model
    test(args)

