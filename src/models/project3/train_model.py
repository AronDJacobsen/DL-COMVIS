import argparse

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()

    # GENERAL
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo-randomness.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Run on GPU or not.")
    
    # TRAINING PARAMETERS
    parser.add_argument("--model_name", type=str,
                        help="Name of the model. Either VanillaGAN, LSGAN or WGAN (currently).")
    parser.add_argument("--epochs", type=int, default=10,
                        help='Number of epochs to train model.')

    return parser.parse_args()

def train(model,
    train_loader: DataLoader, 
    batch_size: int,
    num_epochs: int = 10,
    seed: int = 0,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup figure
    plt.figure(figsize=(20,10))
    subplots = [plt.subplot(2, 6, k+1) for k in range(12)]
    discriminator_final_layer = torch.sigmoid

    for epoch in range(num_epochs):
        for minibatch_no, (x, target) in enumerate(train_loader):
            
            # Scale real image to (-1, 1) range
            x_real = x.to(model.device)*2-1 

            # Generate fake image
            z = torch.randn(x.shape[0], 100).to(model.device)
            x_fake = model.g(z)
            
            #Update discriminator
            model.d.zero_grad()
            d_loss = model.discriminator_loss(x_real, x_fake)
            d_loss.backward()
            model.d_opt.step()

            #Update generator
            model.g.zero_grad()
            g_loss = model.generator_loss(x_real, x_fake) 
            g_loss.backward()
            model.g_opt.step()

            # Check loss is non-nan
            assert(not np.isnan(d_loss.item()))

            #Plot results every 100 minibatches
            if minibatch_no % 100 == 0:
                with torch.no_grad():

                    # Generate example images
                    P = discriminator_final_layer(model.d(x_fake))
                    for k in range(11):

                        # Visualize generated sample + discriminator sigmoid score
                        x_fake_k = x_fake[k].cpu().squeeze()/2+.5
                        subplots[k].imshow(x_fake_k, cmap='gray')
                        subplots[k].set_title('d(x)=%.2f' % P[k])
                        subplots[k].axis('off')
                    
                    # Get histogram data for real and fake images
                    z = torch.randn(batch_size, 100).to(model.device)
                    H1 = discriminator_final_layer(model.d(model.g(z))).cpu()
                    H2 = discriminator_final_layer(model.d(x_real)).cpu()

                    # Compute stats
                    plot_min = min(H1.min(), H2.min()).item()
                    plot_max = max(H1.max(), H2.max()).item()
                    
                    # Plot histograms
                    subplots[-1].cla()
                    subplots[-1].hist(H1.squeeze(), label='fake', range=(plot_min, plot_max), alpha=0.5)
                    subplots[-1].hist(H2.squeeze(), label='real', range=(plot_min, plot_max), alpha=0.5)
                    subplots[-1].legend()
                    subplots[-1].set_xlabel('Probability of being real')
                    subplots[-1].set_title('Discriminator loss: %.2f' % d_loss.item())

                    # Update figure
                    title = 'Epoch {e} - minibatch {n}/{d}'.format(e=epoch+1, n=minibatch_no, d=len(train_loader))
                    plt.gcf().suptitle(title, fontsize=20)
                    
                    raise NotImplementedError("Save images instead of IPythong version with `display`")
                    # old: display.display(plt.gcf())
                    # old: display.clear_output(wait=True)

if __name__ == '__main__':
    from src.models.project3 import *

    # Parse arguments
    args = parse_arguments()
    
    # Set device
    device = torch.device(args.device)

    # Get model
    if args.model_name == 'VanillaGAN':
        model = VanillaGAN(device=device)
    elif args.model_name == 'LSGAN':
        model = LSGAN(device=device)
    elif args.model_name == 'WGAN':
        model = WGAN(device=device)

    # Get dataloaders
    loaders = ...           # dictionary structure

    # Train GAN networks
    train(model, loaders['train'], num_epochs=args.epochs, seed=args.seed)