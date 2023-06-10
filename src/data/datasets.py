#!/usr/bin/env python3


import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_data(data_name, batch_size):


    num_workers = 1

    if data_name == 'CIFAR10':
        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())#, num_workers=num_workers)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())#, num_workers=num_workers)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif data_name == 'MNIST':
        trainset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

