import torch
from torch.utils.data import *
from torchvision import datasets, transforms  

def get_dataloader():

    batch_size = 3
    kwargs = {'batch_size': batch_size}

    kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                       transform=transform)
    train_dataloader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_dataloader = torch.utils.data.DataLoader(dataset2, **kwargs)

    return train_dataloader, test_dataloader