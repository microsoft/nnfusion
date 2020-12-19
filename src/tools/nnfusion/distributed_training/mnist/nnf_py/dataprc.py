# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torchvision import datasets, transforms
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def get_dataloader(device_id, world_size):

    batch_size = 3
    kwargs = {'batch_size': batch_size}

    kwargs.update({'num_workers': 1,
                   'pin_memory': True,
                   },
                  )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if(rank == 0):
        datasets.MNIST('/tmp',
                       train=True,
                       download=True,
                       )
        datasets.MNIST('/tmp',
                       train=False,
                       download=True,
                       )
    comm.barrier()
    dataset1 = datasets.MNIST('/tmp',
                              train=True,
                              download=False,
                              transform=transform)
    dataset2 = datasets.MNIST('/tmp',
                              train=False,
                              download=False,
                              transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset1,
        num_replicas=world_size,
        rank=device_id,
        shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset1, sampler=train_sampler, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(dataset2, **kwargs)

    return train_dataloader, test_dataloader
