# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torchvision import datasets, transforms


def get_mnist_dataloader(**kwargs):
    data_config = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    data_config.update(kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    dataset1 = datasets.MNIST('./tmp',
                              train=True,
                              download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./tmp',
                              train=False,
                              download=True,
                              transform=transform)
    train_dataloader = torch.utils.data.DataLoader(dataset1, **data_config)
    test_dataloader = torch.utils.data.DataLoader(dataset2, **data_config)

    return train_dataloader, test_dataloader
