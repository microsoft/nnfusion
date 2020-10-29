# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch, torchvision


def main():
    model = torchvision.models.vgg16()
    model.eval()

    example_input = torch.ones([1, 3, 224, 224])
    trace_module = torch.jit.trace(model, example_input)
    trace_module.save('vgg16_trace_module.pt')


def test():
    trace_module = torch.jit.load("vgg16_trace_module.pt")
    example_input = torch.ones([1, 3, 224, 224])
    trace_module.eval()
    out = trace_module(example_input)
    print(out)
    print(out.shape)


if __name__ == '__main__':
    main()
    test()