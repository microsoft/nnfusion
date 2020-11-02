# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NNFusion codegen flags: nnfusion /path/to/vgg16.onnx -f onnx
from pytorch_freezer import IODescription, ModelDescription, PTFreezer
import torch
import torchvision


def main():
    # define pytorch module
    model = torchvision.models.vgg16()
    # define the pytorch model input/output description
    input_desc = [IODescription("data", [1, 3, 224, 224], torch.float32)]
    output_desc = [IODescription("logits", [1, 1000], torch.float32)]
    model_desc = ModelDescription(input_desc, output_desc)
    # save the onnx model somewhere
    freezer = PTFreezer(model, model_desc)
    freezer.execute("./vgg16.onnx")


if __name__ == '__main__':
    main()