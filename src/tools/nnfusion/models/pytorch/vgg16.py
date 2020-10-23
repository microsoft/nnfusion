from freezer import IODescription, ModelDescription, Freezer
import torch
import torchvision


def main():
    model = torchvision.models.vgg16()
    input_desc = [IODescription("data", [1, 3, 224, 224], torch.float32)]
    output_desc = [IODescription("logits", [1, 1000], torch.float32)]
    model_desc = ModelDescription(input_desc, output_desc)
    freezer = Freezer(model, model_desc)
    freezer.freeze_onnx_model("./vgg16.onnx")


if __name__ == '__main__':
    main()