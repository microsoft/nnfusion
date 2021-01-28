*Note:* In this tutorial, we assume nnfusion cli has been installed as [Build Guide](https://github.com/microsoft/nnfusion/blob/master/docs/Build-Guide.md).

NNFusion leverages ONNX to support PyTorch. So this tutorial focuses on how to freeze an ONNX model from PyTorch source code. You could get NNFusion supported ONNX ops [here](https://github.com/microsoft/nnfusion/blob/master/src/nnfusion/frontend/onnx_import/ops_bridge.cpp).

## Freeze model by PyTorch ONNX exporter

Please refer to PyTorch [onnx section](https://pytorch.org/docs/stable/onnx.html) to convert a PyTorch model to ONNX format, currently it already supports a great majority of deeplearning workloads.

## Freeze model by NNFusion pt_freezer

On PyTorch onnx_exporter, we build a simple wrapper called [pt_freezer](https://github.com/microsoft/nnfusion/blob/master/models/pytorch2onnx/pytorch_freezer.py), it wraps PyTorch onnx_exporter with control flow and op availability(not implemented yet) check. We provide a well self-explanatory [VGG example](https://github.com/microsoft/nnfusion/blob/master/models/pytorch2onnx/vgg16_model.py) for this tool:

```bash
# step0: install prerequisites
apt update && sudo apt install python3-pip
pip3 install onnx torch torchvision

# step1: freeze vgg16 model
python3 vgg16_model.py
```

## Freeze model from thirdparties

Of course, you could freeze ONNX model from thirdparties, like [huggingface transformer](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb), which supports exporting to ONNX format.

## Freezed ONNX models

| model         | nnf codegen flags                 | download link |
| -----------   | -----------                       | -----------   |
| VGG16         | -f onnx                           | [vgg16.onnx](https://nnfusion.blob.core.windows.net/models/onnx/vgg16.onnx) |
| BERT_base     | -f onnx -p 'batch:3;sequence:512' | [pt-bert-base-cased.onnx](https://nnfusion.blob.core.windows.net/models/onnx/bert/pt-bert-base-cased.onnx) |
