# Freeze ONNX model from PyTorch

If you want to test NNFusion for PyTorch model, we recommend ONNX as intermediary, because PyTorch [onnx_exporter](https://pytorch.org/docs/stable/onnx.html) handles many dynamics and we don't want to reimplement them.

On PyTorch onnx_exporter, we build a simple wrapper to freeze and check(not fully implemented) whether NNFusion could accept the model. We provide a [VGG example](./vgg16_model.py) for this tool.

Of course, you could freeze ONNX model from thirdparties, like the [transformer example](./bert_model.py).
