# Reference: https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb

from pathlib import Path
from transformers.convert_graph_to_onnx import convert

convert(framework="pt",
        model="bert-base-cased",
        output=Path("onnx/pt-bert-base-cased.onnx"),
        opset=11)
