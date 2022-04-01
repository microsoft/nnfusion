#!/bin/python
from concurrent.futures import thread
from pickletools import read_float8
import sys, os
from __operator__ import OperatorBase, OperatorTestBase

# ONNX Reference Link:https://github.com/onnx/onnx/blob/main/docs/Operators.md#topk


class TopK(OperatorBase):
    def __init__(self, input_dict=None, config_infer=None):
        # Here is some difference with original ONNX define
        super().__init__(input_dict, self.config_infer)
        self.attach_directx_hlsl_kernel(input_dict)

    def read_file(self, file_name):
        with open(os.path.join(os.path.dirname(__file__), file_name)) as f:
            lines = f.readlines()
            return "".join(lines)

    # Generate a HLSL Kernels
    # How about generating host call?
    def attach_directx_hlsl_kernel(self, input_dict=None):
        axis_stride = 1
        for r in range(self['axis'] + 1, len(input_dict["input"]["shape"][0])):
            axis_stride *= input_dict["input"]["shape"][0][r]
        
        blocks = 1
        for r in range(0, len(input_dict["input"]["shape"][0])):
            if r == self["axis"]:
                continue
            blocks *= input_dict["input"]["shape"][0][r]

        threads = 1
        while threads < input_dict["input"]["shape"][0][self["axis"]]:
            threads *= 2

        if threads <= 512:
            self["hlsl_kernel"] = self.read_file("hlsl/topk_in_block_sort.hlsl"
                ).replace("__threads__", str(threads)
                ).replace("__axis_stride__", str(axis_stride)
                ).replace("__k__", str(self["K"])
                ).replace("__n__", str(input_dict["input"]["shape"][0][self["axis"]]))
            self["launch_config"] = [[blocks, 1, 1], [threads, 1, 1]]
        self["entry_point"] = "TopK"
        print(self["hlsl_kernel"])

    def config_infer(self, input_dict=None):
        if len(input_dict["input"]["shape"]) > 1:
            sys.stderr.write(
                "TopK only support one input: K should be translated to constant attribution.")
            exit(-1)
        outputs = {"shape": [], "dtype": []}
        for ele in input_dict["input"]["shape"]:
            outputs["shape"].append(ele)
        for ele in input_dict["input"]["dtype"]:
            outputs["dtype"].append(ele)

        if self['axis'] < 0:
            self['axis'] += len(outputs["shape"][0])

        outputs["shape"][0][self['axis']] = input_dict['K']
        return outputs


class TopKTest(OperatorTestBase, TopK):
    def __init__(self, input_dict=None, config_infer=None):
        # <Inputs>
        # self["x"]  # Tensor of shape [a_1, a_2, ..., a_n, r]
        # self["K"]  # A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve
        # <Outputs>
        # Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing top K values from the input tensor
        # self["Values"]
        # Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing the corresponding input tensor indices for the top K values.
        # self["Indices"]
        # <Default Value>
        # Attribute
        # Dimension on which to do the sort. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
        self["axis"] = -1
        self["largest"] = 1 # Whether to return the top-K largest or smallest elements.
        self["sorted"] = 1  # Whether to return the elements in sorted order.
        self["K"] = 1
        self.name = "TopK"

    def create_topk_test(self):
        import numpy as np
        import torch
        self["axis"] = 1
        self["largest"] = 1
        self["K"] = 3
        self["input"] = {}
        self["input"]["shape"] = [[3, 4]]
        self["input"]["dtype"] = [["float32"]]
        X = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [
            8, 9, 10, 11], ], dtype=np.float32)
        values_ref = np.array(
            [[3,  2,  1], [7,  6,  5], [11, 10,  9]], dtype=np.float32)
        indicies_ref = np.array(
            [[3,  2,  1], [3, 2, 1], [3, 2, 1]], dtype=np.int64)

        return {"kernel": TopK(self), "input": [X], "output": [values_ref, indicies_ref]}
