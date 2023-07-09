#!/bin/python
from concurrent.futures import thread
from math import ceil
import math
import sys
import os
import numpy as np
from __operator__ import OperatorBase, OperatorTestBase, get_type_info, get_antares_type_str, read_file, replace_template_args


class ArgMax(OperatorBase):
    class ArgMaxConfig(dict):
        def __init__(self, op) -> None:
            self.input_dtype_0 = op["input"]["dtype"][0]
            self.input_shape_0 = op["input"]["shape"][0]
            self.output_dtype_0 = op["output"]["dtype"][0]
            self.output_shape_0 = op["output"]["shape"][0]
            self.axis = op["axis"]
            self.keepdims = op["keepdims"]
            self.get_config()
            self.init_dict()
        
        def init_dict(self):
            for k in self.__dict__:
                if k.startswith("__") and k.endswith("__"):
                    self[k] = str(self.__dict__[k])
        
        def get_config(self):
            # [GreaterBlock(SmallerBlock(Threads for max 512 elements))]
            # Maximum elements processed by one block of threads
            self.__axis_size__ = self.input_shape_0[self.axis]
            self.__value_type__ , dx_type_size, dx_type_min, dx_type_max = get_type_info(self.input_dtype_0)
            self.__index_type__ , _, _, _ = get_type_info(self.output_dtype_0)
            self.__boundary_value__ = dx_type_min

            in_block_number = 4096 // dx_type_size
            self.__block_max_element__ = min(in_block_number, self.__axis_size__)

            self.__blocks__ = 1
            for r in range(0, len(self.input_shape_0)):
                if r == self.axis:
                    continue
                self.__blocks__ *= self.input_shape_0[r]
            self.__axis_stride__ = 1 
            for r in range(self.axis+1, len(self.input_shape_0)):
                self.__axis_stride__ *= self.input_shape_0[r]

            self.__threads__ = min(self.input_shape_0[self.axis], self.__block_max_element__)
            self.__step_size__ = self.__axis_size__ 

    def __init__(self, input_dict=None, config_infer=None):
        self.cs_5_compatible_mode = True
        super().__init__(input_dict, self.config_infer)
        self.attach_directx_hlsl_kernel()

    def attach_antares_hlsl_kernel_config(self):
        antares_info = "// GLOBALS: input0:{0}[{1}] -> output0:{2}[{3}]".format(
            get_antares_type_str(self["input"]["dtype"][0]),
            ", ".join([str(i) for i in self["input"]["shape"][0]]),
            get_antares_type_str(self["output"]["dtype"][0]),
            ", ".join([str(i) for i in self["output"]["shape"][0]]),
        )
        self["hlsl_kernel"] = antares_info + "\n\n\n" + "// ---------------------------------------------------------------------------\n" + antares_info.replace("GLOBALS:", "LOCAL: CSMain --") + "\n" + self["hlsl_kernel"]

    # Generate a HLSL Kernels
    def attach_directx_hlsl_kernel(self):
        conf = self.ArgMaxConfig(self)
        in_block_kernel = read_file("hlsl/argmax/argmax.hlsl")
        self.in_block_kernel = replace_template_args(in_block_kernel, conf)

        self["hlsl_kernel"] = "\n".join([self.in_block_kernel])
        self["launch_config"] = [[conf.__blocks__, 1, 1], [conf.__threads__, 1, 1]]
        self["entry_point"] = "CSMain"

        self.attach_antares_hlsl_kernel_config()
    
    def config_infer(self, input_dict=None):
        # input is {shape[1], dtype[1], axis, keepdims}
        outputs = {"shape": [], "dtype": []}
        outputs["shape"].append(input_dict["input"]["shape"][0].copy())
        if "axis" not in input_dict:
            input_dict["axis"] = 0
        if "keepdims" not in input_dict:
            input_dict["keepdims"] = 0
        axis = input_dict["axis"]
        keepdims = input_dict["keepdims"]

        if keepdims == 0:
            del outputs["shape"][0][axis]
        else:
            outputs["shape"][0][axis] = 1

        if self.cs_5_compatible_mode:
            outputs["dtype"].append("int32")
        else:
            outputs["dtype"].append("int64_t")
        return outputs
    
class ArgMaxTest(OperatorTestBase, ArgMax):
    def __init__(self, input_dict=None, config_infer=None):
        self.name = "TopK"
    
    def create_topk_test_random_float(self):
        import torch
        shape = [1, 2048]
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float32"]
        self["axis"] = 1
        self["keepdims"] = 0

        X = torch.rand(tuple(shape), dtype=torch.float32) * 100
        Y = torch.argmax(X, dim=self["axis"], keepdim=(self["keepdims"]==1))

        op = ArgMax(self)
        if op.cs_5_compatible_mode:
            Y = Y.int()

        return {"kernel": op, "input": [X.numpy()], "output": [Y.numpy()]}