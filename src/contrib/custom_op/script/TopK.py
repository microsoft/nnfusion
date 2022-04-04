#!/bin/python
from concurrent.futures import thread
import sys
import os
import numpy as np

from __operator__ import OperatorBase, OperatorTestBase

# ONNX Reference Link:https://github.com/onnx/onnx/blob/main/docs/Operators.md#topk


def get_type_info(typestr):
    if typestr == "half" or typestr == "float16":
        return ("float16_t", 2, np.finfo(np.float16).min, np.finfo(np.float16).max)
    if typestr == "float32" or typestr == "float":
        return ("float", 4, np.finfo(np.float32).min, np.finfo(np.float32).max)
    if typestr == "double":
        return ("double", 8, np.finfo(np.double).min, np.finfo(np.double).max)
    if typestr == "int" or typestr == "int32":
        return ("int", 4, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    if typestr == "int64" or typestr == "long long":
        return ("int64_t", 8, np.iinfo(np.int64).min, np.iinfo(np.int64).max)
    exit(-1)

def get_antares_type_str(typestr):
    if typestr == "half" or typestr == "float16":
        return "float16"
    if typestr == "float32" or typestr == "float":
        return "float32"
    if typestr == "double":
        return "float64"
    if typestr == "int" or typestr == "int32":
        return "int32"
    if typestr == "int64" or typestr == "int64_t" or typestr == "long long":
        return "int64"


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
        for r in range(self['axis']+1, len(input_dict["input"]["shape"][0])):
            axis_stride *= input_dict["input"]["shape"][0][r]
        # for r in range(0, self['axis']):
        #    axis_stride *= input_dict["input"]["shape"][0][r]

        blocks = 1
        for r in range(0, len(input_dict["input"]["shape"][0])):
            if r == self["axis"]:
                continue
            blocks *= input_dict["input"]["shape"][0][r]

        max_element = 1
        while max_element < input_dict["input"]["shape"][0][self["axis"]]:
            max_element *= 2

        in_block_number = 512
        (dx_type_str, dx_type_size, dx_type_min, dx_type_max) = get_type_info(
            input_dict["input"]["dtype"][0])
        # element is shared memory: {_type_, int}
        # Only 4096 Bytes shared memory(L1 cache) in DX
        in_block_number = 4096 // (dx_type_size + 4)

        m_value = ""
        def_largest = "#define LARGEST"
        if 'largest' in input_dict and input_dict["largest"] == 0:
            def_largest = ""
            m_value = str(dx_type_max)
        else:
            m_value = str(dx_type_min)

        if max_element <= in_block_number:
            threads = max_element // 2
            self["hlsl_kernel"] = self.read_file("hlsl/topk_in_block_sort.hlsl").replace("__threads__", str(threads)).replace("__max_element__", str(max_element)).replace("__axis_stride__", str(
                axis_stride)).replace("__k__", str(self["K"])).replace("__type__", dx_type_str).replace("__define_largest__", def_largest).replace("__n__", str(input_dict["input"]["shape"][0][self["axis"]])
                ).replace("__M_VALUE__", m_value).replace("__blocks__", str(blocks))
            self["launch_config"] = [[blocks, 1, 1], [threads, 1, 1]]
        else:
            # Cannot use shared memory
            pass
        self["entry_point"] = "CSMain"
    
        antares_info = "// GLOBALS: input0:{0}[{1}] -> output0:{2}[{3}], output1:{4}[{5}]".format(
            get_antares_type_str(self["input"]["dtype"][0]),
            ", ".join([str(i) for i in self["input"]["shape"][0]]),
            get_antares_type_str(self["output"]["dtype"][0]),
            ", ".join([str(i) for i in self["output"]["shape"][0]]),
            get_antares_type_str(self["output"]["dtype"][1]),
            ", ".join([str(i) for i in self["output"]["shape"][1]]),
        )

        self["hlsl_kernel"] = antares_info + "\n" + antares_info.replace("GLOBALS:", "LOCAL: CSMain --") + "\n" + self["hlsl_kernel"]

    def config_infer(self, input_dict=None):
        outputs = {"shape": [], "dtype": []}
        outputs["shape"].append(input_dict["input"]["shape"][0].copy())
        outputs["shape"].append(input_dict["input"]["shape"][0].copy())
        outputs["dtype"].append(input_dict["input"]["dtype"][0])
        outputs["dtype"].append("int64_t")

        if self['axis'] < 0:
            self['axis'] += len(outputs["shape"][0])

        k = outputs["shape"][0][self['axis']]

        if "k" in input_dict:
            input_dict['K'] = input_dict['k']

        if 'data' in input_dict['input']:
            if 1 in input_dict['input']['data']:
                input_dict['K'] = int(input_dict['input']['data'][1][0])

        outputs["shape"][0][self['axis']] = input_dict["K"]
        outputs["shape"][1][self['axis']] = input_dict["K"]
        return outputs


class TopKTest(OperatorTestBase, TopK):
    def __init__(self, input_dict=None, config_infer=None):
        self.name = "TopK"

    def create_topk_test(self):
        import numpy as np
        self["axis"] = 1
        self["largest"] = 1
        self["sorted"] = 1
        self["K"] = 3
        self["input"] = {}
        self["input"]["shape"] = [[3, 4]]
        self["input"]["dtype"] = ["float32"]
        X = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [
            8, 9, 10, 11], ], dtype=np.float32)
        values_ref = np.array(
            [[3,  2,  1], [7,  6,  5], [11, 10,  9]], dtype=np.float32)
        indicies_ref = np.array(
            [[3,  2,  1], [3, 2, 1], [3, 2, 1]], dtype=np.int64)

        return {"kernel": TopK(self), "input": [X], "output": [values_ref, indicies_ref]}

    def create_topk_test_random_int(self):
        import random
        import torch
        shape = [123, 123, 123]
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["int32"]

        self["axis"] = 0
        self["largest"] = 1
        self["K"] = 57

        X = torch.randint(high=512, size=tuple(shape), dtype=torch.int32)
        (values_ref, indicies_ref) = torch.topk(
            X, k=self["K"], dim=self["axis"], largest=True, sorted=True)

        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy()]}

    def create_topk_test_random_fp16(self):
        import torch
        if not torch.cuda.is_available():
            return {}
        shape = [512, 512]
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float16"]

        self["axis"] = 0
        self["largest"] = 1
        self["K"] = 500

        X = torch.rand(size=tuple(shape), dtype=torch.float16).cuda()
        (values_ref, indicies_ref) = torch.topk(
            X, k=self["K"], dim=self["axis"], largest=True, sorted=True)

        return {"kernel": TopK(self), "input": [X.cpu().numpy()], "output": [values_ref.cpu().numpy(), indicies_ref.cpu().numpy()]}

    def create_topk_test_random_float(self):
        import random
        import torch
        shape = []
        for i in range(0, random.randint(2, 3)):
            shape.append(random.randint(100, 512))
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float32"]

        self["axis"] = random.randint(0, len(shape)-1)
        self["largest"] = 1
        k = random.randint(1, shape[self["axis"]])
        self['input']['data'] = {1: [str(k)]}

        X = torch.rand(tuple(shape), dtype=torch.float32) * 100
        (values_ref, indicies_ref) = torch.topk(
            X, k=k, dim=self["axis"], largest=True, sorted=True)

        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy()]}

    def create_topk_test_random_float_smallest(self):
        import torch
        import random
        shape = []
        for i in range(0, random.randint(2, 3)):
            shape.append(random.randint(100, 512))
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float32"]

        self["axis"] = random.randint(0, len(shape)-1)
        self["largest"] = 0
        self["K"] = random.randint(1, shape[self["axis"]])

        X = torch.rand(tuple(shape), dtype=torch.float32) * 100
        (values_ref, indicies_ref) = torch.topk(
            X, k=self["K"], dim=self["axis"], largest=False, sorted=True)

        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy()]}

    def allclose(self, truth, output):
        return super().allclose(truth[:1], output[:1])
    
    def export_onnx_test(self):
        import torch
        from torch import nn
        class T(nn.Module):
            def forward(self, a, b):
                m = a + b
                r = torch.topk(m, k = 97, dim = 0)
                return r
        m = T()
        torch.onnx.export(m, (torch.randn((232,124), dtype=torch.float32),  torch.randn((232, 124), dtype=torch.float32)), "topk.hlsl.onnx")