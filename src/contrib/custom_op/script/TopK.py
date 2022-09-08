#!/bin/python
from concurrent.futures import thread
from math import ceil
import sys
import os
import numpy as np
from __operator__ import OperatorBase, OperatorTestBase, get_type_info, get_antares_type_str, read_file, replace_template_args


class TopK(OperatorBase):
    class TopKConfig(dict):
        def __init__(self, topkop):
            self.input_dtype_0 = topkop["input"]["dtype"][0]
            self.input_shape_0 = topkop["input"]["shape"][0]
            self.output_dtype_1 = topkop["output"]["dtype"][1]
            self.axis = topkop["axis"]
            self.largest = True 
            self.__K__ = topkop["K"]
            if 'largest' in topkop and topkop["largest"] == 0:
                self.largest = False
            self.get_config()
            self.init_dict()
        
        def init_dict(self):
            for k in self.__dict__:
                if k.startswith("__") and k.endswith("__"):
                    self[k] = str(self.__dict__[k])

        def get_block_max_element(self, dtype):
            (dx_type_str, dx_type_size, dx_type_min, dx_type_max) = get_type_info(dtype)
            in_block_number = 512
            # Element in shared memory: [{_type_, int}, ... ]
            # Only 4096 Bytes shared memory(L1 cache) in DirectX
            in_block_number = 4096 // (dx_type_size + 4)
            return in_block_number
        
        def get_config(self):
            # [GreaterBlock(SmallerBlock(Threads for max 512 elements))]
            # Maximum elements processed by one block of threads
            self.__block_max_element__ = self.get_block_max_element(self.input_dtype_0)
            # Maximum elements for one sequence in one block
            self.__block_max_seq_element__ = self.__block_max_element__ // 2
            # Stride between two elements in original 
            self.__axis_stride__ = 1 
            for r in range(self.axis+1, len(self.input_shape_0)):
                self.__axis_stride__ *= self.input_shape_0[r]
            # "Greader block" is which process one batch of data
            self.__greater_blocks__ = 1
            for r in range(0, len(self.input_shape_0)):
                if r == self.axis:
                    continue
                self.__greater_blocks__ *= self.input_shape_0[r]
            # Max element placeholders to process topK
            self.__thread_max_element__ = 1
            while self.__thread_max_element__ < self.__K__:
                self.__thread_max_element__ *= 2
            self.__axis_size__ = self.input_shape_0[self.axis]
            self.__threads__ = self.__thread_max_element__
            self.__smaller_blocks__ = ceil(float(self.__axis_size__) / float(self.__thread_max_element__))
            self.__value_type__ , _, dx_type_min, dx_type_max = get_type_info(self.input_dtype_0)
            self.__index_type__ , _, _, _ = get_type_info(self.output_dtype_1)
            self.__boundary_value__ = ""
            self.__largest__ = 1
            if self.largest:
                self.__largest__ = 1
                self.__boundary_value__ = str(dx_type_min)
            else:
                self.__largest__ = 0
                self.__boundary_value__ = str(dx_type_max)
            self.__max_mega_step__ = 1
            while self.__max_mega_step__ < self.__smaller_blocks__:
                self.__max_mega_step__ *= 2

            if self.__thread_max_element__ > self.__block_max_seq_element__:
                # max elements size makes algorithm needs more shared memory than DX12 can provide.
                exit(0)

    def __init__(self, input_dict=None, config_infer=None):
        # Here is some difference with original ONNX define
        self.cs_5_compatiable_mode = True
        super().__init__(input_dict, self.config_infer)
        self.attach_directx_hlsl_kernel()

    # Attach this to let the NNFusion HLSL codegen organize the kernel
    def attach_antares_hlsl_kernel_config(self):
        antares_info = "// GLOBALS: input0:{0}[{1}] -> output0:{2}[{3}], output1:{4}[{5}], output2:{6}[{7}]".format(
            get_antares_type_str(self["input"]["dtype"][0]),
            ", ".join([str(i) for i in self["input"]["shape"][0]]),
            get_antares_type_str(self["output"]["dtype"][0]),
            ", ".join([str(i) for i in self["output"]["shape"][0]]),
            get_antares_type_str(self["output"]["dtype"][1]),
            ", ".join([str(i) for i in self["output"]["shape"][1]]),
            get_antares_type_str(self["output"]["dtype"][2]),
            ", ".join([str(i) for i in self["output"]["shape"][2]]),
        )
        self["hlsl_kernel"] = antares_info + "\n\n\n" + "// ---------------------------------------------------------------------------\n" + antares_info.replace("GLOBALS:", "LOCAL: CSMain --") + "\n" + self["hlsl_kernel"]

    # Generate a HLSL Kernels
    def attach_directx_hlsl_kernel(self):
        topkconf = self.TopKConfig(self)
        in_block_kernel = read_file("hlsl/topk/topk.hlsl")
        self.in_block_kernel = replace_template_args(in_block_kernel, topkconf)

        self["hlsl_kernel"] = "\n".join([self.in_block_kernel])
        self["launch_config"] = [[topkconf.__greater_blocks__, 1, 1], [topkconf.__threads__, 1, 1]]
        self["entry_point"] = "CSMain"

        self.attach_antares_hlsl_kernel_config()

    def config_infer(self, input_dict=None):
        outputs = {"shape": [], "dtype": []}
        # output 0: values
        outputs["shape"].append(input_dict["input"]["shape"][0].copy())
        # output 1: indicies
        outputs["shape"].append(input_dict["input"]["shape"][0].copy())
        # output 2: temporary indicies
        outputs["shape"].append(input_dict["input"]["shape"][0].copy())
        outputs["dtype"].append(input_dict["input"]["dtype"][0])
        if self.cs_5_compatiable_mode:
            outputs["dtype"].append("int32")
            outputs["dtype"].append("int32")
        else:
            outputs["dtype"].append("int64_t")
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

        buf = np.zeros(X.shape, dtype=np.int32)
        return {"kernel": TopK(self), "input": [X], "output": [values_ref, indicies_ref, buf]}

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

        buf = np.zeros(X.shape, dtype=np.int32)
        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy(), buf]}

    def create_topk_test_random_fp16(self):
        import torch
        if not torch.cuda.is_available():
            print("No CUDA Runtime Found.")
            return {}
        shape = [10, 512]
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float16"]

        self["axis"] = 1
        self["largest"] = 1
        self["K"] = 129

        X = torch.rand(size=tuple(shape), dtype=torch.float16).cuda()
        (values_ref, indicies_ref) = torch.topk(
            X, k=self["K"], dim=self["axis"], largest=True, sorted=True)

        buf = np.zeros(X.shape, dtype=np.int32)
        return {"kernel": TopK(self), "input": [X.cpu().numpy()], "output": [values_ref.cpu().numpy(), indicies_ref.cpu().numpy(), buf]}

    def create_topk_test_random_float(self):
        import random
        import torch
        shape = []
        for i in range(0, random.randint(2, 3)):
            shape.append(random.randint(100, 256))
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float32"]

        self["axis"] = random.randint(0, len(shape)-1)
        self["largest"] = 1
        k = random.randint(1, shape[self["axis"]])
        self['K'] = k

        X = torch.rand(tuple(shape), dtype=torch.float32) * 100
        (values_ref, indicies_ref) = torch.topk(
            X, k=k, dim=self["axis"], largest=True, sorted=True)

        buf = np.zeros(X.shape, dtype=np.int32)
        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy(), buf]}

    def create_topk_test_random_float_smallest(self):
        import torch
        import random
        shape = []
        for i in range(0, random.randint(2, 3)):
            shape.append(random.randint(100, 256))
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float32"]

        self["axis"] = random.randint(0, len(shape)-1)
        self["largest"] = 0
        self["K"] = random.randint(1, shape[self["axis"]])

        X = torch.rand(tuple(shape), dtype=torch.float32) * 100
        (values_ref, indicies_ref) = torch.topk(
            X, k=self["K"], dim=self["axis"], largest=False, sorted=True)

        buf = np.zeros(X.shape, dtype=np.int32)
        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy(), buf]}

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
    
    def create_topk_test_random_float_very_large(self):
        # Get top 60 from 1x[1]3018 tensor
        import random
        import torch
        shape = [1, 3018]
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float32"]

        self["axis"] = 1
        self["largest"] = 1
        k = 6
        self["K"] = k

        X = torch.rand(tuple(shape), dtype=torch.float32) * 100
        (values_ref, indicies_ref) = torch.topk(
            X, k=k, dim=self["axis"], largest=True, sorted=True)

        buf = np.zeros(X.shape, dtype=np.int32)
        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy(), buf]}