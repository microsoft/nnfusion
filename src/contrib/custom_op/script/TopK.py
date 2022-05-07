#!/bin/python
from concurrent.futures import thread
from math import ceil
import sys
import os
import numpy as np
from __operator__ import OperatorBase, OperatorTestBase, get_type_info, get_antares_type_str, read_file


class TopK(OperatorBase):
    class TopKConfig:
        def __init__(self, topkop) -> None:
            self.input_dtype_0 = topkop["input"]["dtype"][0]
            self.input_shape_0 = topkop["input"]["shape"][0]
            self.output_dtype_1 = topkop["output"]["dtype"][1]
            self.axis = topkop["axis"]
            self.largest = False
            self.K = topkop["K"]
            if 'largest' in topkop and topkop["largest"] == 0:
                self.largest = True
            self.get_config()

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
            self.m_block_max_element = self.get_block_max_element(self.input_dtype_0)
            # Maximum elements for one sequence in one block
            self.m_block_max_seq_element = self.m_block_max_element / 2
            # Stride between two elements in orginal 
            self.m_axis_size = self.axis
            self.m_axis_stride = 1 
            for r in range(self.axis+1, len(self.input_shape_0)):
                self.m_axis_stride *= self.input_shape_0[r]
            # "Greader block" is which process one batch of data
            self.m_greater_blocks = 1
            for r in range(0, len(self.input_shape_0)):
                if r == self.axis:
                    continue
                self.m_greater_blocks *= self.input_shape_0[r]
            # Max element placeholders to process topK
            self.m_thread_max_element = 1
            while self.m_thread_max_element < self.K:
                self.m_thread_max_element *= 2
            self.m_thread = self.m_thread_max_element
            self.m_smaller_blocks = ceil(float(self.axis) / float(self.m_thread_max_element))
            self.m_value_type , _, dx_type_min, dx_type_max = get_type_info(self.input_dtype_0)
            self.m_index_type , _, _, _ = get_type_info(self.output_dtype_1)
            self.m_boundary_value = ""
            self.macro_def_largest = "#define LARGEST"
            if self.largest:
                self.macro_def_largest = ""
                self.m_boundary_value = str(dx_type_max)
            else:
                self.m_boundary_value = str(dx_type_min)
            
            if self.m_max_element > self.m_block_max_seq_element:
                # max elements size makes algorithm needs more shared memory than DX12 can provide.
                exit(0)

            # Yield:
            # - Blocks:
            #  self.m_block_max_element
            #  self.m_greater_blocks
            #  self.m_smaller_blocks
            # - Theads:
            #  self.m_thread_max_element
            #  self.m_threads
            # - Data:
            #  self.m_axis_stride
            # - Other:
            #  self.m_boundary_value
            #  self.macro_def_largest

    def __init__(self, input_dict=None, config_infer=None):
        # Here is some difference with original ONNX define
        self.cs_5_compatiable_mode = True
        super().__init__(input_dict, self.config_infer)
        self.attach_directx_hlsl_kernel(input_dict)
        self.attach_antares_hlsl_kernel_config()

    # Attach this to let the NNFusion HLSL codegen organize the kernel
    def attach_antares_hlsl_kernel_config(self):
        antares_info = "// GLOBALS: input0:{0}[{1}] -> output0:{2}[{3}], output1:{4}[{5}]".format(
            get_antares_type_str(self["input"]["dtype"][0]),
            ", ".join([str(i) for i in self["input"]["shape"][0]]),
            get_antares_type_str(self["output"]["dtype"][0]),
            ", ".join([str(i) for i in self["output"]["shape"][0]]),
            get_antares_type_str(self["output"]["dtype"][1]),
            ", ".join([str(i) for i in self["output"]["shape"][1]]),
        )
        self["hlsl_kernel"] = antares_info + "\n" + antares_info.replace("GLOBALS:", "LOCAL: CSMain --") + "\n" + self["hlsl_kernel"]
    
    # Generate a HLSL Kernels
    def attach_directx_hlsl_kernel(self, input_dict=None):
        topkconf = self.TopKConfig(input_dict)
        print(topkconf.m_max_element)
        exit(0)
        return

        self["hlsl_kernel"] = \
            self.read_file("hlsl/topk_in_block_sort.hlsl") \
            .replace("__threads__", str(threads)) \
            .replace("__max_element__", str(m_max_element)) \
            .replace("__axis_stride__", str(m_axis_stride)) \
            .replace("__k__", str(self["K"])) \
            .replace("__type__", dx_type_str) \
            .replace("__define_largest__", def_largest) \
            .replace("__n__", str(input_dict["input"]["shape"][0][self["axis"]])) \
            .replace("__M_VALUE__", m_value) \
            .replace("__blocks__", str(blocks)) \
            .replace("__out_type__", dx_out_type_str)
        self["launch_config"] = [[blocks, 1, 1], [threads, 1, 1]]
        self["entry_point"] = "CSMain"

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
    
    def create_topk_test_random_float_very_large(self):
        # Get top 60 from 1x[1]3018 tensor
        import random
        import torch
        shape = [1, 3018]
        self["input"] = {}
        self["input"]["shape"] = [shape]
        self["input"]["dtype"] = ["float32"]

        self["axis"] = random.randint(0, len(shape)-1)
        self["largest"] = 1
        k = 60
        self['input']['data'] = {1: [str(k)]}

        X = torch.rand(tuple(shape), dtype=torch.float32) * 100
        (values_ref, indicies_ref) = torch.topk(
            X, k=k, dim=self["axis"], largest=True, sorted=True)

        return {"kernel": TopK(self), "input": [X.numpy()], "output": [values_ref.numpy(), indicies_ref.numpy()]}