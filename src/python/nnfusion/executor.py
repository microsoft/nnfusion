# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import platform
import json
import ctypes
from . import dtypes
from .utils import cd
from .description import IODescription


def find_nnf_rt(nnf_rt_dir):
    def is_nnf_rt(file_name):
        if platform.system().lower() == "linux":
            return file_name.startswith("libnnf") and file_name.endswith(
                "rt.so")
        elif platform.system().lower() == "windows":
            return file_name.startswith("nnf") and file_name.endswith("rt.dll")
        else:
            return False

    for file_name in os.listdir(nnf_rt_dir):
        if is_nnf_rt(file_name):
            return os.path.join(nnf_rt_dir, file_name)
    return ""


def parse_nnf_params(param_file):
    with open(param_file) as f:
        nnf_params = json.load(f)

    def convert_nnf_info(nnf_info, is_nnf_input):
        split_index_str = "inputs[" if is_nnf_input else "outputs["
        out = {}
        for name, desc in nnf_info.items():
            index = desc["id"].split(split_index_str)[1].split("]")[0]
            dtype = desc["id"][2:].split("*")[0]
            if dtype.endswith("_t"):
                dtype = dtype.rstrip("_t")
            elif dtype == "float":
                dtype = "float32"
            elif dtype == "double":
                dtype = "float64"
            shape = desc["shape"]
            out[name] = {
                "name": name,
                "id": int(index),
                "dtype": dtype,
                "shape": shape,
                "raw_id": desc["id"],
                "nnf_name": desc["name"]
            }
        return out

    weights = convert_nnf_info(nnf_params.get("weight", dict()),
                               is_nnf_input=True)
    inputs = convert_nnf_info(nnf_params.get("input", dict()),
                              is_nnf_input=True)
    outputs = convert_nnf_info(nnf_params.get("output", dict()),
                               is_nnf_input=False)

    return weights, inputs, outputs


class Executor(object):
    """
    Python wrapper for NNFusion runtime.
    Executor loads a compiled nnf_rt dynamic lib, provide a __call__ func to
    execute the "kernel_entry" in nnf_rt with given data.
    """
    device_type_map = {
        0: ("cuda_init", "cuda_free"),  # CUDA_GPU
        1: ("rocm_init", "rocm_free"),  # ROCM_GPU
        2: ("cpu_init", "cpu_free"),  # GENERIC_CPU
        3: ("hlsl_init", "hlsl_free"),  # HLSL
        4: ("graphcore_init", "graphcore_free"),  # GraphCore
        5: ("", ""),  # UNKNOWN
    }

    def __init__(self, nnf_rt_dir):
        """
        Parameters:
            nnf_rt_dir: A full string path to nnfusion runtime,
                it's usually like "codegen_root/nnfusion_rt/cuda_codegen".
        """
        nnf_rt_dir = os.path.abspath(nnf_rt_dir)
        self.libnnf_path = find_nnf_rt(nnf_rt_dir)
        if self.libnnf_path == "":
            raise Exception(
                "nnf_rt lib not found in folder {}".format(nnf_rt_dir))

        # prepare init/free/kernel_entry
        self.init_flag = False
        # dxil.dll and dxcompiler.dll must be manually imported
        if os.path.exists(os.path.join(nnf_rt_dir, "dxil.dll")):
            ctypes.cdll.LoadLibrary(os.path.join(nnf_rt_dir, "dxil.dll"))
        if os.path.exists(os.path.join(nnf_rt_dir, "dxcompiler.dll")):
            ctypes.cdll.LoadLibrary(os.path.join(nnf_rt_dir, "dxcompiler.dll"))
        self.libnnf = ctypes.cdll.LoadLibrary(self.libnnf_path)
        if hasattr(self.libnnf, "kernel_entry_host"):
            self.kernel_entry = self.libnnf.kernel_entry_host
        elif hasattr(self.libnnf, "kernel_entry"):
            self.kernel_entry = self.libnnf.kernel_entry
        else:
            raise Exception("No kernel_entry found in nnfurion_rt")
        device_type = self.get_device_type()
        if device_type not in self.device_type_map:
            raise Exception(f"Unknown device type: {device_type}")
        self.device_type = device_type
        init_func_name, free_func_name = self.device_type_map[device_type]
        self.nnf_rt_init = getattr(self.libnnf, init_func_name, None)
        self.nnf_rt_free = getattr(self.libnnf, free_func_name, None)
        if self.nnf_rt_init:
            with cd(nnf_rt_dir):
                self.nnf_rt_init()
                self.init_flag = True

        # parse input/output
        para_info = os.path.join(nnf_rt_dir, "para_info.json")
        weights, inputs, outputs = parse_nnf_params(para_info)
        input_descs = [None] * (len(weights) + len(inputs))
        input_index = {}
        output_descs = [None] * (len(outputs))
        output_index = {}
        for weight in weights.values():
            input_descs[weight["id"]] = IODescription(weight["name"],
                                                      weight["shape"],
                                                      weight["dtype"])
            input_index[weight["name"]] = weight["id"]
        for input in inputs.values():
            input_descs[input["id"]] = IODescription(input["name"],
                                                     input["shape"],
                                                     input["dtype"])
            input_index[input["name"]] = input["id"]
        for output in outputs.values():
            output_descs[output["id"]] = IODescription(output["name"],
                                                       output["shape"],
                                                       output["dtype"])
            output_index[output["name"]] = output["id"]
        if None in input_descs:
            raise Exception("Missed input index in para_info.json")
        if None in output_descs:
            raise Exception("Missed output index in para_info.json")
        self.input_descs = input_descs
        self.input_index = input_index
        self.output_descs = output_descs
        self.output_index = output_index

    def get_device_type(self):
        if not hasattr(self.libnnf, "get_device_type"):
            raise Exception("No get_device_type in nnfusion_rt")
        self.libnnf.get_device_type.restype = ctypes.c_int
        return self.libnnf.get_device_type()

    def get_inputs(self):
        return tuple(self.input_descs)

    def get_outputs(self):
        return tuple(self.output_descs)

    def __del__(self):
        nnf_rt_free = getattr(self, "nnf_rt_free", None)
        if self.init_flag and nnf_rt_free:
            nnf_rt_free()
            self.init_flag = False

    def __call__(self, *args, **kwargs):
        # self.feed_tensors(*args, **kwargs)
        self.feed_data(*args, **kwargs)

    def feed_data(self, inputs, outputs, strict=True):
        """
        Execute the kernel_entry in nnf runtime

        Parameters:
            inputs: a dict from name to nnf DataFormat
            outputs: a dict from name to nnf DataFormat
            strict: False if allow unused inputs/outputs

        Returns:
            None
        """
        signature = [None] * (len(self.input_descs) + len(self.output_descs))
        params = [None] * (len(self.input_descs) + len(self.output_descs))
        for name, data_format in inputs.items():
            if name in self.input_index:
                index = self.input_index[name]
                if data_format.shape != self.input_descs[
                        index].shape or data_format.dtype != self.input_descs[
                            index].dtype:
                    raise Exception(
                        f"Shape or type mismatch for NNFusion model input {name}, expect [{self.input_descs[index].shape}, {self.input_descs[index].dtype}], feed [{data_format.shape}, {data_format.dtype}]"
                    )
                signature[index] = data_format.pointer_type
                params[index] = data_format.pointer
            else:
                if strict:
                    raise Exception(f"Unused input {name}")
        for name, data_format in outputs.items():
            if name in self.output_index:
                index = self.output_index[name]
                if data_format.shape != self.output_descs[
                        index].shape or data_format.dtype != self.output_descs[
                            index].dtype:
                    raise Exception(
                        f"Shape or type mismatch for NNFusion model output {name}, expect [{self.output_descs[index].shape}, {self.output_descs[index].dtype}], feed [{data_format.shape}, {data_format.dtype}]"
                    )
                signature[len(self.input_descs) +
                          index] = data_format.pointer_type
                params[len(self.input_descs) + index] = data_format.pointer
            else:
                if strict:
                    raise Exception(f"Unused output {name}")
        self.feed_pointers(signature, params)

    def alloc_output_buffer(self):
        return tuple(desc.get_torch_cuda_buffer() for desc in self.output_descs) 

    def feed_pointers(self, signature, params):
        self.kernel_entry.argtypes = signature
        self.kernel_entry(*params)