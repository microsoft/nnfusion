# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import platform
from ctypes import *
from . import dtypes
from .utils import cd


def find_nnf_rt(nnf_rt_dir):
    def is_nnf_rt(file_name):
        if platform.system().lower() == "linux":
            return file_name.startswith("libnnf") and file_name.endswith("rt.so")
        elif platform.system().lower() == "windows":
            return file_name.startswith("nnf") and file_name.endswith("rt.dll")
        else:
            return False

    for file_name in os.listdir(nnf_rt_dir):
        if is_nnf_rt(file_name):
            return os.path.join(nnf_rt_dir, file_name)
    return ""

def deduce_device_type(nnf_rt_dir):
    nnf_rt_dir = os.path.abspath(nnf_rt_dir)
    if "cuda_codegen" in nnf_rt_dir:
        return "cuda"
    elif "cpu_codegen" in nnf_rt_dir:
        return "cpu"
    elif "dxcompute_codegen" in nnf_rt_dir:
        return "hlsl"
    return ""


class Executor(object):
    """
    Python wrapper for NNFusion runtime.
    Executor loads a compiled nnf_rt dynamic lib, provide a __call__ func to
    execute the "kernel_entry" in nnf_rt with given tensors.
    """
    def __init__(self, nnf_rt_dir, device_type=None):
        """
        Parameters:
            nnf_rt_dir: A full string path to nnfusion runtime,
                it's usually like "codegen_root/nnfusion_rt/cuda_codegen".
            device_type: one of ("cpu", "cuda", "hlsl"). If not provided, 
                device type will be infered from folder name.
        """
        nnf_rt_dir = os.path.abspath(nnf_rt_dir)
        self.libnnf_path = find_nnf_rt(nnf_rt_dir)
        if self.libnnf_path == "":
            raise Exception("nnf_rt lib not found in folder {}".format(nnf_rt_dir))
        
        self.device_type = device_type
        if self.device_type is None:
            self.device_type = deduce_device_type(nnf_rt_dir)
            if not self.device_type:
                raise Exception("Cannot deduce device type")

        self.libnnf = cdll.LoadLibrary(self.libnnf_path)
        with cd(nnf_rt_dir):
            self._init()

    def _init(self):
        if self.device_type == "cpu":
            self.libnnf.cpu_init()
        elif self.device_type == "cuda":
            self.libnnf.cuda_init()
        elif self.device_type == "hlsl":
            hlsl_root = os.getcwd()
            while not os.path.isdir(os.path.join(hlsl_root, "HLSL")):
                hlsl_root = os.path.join(hlsl_root, os.pardir)
            with cd(hlsl_root):
                self.libnnf.hlsl_init()

    def __del__(self):
        self._free()

    def _free(self):
        if self.device_type == "cpu":
            self.libnnf.cpu_free()
        elif self.device_type == "cuda":
            self.libnnf.cuda_free()
        elif self.device_type == "hlsl":
            self.libnnf.hlsl_free()

    def __call__(self, *args, **kwargs):
        """
        Execute the kernel_entry in nnf runtime

        Parameters:
            args: a list of PyTorch tensor, include inputs and outputs,
                presented with the sequence in kernel entry,
                should be exactly matched the type/shape/device.

        Returns:
            None
        """
        self.feed_tensors(*args, **kwargs)

    def feed_tensors(self, tensors):
        self.feed_pointers(dtypes.deduce_signatrue(tensors),
                           dtypes.get_data_addr(tensors))

    def feed_pointers(self, signature, params):
        self.libnnf.argtypes = signature
        self.libnnf.kernel_entry(*params)