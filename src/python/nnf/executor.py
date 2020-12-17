# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from ctypes import *
from . import dtypes
from .utils import cd


def find_nnf_rt(nnf_rt_dir):
    for file_name in os.listdir(nnf_rt_dir):
        if file_name.startswith("libnnf") and file_name.endswith("rt.so"):
            return os.path.join(nnf_rt_dir, file_name)
    return ""


class Executor(object):
    """
    Python wrapper for NNFusion runtime.
    Executor loads a compiled nnf_rt dynamic lib, provide a __call__ func to
    execute the "kernel_entry" in nnf_rt with given tensors.
    """
    def __init__(self, nnf_rt_dir):
        """
        Parameters:
            nnf_rt_dir: A full string path to nnfusion runtime,
                it's usually like "codegen_root/nnfusion_rt/cuda_codegen".
        """
        self.libnnf_path = find_nnf_rt(nnf_rt_dir)
        assert self.libnnf_path != "", "libnnf not found in path {}".format(
            nnf_rt_dir)
        self.libnnf = cdll.LoadLibrary(self.libnnf_path)
        with cd(nnf_rt_dir):
            self._init()

    def _init(self):
        if "cpu" in self.libnnf_path:
            self.libnnf.cpu_init()
        else:
            self.libnnf.cuda_init()

    def __del__(self):
        self._free()

    def _free(self):
        if "cpu" in self.libnnf_path:
            self.libnnf.cpu_free()
        else:
            self.libnnf.cuda_free()

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