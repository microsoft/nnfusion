# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
from ctypes import cdll, c_char_p, c_int, byref
import dtypes


class Runtime:
    def __init__(self):
        # detect existed library of nnfusion runtime
        libnnf_rt = "none"
        if "LIB_NNF_RT" not in os.environ.keys():
            logging.info(
                "libnnfusion_rt is not specified \
                by system enviroment variable: LIB_NNF_RT")
            default_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..')
            for file in os.listdir(default_path):
                if file.startswith("libnnf") and file.endswith("rt.so"):
                    libnnf_rt = os.path.join(default_path, file)
                    logging.info("libnnfusion_rt library detected")
            self.default_path = default_path
        else:
            libnnf_rt = os.environ["LIB_NNF_RT"]

        if not os.path.exists(libnnf_rt):
            raise Exception("libnnfusion_rt: %s is not existed!" % (libnnf_rt))

        try:
            libnnf = cdll.LoadLibrary(libnnf_rt)
        except Exception:
            raise Exception("libnnfusion_rt: %s is not loaded!" % (libnnf_rt))

        # member of session
        self.libnnf_path = libnnf_rt
        self.libnnf = libnnf

    # call for init session
    def init(self, plan_file_path):
        if "cpu" in self.libnnf_path:
            self.libnnf.cpu_init()
        else:
            self.libnnf.cuda_init(c_char_p(plan_file_path.encode('utf-8')))

    def device_id(self):
        device_id = c_int()
        self.libnnf.sc_get_device_id(byref(device_id))
        return device_id.value

    def world_size(self):
        world_size = c_int()
        self.libnnf.sc_get_world_size(byref(world_size))
        return world_size.value

    def feed(self, tensors=[], signature=(), params=()):
        if tensors is not []:
            self.libnnf.argtypes = dtypes.deduce_signatrue(tensors)
            self.libnnf.kernel_entry(*(dtypes.get_data_addr(tensors)))
        else:
            self.libnnf.argtypes = signature
            self.libnnf.kernel_entry(*params)

    def free(self):
        if "cpu" in self.libnnf_path:
            self.libnnf.cpu_free()
        else:
            self.libnnf.cuda_free()

        del self.libnnf
        del self.libnnf_path
