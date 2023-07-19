# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import ctypes
import json
import os
import platform
import torch

from .data_format import HLSLTensor, cast_pytorch_tensor, cast_hlsl_tensor
from .description import IODescription
from .dtypes import to_torch_type
from .utils import cd


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
            if len(shape) == 0:
                shape = [1]

            if "symbolic_shape" in desc:
                symbolic_shape = desc["symbolic_shape"]
            else:
                symbolic_shape = shape
            if len(symbolic_shape) == 0:
                symbolic_shape = [1]

            out[name] = {
                "name": name,
                "id": int(index),
                "dtype": dtype,
                "shape": shape,
                "symbolic_shape": symbolic_shape,
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

    def __init__(self, nnf_rt_dir, device=None):
        """
        Parameters:
            nnf_rt_dir: A full string path to nnfusion runtime,
                it's usually like "codegen_root/nnfusion_rt/cuda_codegen".
            device: A device type (`torch.device`) that is used for workspace
                memory reservation (if needed) by nnfusion runtime.
        """
        nnf_rt_dir = os.path.abspath(nnf_rt_dir)
        self.libnnf_path = find_nnf_rt(nnf_rt_dir)
        if self.libnnf_path == "":
            raise Exception(
                "nnf_rt lib not found in folder {}".format(nnf_rt_dir))

        # prepare init/free/kernel_entry
        self.init_flag = False
        # dxil.dll and dxcompiler.dll must be manually imported
        self.lib_dxil, self.lib_dxcompiler = None, None
        if os.path.exists(os.path.join(nnf_rt_dir, "dxil.dll")):
            self.lib_dxil = ctypes.cdll.LoadLibrary(os.path.join(nnf_rt_dir, "dxil.dll"))
        if os.path.exists(os.path.join(nnf_rt_dir, "dxcompiler.dll")):
            self.lib_dxcompiler = ctypes.cdll.LoadLibrary(os.path.join(nnf_rt_dir, "dxcompiler.dll"))
        # antares.dll must be loaded after dxil.dll and dxcompiler.dll
        if os.path.exists(os.path.join(nnf_rt_dir, "antares.dll")):
            HLSLTensor.init_antares_lib(os.path.join(nnf_rt_dir, "antares.dll"))
        self.libnnf = ctypes.cdll.LoadLibrary(self.libnnf_path)
        if hasattr(self.libnnf, "kernel_entry_host"):
            self.kernel_entry = self.libnnf.kernel_entry_host
            self.host_mode = True
        elif hasattr(self.libnnf, "kernel_entry"):
            self.kernel_entry = self.libnnf.kernel_entry
            self.host_mode = False
        else:
            raise Exception("No kernel_entry found in nnfusion_rt")
        device_type = self.get_device_type()
        if device_type not in self.device_type_map:
            raise Exception(f"Unknown device type: {device_type}")
        self.device_type = device_type
        init_func_name, free_func_name = self.device_type_map[device_type]
        self.nnf_rt_init = getattr(self.libnnf, init_func_name, None)
        self.nnf_rt_free = getattr(self.libnnf, free_func_name, None)

        if self.nnf_rt_init:
            with cd(nnf_rt_dir):
                workspace_ptr = self._maybe_reserve_mem(device)
                if workspace_ptr is not None:
                    self.nnf_rt_init(workspace_ptr)
                else:
                    self.nnf_rt_init()
                self.init_flag = True

        # parse input/output
        para_info = os.path.join(nnf_rt_dir, "para_info.json")
        weights, inputs, outputs = parse_nnf_params(para_info)
        input_descs = [None] * (len(weights) + len(inputs))
        input_index = {}
        output_descs = [None] * (len(outputs))
        output_index = {}
        expected_inputs = {}
        expected_outputs = {}

        def convert_sym_shape(in_shape):
            out_shape = []
            for dim in in_shape:
                if isinstance(dim, str) and not dim.isdigit():
                    out_shape.append(dim)
                else:
                    out_shape.append(int(dim))
            return tuple(out_shape)


        for weight in weights.values():
            input_descs[weight["id"]] = IODescription(weight["name"],
                                                      weight["shape"],
                                                      weight["dtype"])
            input_index[weight["name"]] = weight["id"]
            expected_inputs[weight["name"]] = {
                "shape": convert_sym_shape(weight["symbolic_shape"]),
                "dtype": weight["dtype"]
            }
        for input in inputs.values():
            input_descs[input["id"]] = IODescription(input["name"],
                                                     input["shape"],
                                                     input["dtype"])
            input_index[input["name"]] = input["id"]
            expected_inputs[input["name"]] = {
                "shape": convert_sym_shape(input["symbolic_shape"]),
                "dtype": input["dtype"]
            }
        for output in outputs.values():
            output_descs[output["id"]] = IODescription(output["name"],
                                                       output["shape"],
                                                       output["dtype"])
            output_index[output["name"]] = output["id"]
            expected_outputs[output["name"]] = {
                "shape": convert_sym_shape(output["symbolic_shape"]),
                "dtype": output["dtype"]
            }
        if None in input_descs:
            raise Exception("Missed input index in para_info.json")
        if None in output_descs:
            raise Exception("Missed output index in para_info.json")
        self.input_descs = input_descs # list[IODesc]
        self.input_index = input_index # dict: name -> 0-based index
        self.output_descs = output_descs # list[IODesc]
        self.output_index = output_index # dict: name -> 0-based index
        self.expected_inputs = expected_inputs
        self.expected_outputs = expected_outputs

    def set_symbol(self, key, value):
        func_name = "set_" + key
        func = getattr(self.libnnf, func_name, None)
        if func is None:
            raise Exception(f"{func_name} doesn't exist in nnf_rt")
        func.argtypes = [ctypes.c_int64]
        func(value)

    def get_symbol(self, key):
        func_name = "get_" + key
        func = getattr(self.libnnf, func_name, None)
        if func is None:
            raise Exception(f"{func_name} doesn't exist in nnf_rt")
        func.restype = ctypes.c_int64
        return func()

    def fix_shape(self, sym_shape):
        shape = []
        for dim in sym_shape:
            if isinstance(dim, str):
                sym_name = dim
                dim = self.get_symbol(dim)
                if dim == -1:
                    raise Exception(f"get_symbol({sym_name}) returns -1, provided input shape may not be supported by this model")
            shape.append(dim)
        return shape

    def set_inputs(self, input_shape):
        record = {}
        for name, info in self.expected_inputs.items():
            for idx, dim in enumerate(info["shape"]):
                if isinstance(dim, str):
                    if dim not in record:
                        record[dim] = input_shape[name][idx]
                        self.set_symbol(dim, input_shape[name][idx])
                    else:
                        if record[dim] != input_shape[name][idx]:
                            raise Exception(f"Inconsistent value for symbol {dim}")

    def allocate_outputs(self):
        output_dict = {}
        for name, info in self.expected_outputs.items():
            if info["dtype"] == "int16":
                print("[Warning] int16 is ambiguous which might be bool or int16")
            shape = self.fix_shape(info["shape"])
            if self.host_mode:
                # host mode leverage pytorch tensor as storage
                torch_tensor = torch.empty(size=shape, dtype=to_torch_type(info["dtype"]))
                output_dict[name] = cast_pytorch_tensor(torch_tensor)
            else:
                if self.device_type == 0:
                    # cuda device
                    torch_tensor = torch.empty(size=shape, dtype=to_torch_type(info["dtype"]), device="cuda")
                    output_dict[name] = cast_pytorch_tensor(torch_tensor)
                elif self.device_type == 3:
                    # hlsl device
                    output_dict[name] = cast_hlsl_tensor(HLSLTensor(shape, info["dtype"]))
                else:
                    raise Exception("only support allocate device tensor on cuda/hlsl backend.")
        return output_dict


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
        self._release_dynamic_libraries()

    def __call__(self, *args, **kwargs):
        # self.feed_tensors(*args, **kwargs)
        return self.feed_data(*args, **kwargs)

    def _release_dynamic_libraries(self):
        # There are four DLLs loaded: antares.dll, dxcompiler.dll, dxil.dll, nnfusion_rt.dll
        # But the antares.dll is warpped in the class HLSLTensor and loaded only once during process
        # Thus the other three DLLs are loaded for each test case and need to be explicitly released
        if platform.system().lower() == "windows":
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            # release nnfusion_rt.dll
            handle = self.libnnf._handle
            del self.libnnf
            ctypes.windll.kernel32.FreeLibrary(handle)
            # release dxil.dll
            if self.lib_dxil:
                handle = self.lib_dxil._handle
                del self.lib_dxil
                ctypes.windll.kernel32.FreeLibrary(handle)
            # release dxcompiler.dll
            if self.lib_dxcompiler:
                handle = self.lib_dxcompiler._handle
                del self.lib_dxcompiler
                ctypes.windll.kernel32.FreeLibrary(handle)
        elif platform.system().lower() == "linux":
            pass  # TODO: release libraries in linux
        return

    def _dict_to_pointer_list(self, inputs, outputs, strict=True):
        signature = [None] * (len(self.input_descs) + len(self.output_descs))
        params = [None] * (len(self.input_descs) + len(self.output_descs))

        def check_compatible(feed, expect):
            if data_format.dtype != expect["dtype"]:
                return False
            if len(feed.shape) != len(expect["shape"]):
                return False
            for feed_dim, expect_dim in zip(feed.shape, expect["shape"]):
                if isinstance(expect_dim, str):
                    # ignore symbolic dim
                    continue
                if feed_dim != expect_dim:
                    return False
            return True

        for name, data_format in inputs.items():
            if name in self.expected_inputs:
                expect_input = self.expected_inputs[name]
                if not check_compatible(data_format, expect_input):
                    raise Exception(
                        f"Shape or type mismatch for NNFusion model input {name}, expect [{expect_input['shape']}, {expect_input['dtype']}], feed [{data_format.shape}, {data_format.dtype}]"
                    )
                index = self.input_index[name]
                signature[index] = data_format.pointer_type
                params[index] = data_format.pointer
            else:
                if strict:
                    raise Exception(f"Unused input {name}")
        for name, data_format in outputs.items():
            if name in self.expected_outputs:
                expect_output = self.expected_outputs[name]
                if not check_compatible(data_format, expect_output):
                    raise Exception(
                        f"Shape or type mismatch for NNFusion model output {name}, expect [{expect_output['shape']}, {expect_output['dtype']}], feed [{data_format.shape}, {data_format.dtype}]"
                    )
                index = self.output_index[name]
                signature[len(self.input_descs) +
                          index] = data_format.pointer_type
                params[len(self.input_descs) + index] = data_format.pointer
            else:
                if strict:
                    raise Exception(f"Unused output {name}")
        return signature, params

    def feed_data(self, inputs, outputs=None, strict=True):
        """
        Execute the kernel_entry in nnf runtime

        Parameters:
            inputs: a dict from name to nnf DataFormat
            outputs: a dict from name to nnf DataFormat
            strict: False if allow unused inputs/outputs

        Returns:
            outputs: a dict from name to nnf DataFormat
        """
        input_shape = {name: value.shape for name, value in inputs.items()}
        self.set_inputs(input_shape)
        if outputs is None:
            outputs = self.allocate_outputs()
        signature, params = self._dict_to_pointer_list(inputs, outputs, strict=strict)
        self.feed_pointers(signature, params)
        return outputs

    def feed_pointers(self, signature, params):
        self.kernel_entry.argtypes = signature
        self.kernel_entry(*params)
        # synchronize should be included in kernel_entry
        if HLSLTensor.antares_lib:
            HLSLTensor.antares_lib.dxStreamSynchronize(None)

    def _maybe_reserve_mem(self, device):
        get_workspace_size = getattr(self.libnnf, 'get_workspace_size', None)
        if get_workspace_size is None:
            return None
        get_workspace_size.restype = ctypes.c_size_t
        n_byte = get_workspace_size()
        if not n_byte:
            return None

        self._reserved_mem = torch.empty(n_byte,
                                         dtype=torch.int8, device=device)
        return cast_pytorch_tensor(self._reserved_mem).pointer
