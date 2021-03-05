# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import print_function
import os
import copy
import tempfile
import torch
import json
import logging
from .utils import cd, execute
from .executor import Executor
from .description import IODescription, ModelDescription, generate_sample

logger = logging.getLogger(__name__)


def generate_output_desc(model, input_desc, device="cpu"):
    fake_inputs = [generate_sample(desc, device) for desc in input_desc]
    model_copy = copy.deepcopy(model).to(device)
    out = model_copy(*fake_inputs)
    if isinstance(out, torch.Tensor):
        out = (out, )
    return tuple(
        IODescription("output_{}".format(i), t.shape, t.dtype)
        for i, t in enumerate(out))


def convert_model_to_onnx(model, model_desc, device, file_name):
    model.to(device)
    input_names = [input.name_ for input in model_desc.inputs_]
    output_names = [output.name_ for output in model_desc.outputs_]
    sample_inputs = [
        generate_sample(input, device) for input in model_desc.inputs_
    ]
    sample_outputs = [
        generate_sample(output, device) for output in model_desc.outputs_
    ]
    # note: onnx exporter might have side effect, so copy a new model
    torch.onnx.export(copy.deepcopy(model).to(device),
                      tuple(sample_inputs),
                      file_name,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=12,
                      _retain_param_name=True,
                      example_outputs=tuple(sample_outputs),
                      do_constant_folding=False)

    return model


def codegen(model_path, flags, output_dir):
    model_path = os.path.abspath(model_path)
    with cd(output_dir):
        command = "{} {} {}".format("nnfusion", model_path, flags)
        execute(command)


def modify_nnfusion_rt(rt_dir):
    with cd(rt_dir):
        # remove cudaDevice reset in cuda_init()
        command = "sed -i '/cudaDeviceReset()/s:^://:'" + " " + "nnfusion_rt.cu"
        execute(command)


def build(rt_dir):
    with cd(rt_dir):
        command = "cmake ."
        execute([command])

        command = "make -j"
        execute(command)


def parse_nnf_params(param_file):
    with open(param_file) as f:
        nnf_params = json.load(f)

    weights = {}
    for name, desc in nnf_params.get("weight", dict()).items():
        index = desc["id"].split("inputs[")[1].split("]")[0]
        dtype = desc["id"][2:].split("*")[0]
        shape = desc["shape"]
        weights[name] = {
            "name": name,
            "id": index,
            "dtype": dtype,
            "shape": shape,
            "raw_id": desc["id"],
            "nnf_name": desc["name"]
        }

    inputs = {}
    for name, desc in nnf_params.get("input", dict()).items():
        index = desc["id"].split("inputs[")[1].split("]")[0]
        dtype = desc["id"][2:].split("*")[0]
        shape = desc["shape"]
        inputs[name] = {
            "name": name,
            "id": index,
            "dtype": dtype,
            "shape": shape,
            "raw_id": desc["id"],
            "nnf_name": desc["name"]
        }

    outputs = {}
    for name, desc in nnf_params.get("output", dict()).items():
        index = desc["id"].split("outputs[")[1].split("]")[0]
        dtype = desc["id"][2:].split("*")[0]
        shape = desc["shape"]
        outputs[name] = {
            "name": name,
            "id": index,
            "dtype": dtype,
            "shape": shape,
            "raw_id": desc["id"],
            "nnf_name": desc["name"]
        }

    return weights, inputs, outputs


def str2dtype(ss):
    if ss == "float":
        return torch.float
    elif ss == "int64_t":
        return torch.int64
    else:
        raise Exception("Unsupported type: {}".format(ss))


def validate_nnf_weights(nnf_weights, torch_weights):
    assert set(nnf_weights.keys()) == set(torch_weights.keys())
    for name, desc in nnf_weights.items():
        torch_tensor = torch_weights[name]
        assert tuple(desc["shape"]) == tuple(torch_tensor.shape)
        assert str2dtype(desc["dtype"]) == torch_tensor.dtype


def validate_nnf_inputs(nnf_inputs, inputs_desc):
    assert set(nnf_inputs.keys()) == {desc.name_ for desc in inputs_desc}
    for desc in inputs_desc:
        assert tuple(desc.shape_) == tuple(nnf_inputs[desc.name_]["shape"])
        assert desc.dtype_ == str2dtype(nnf_inputs[desc.name_]["dtype"])


def validate_nnf_outputs(nnf_outputs, outputs_desc):
    for desc in outputs_desc:
        assert tuple(desc.shape_) == tuple(nnf_outputs[desc.name_]["shape"])
        assert desc.dtype_ == str2dtype(nnf_outputs[desc.name_]["dtype"])


class Session(object):
    """
    A pipeline converting PyTorch model to NNFusion with specific inputs,
    provide a __call__ func to replace the origin model forward.
    """
    def __init__(self,
                 model,
                 input_desc,
                 device,
                 output_desc=None,
                 workdir=None,
                 model_format="onnx",
                 codegen_flags=None,
                 **kwargs):
        """
        Parameters:
            model: torch.nn.Module to be converted.
            input_desc: A list of IODescription representing inputs.
            device: A string representing execution device like "cuda:0",
                currently only tested against cuda device.
            output_desc: Optional, a list of IODescription representing outputs,
                if not provided, the description will be generated by executing PyTorch model.
            workdir: Optional, a string path to generated model & code, if not provided,
                model & code will be stored in a temporary folder, then be cleaned automatically .
            model_format: Intermedia model format, currently only support "onnx".
            codegen_flags: NNFusion codegen flags, 
                ref: https://github.com/microsoft/nnfusion/wiki/4.3-NNFusion-CLI-Interface#cli-flags
        """
        self._model = model
        if model_format != "onnx":
            raise Exception("{} format not supported yet".format(model_format))
        self._model_format = model_format
        self._torch_weights = {
            name: param
            for name, param in self._model.named_parameters()
        }
        self._torch_weights.update(
            {name: param
             for name, param in self._model.named_buffers()})
        self._input_desc = input_desc
        self._device = device
        if output_desc is None:
            output_desc = generate_output_desc(self._model, self._input_desc,
                                               self._device)
        else:
            # todo: validate output shape/type against real outputs
            pass
        self._output_desc = output_desc
        self._model_desc = ModelDescription(self._input_desc,
                                            self._output_desc)
        if workdir:
            workdir = os.path.expandvars(os.path.expanduser(workdir))
            self._dir_ctx = None
            self._workdir = workdir
            os.makedirs(workdir, exist_ok=True)
        else:
            self._dir_ctx = tempfile.TemporaryDirectory(prefix="nnf_")
            self._workdir = self._dir_ctx.name
        ## convert torch model to onnx
        self._onnx_model_path = os.path.join(self._workdir, "nnf.onnx")
        convert_model_to_onnx(self._model, self._model_desc, self._device,
                              self._onnx_model_path)
        torch.cuda.empty_cache()

        ## codegen
        self._codegen_flags = {"extern_result_memory": 1}
        self._codegen_flags.update(codegen_flags or {})
        self._executor = self._create_executor()

    def _create_executor(self):
        if "cuda" in self._device:
            rt_dir = os.path.join(self._workdir, "nnfusion_rt/cuda_codegen")
        elif "cpu" in self._device:
            raise Exception("CPU not supported yet")
        elif "rocm" in self._device:
            ## todo: support allocate torch tensors on ROCM device
            raise Exception("ROCm not supported yet")
        else:
            raise Exception("Unknown device {}".format(self._device))

        flags_str = "-f {} ".format(self._model_format)
        flags_str += " ".join(
            ["-f{}={}".format(k, v) for k, v in self._codegen_flags.items()])

        codegen(self._onnx_model_path, flags_str, self._workdir)
        modify_nnfusion_rt(rt_dir)
        build(rt_dir)

        param_file = os.path.join(rt_dir, "para_info.json")
        self._binding_exectuor_inputs(param_file)
        return Executor(rt_dir)

    def _binding_exectuor_inputs(self, para_info_path):
        nnf_weights, nnf_inputs, nnf_outputs = parse_nnf_params(para_info_path)
        self._feed_tensors = [None] * (len(nnf_weights) + len(nnf_inputs) +
                                       len(nnf_outputs))
        validate_nnf_inputs(nnf_inputs, self._input_desc)
        self._nnf_inputs = nnf_inputs
        if self._codegen_flags.get("training_mode"):
            validate_nnf_weights(nnf_weights, self._torch_weights)
            for name, desc in nnf_weights.items():
                self._feed_tensors[int(desc["id"])] = self._torch_weights[name]

        self._nnf_outputs_indexes = []
        if self._codegen_flags.get("extern_result_memory"):
            validate_nnf_outputs(nnf_outputs, self._output_desc)
            output_offset = len(nnf_weights) + len(nnf_inputs)
            output_names = {desc.name_ for desc in self._output_desc}
            for name, desc in nnf_outputs.items():
                self._feed_tensors[int(desc["id"]) +
                                   output_offset] = torch.ones(
                                       desc["shape"],
                                       dtype=str2dtype(desc["dtype"]),
                                       device=self._device)
                if name in output_names:
                    self._nnf_outputs_indexes.append(
                        int(desc["id"]) + output_offset)

    def __call__(self, feed_data):
        return self.run_by_nnf(feed_data)

    def run_by_pytorch(self, feed_data):
        args = [feed_data[desc.name_] for desc in self._input_desc]
        with torch.no_grad():
            out = self._model(*args)
        return out

    def run_by_nnf(self, feed_data):
        """
        Parameters:
            feed_data: a dict from name to PyTorch tensors, name should be presented in input desc.
        
        Returns:
            a list of PyTorch tensors executed by NNFusion,
            they should be the same as origin PyTorch model forward results.
        """
        for key, value in feed_data.items():
            index = int(self._nnf_inputs[key]["id"])
            self._feed_tensors[index] = value
        self._executor(tensors=self._feed_tensors)
        # assert self.is_weights_nan() is False
        return [
            self._feed_tensors[index] for index in self._nnf_outputs_indexes
        ]

    def is_weights_nan(self):
        have_nan = False
        for name, weight in self._torch_weights.items():
            if bool(torch.isnan(weight).any()) or bool(
                    torch.isinf(weight).any()):
                logger.error("Nan or inf found in {}".format(name))
                # logger.error(weight)
                have_nan = True
        return have_nan


if __name__ == "__main__":
    pass
