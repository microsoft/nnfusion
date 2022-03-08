import filecmp
import inspect
import os
import re
import tempfile
from pathlib import Path

import torch
import torch.onnx

from .data_format import cast_pytorch_tensor
from .executor import Executor
from .jit_utils import TorchModule
from .session import build, codegen, modify_nnfusion_rt


class NNFusionRT:
    def __init__(self, model, server="127.0.0.1:8880", steps=1000):
        self.model = model

        self.workdir = os.path.join("tmp", self._signature)
        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir)

        self.onnx_path = os.path.join(self.workdir, "model.onnx")
        self.rt_dir = os.path.join(self.workdir, "nnfusion_rt/cuda_codegen")

        self.compile_flag = self._get_compile_flag(steps, server)
        self.executor = None
        self._reserved_mem = None

    def compile(self, inputs, outputs, force_build=False):

        def export_onnx(fname):
            input_names = ["input" + str(i) for i in range(len(inputs))]
            output_names = ["output" + str(i) for i in range(len(outputs))]
            torch.onnx.export(self.model, inputs, fname,
                              input_names=input_names,
                              output_names=output_names)  # , opset_version=11)

        def check_if_need_build():
            if not os.path.exists(self.onnx_path):
                return True

            # Compare onnx file to check if modified
            with tempfile.TemporaryDirectory(dir=self.workdir) as tmp:
                temp_onnx_path = os.path.join(tmp, "temp.onnx")
                export_onnx(temp_onnx_path)

                if not filecmp.cmp(temp_onnx_path, self.onnx_path):
                    # Replace the original to avoid exporting onnx twice
                    os.remove(self.onnx_path)
                    os.link(temp_onnx_path, self.onnx_path)
                    return True
            return False

        def do_compile():
            if not os.path.exists(self.onnx_path):
                export_onnx(self.onnx_path)

            codegen(self.onnx_path, self.compile_flag, self.workdir)
            modify_nnfusion_rt(self.rt_dir)
            build(self.rt_dir)

        if force_build and os.path.exists(self.onnx_path):
            os.remove(self.onnx_path)

        if check_if_need_build():
            do_compile()

        self.executor = Executor(self.rt_dir, device=inputs[0].device)


    def run(self, inputs, outputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]

        in_dict = {
            desc.name: cast_pytorch_tensor(tensor)
            for desc, tensor in zip(self.executor.get_inputs(), inputs)
        }
        out_dict = {
            desc.name: cast_pytorch_tensor(tensor)
            for desc, tensor in zip(self.executor.get_outputs(), outputs)
        }
        self.executor(in_dict, out_dict)

    def _get_compile_flag(self, tuning_step, codegen_server):
        return " ".join([
            "-f onnx",
            "-fextern_result_memory=1",
            "-fkernel_tuning_steps=" + str(tuning_step),
            "-fir_based_fusion=1",
            "-ffunction_codegen=1",
            "-fkernel_fusion_level=0",
            # "-fantares_mode=1",
            # f"-fantares_codegen_server={codegen_server}",
            "-fblockfusion_level=0",
        ])

    @property
    def _signature(self):
        """
        Signature of a function or torch.nn.Module instance to detect reusable
        kernel.
        """
        def get_qualname():
            if isinstance(self.model, TorchModule):
                name = self.model.func.__qualname__
            else:
                name = self.model.__class__.__qualname__
            # Remove special chars to avoid the trouble of dealing with paths
            return re.sub("[<>]", "", name)

        def get_path():
            # Avoid collision between different files
            if isinstance(self.model, TorchModule):
                obj_path = inspect.getsourcefile(self.model.func)
            else:
                obj_path = inspect.getsourcefile(self.model.__class__)
            relpath = os.path.relpath(obj_path)
            return "-".join(Path(os.path.splitext(relpath)[0]).parts)

        return "-".join((get_path(), get_qualname()))
