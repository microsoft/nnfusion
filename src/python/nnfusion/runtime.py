import filecmp
import os
import tempfile

import torch
import torch.onnx

from .data_format import cast_pytorch_tensor
from .executor import Executor
from .session import build, codegen, modify_nnfusion_rt
from .utils import get_sha256_of_file, get_sha256_of_str


class NNFusionRT:
    def __init__(self, model, config, signature):
        """
        Parameters:
            model: the `torch.nn.Module` to be compiled.
            config: nnfusion compilation config
            signature (str): signature of model so that we can reuse compiled
                kernel (if any).
        """

        self.model = model
        self.weight_dict = {
            name: cast_pytorch_tensor(tensor)
            for name, tensor in model.state_dict().items()
        }

        self.root_dir = os.path.join("nnf-kernels", signature)
        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir)

        self.compile_flag = self._get_compile_flag(config)
        self.executor = None

    def compile(self, inputs, outputs):
        """
        Perform nnfusion codegen and compilation for target input sizes.
        Skip if a kernel with the same signature is found.

        Parameters:
            inputs: a list of model inputs.
            outputs: a list of model outputs.
        """

        def export_onnx(fname):
            input_names = ["input" + str(i) for i in range(len(inputs))]
            output_names = ["output" + str(i) for i in range(len(outputs))]
            torch.onnx.export(self.model, inputs, fname,
                              input_names=input_names,
                              output_names=output_names,
                              export_params=False,
                              )  # , opset_version=11)

        def check_if_need_build():
            """
            Note that this function assume no hash collision 
            """
            need_build = False

            # Compare onnx file to check if modified
            with tempfile.TemporaryDirectory(dir=self.root_dir) as tmp:
                temp_onnx_path = os.path.join(tmp, "temp.onnx")
                export_onnx(temp_onnx_path)

                onnx_hash = get_sha256_of_file(temp_onnx_path)
                onnx_dir = os.path.join(self.root_dir, onnx_hash)

                flag_hash = get_sha256_of_str(self.compile_flag)
                flag_dir = os.path.join(onnx_dir, flag_hash)
                if not os.path.isdir(flag_dir):
                    os.makedirs(flag_dir)

                onnx_path = os.path.join(onnx_dir, "model.onnx")
                if not os.path.exists(onnx_path):
                    os.link(temp_onnx_path, onnx_path)
                    need_build = True

                nnf_dir = os.path.join(flag_dir, "nnfusion_rt/cuda_codegen")
                if not os.path.exists(os.path.join(nnf_dir, 'libnnfusion_naive_rt.so')):
                    need_build = True

            return need_build, onnx_path, flag_dir, nnf_dir

        def do_compile(onnx_path, work_dir, nnf_dir):
            codegen(onnx_path, self.compile_flag, work_dir)
            modify_nnfusion_rt(nnf_dir)
            build(nnf_dir)

        need_build, onnx_path, work_dir, nnf_dir = check_if_need_build()
        if need_build:
            do_compile(onnx_path, work_dir, nnf_dir)

        self.executor = Executor(nnf_dir, device=inputs[0].device)

    def run(self, inputs, outputs):
        """
        Perform the computation. The result will be saved in `outputs`.

        Parameters:
            inputs: the input tensor(s). Can be a list or tuple.
            outputs: the output tensor(s). Can be a list or tuple.
        """
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]

        in_dict = dict(self.weight_dict, **{
            f'input{i}': cast_pytorch_tensor(tensor)
            for i, tensor in enumerate(inputs)
        })
        out_dict = {
            f'output{i}': cast_pytorch_tensor(tensor)
            for i, tensor in enumerate(outputs)
        }
        self.executor(in_dict, out_dict, strict=False)

    def _get_compile_flag(self, config):
        return " ".join([
            "-f onnx",
            config.to_flag()
        ])
