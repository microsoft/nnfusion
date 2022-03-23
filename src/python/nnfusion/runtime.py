import filecmp
import os
import tempfile

import torch
import torch.onnx

from .data_format import cast_pytorch_tensor
from .executor import Executor
from .session import build, codegen, modify_nnfusion_rt


class NNFusionRT:
    def __init__(self, model, config, signature):
        """
        Parameters:
            model: the `torch.nn.Module` to be compiled.
            config: nnfusion compilation config
            signature: signature of model so that we can reuse compiled
                kernel (if any).
        """

        self.model = model
        self.weight_dict = {
            name: cast_pytorch_tensor(tensor)
            for name, tensor in model.state_dict().items()
        }

        self.workdir = os.path.join("tmp", signature)
        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir)

        self.onnx_path = os.path.join(self.workdir, "model.onnx")
        self.rt_dir = os.path.join(self.workdir, "nnfusion_rt/cuda_codegen")

        self.compile_flag = self._get_compile_flag(config)
        self.executor = None

    def compile(self, inputs, outputs, force_build=False):
        """
        Perform nnfusion codegen and compilation for target input sizes.
        Skip if a kernel with the same signature is found.

        Parameters:
            inputs: a list of model inputs.
            outputs: a list of model outputs.
            force_build: whether to replace the previous kernel (if any).
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
            if not os.path.exists(self.onnx_path):
                return True

            if not os.path.exists(os.path.join(self.rt_dir, 'main_test')):
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
