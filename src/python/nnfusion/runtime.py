import os

import torch
import torch.onnx

from nnfusion.data_format import cast_pytorch_tensor
from nnfusion.executor import Executor
from nnfusion.session import build, codegen, modify_nnfusion_rt


class NNFusionRT:
    def __init__(self, model, inputs, outputs, server="127.0.0.1:8880",
                 steps=1000):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.workdir = "./tmp" + str(model.__class__.__name__) + "/"
        if not os.path.isdir(self.workdir):
            os.mkdir(self.workdir)
        self.codegen_server = server
        self.tuning_step = steps

    def compile(self, buildall=False, rebuild=False):
        input_names = ["input" + str(i) for i in range(len(self.inputs))]
        output_names = ["output" + str(i) for i in range(len(self.outputs))]
        rt_dir = os.path.join(self.workdir, "nnfusion_rt/cuda_codegen")
        if buildall:
            torch.onnx.export(self.model,
                              self.inputs,
                              self.workdir + "stencil.onnx",
                              verbose=True,
                              input_names=input_names,
                              output_names=output_names)  # , opset_version=11)
            codegen(os.path.join(self.workdir, "stencil.onnx"),
                    " ".join([
                        "-f onnx",
                        "-fextern_result_memory=1",
                        "-fkernel_tuning_steps=" + str(self.tuning_step),
                        "-fir_based_fusion=1",
                        "-fkernel_fusion_level=0",
                        # "-fantares_mode=1",
                        # f"-fantares_codegen_server={self.codegen_server}",
                        "-fblockfusion_level=0"]),
                    self.workdir)
            modify_nnfusion_rt(rt_dir)
            build(rt_dir)
        if rebuild:
            build(rt_dir)
        self.executor = Executor(rt_dir)
        self.input_name = self.executor.get_inputs()[0].name
        self.output_name = self.executor.get_outputs()[0].name

    def run(self, input, output):
        in_dict = {}
        out_dict = {}
        if isinstance(input, list):
            # assert len(self.executor.get_inputs()) == len(input)
            for i in range(len(input)):
                in_dict[self.executor.get_inputs(
                )[i].name] = cast_pytorch_tensor(input[i])
        else:
            in_dict = {self.input_name: cast_pytorch_tensor(input)}
        if isinstance(output, list):
            # assert len(self.executor.get_outputs()) == len(output)
            for i in range(len(output)):
                out_dict[self.executor.get_outputs(
                )[i].name] = cast_pytorch_tensor(output[i])
        else:
            out_dict = {self.output_name: cast_pytorch_tensor(output)}

        self.executor(in_dict, out_dict)
