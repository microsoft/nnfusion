import os, sys, glob
from typing import IO, Any, Dict, List, Sequence, Union
from importlib import import_module
import numpy as np
import onnx
import torch
import git
import shutil
from onnx import AttributeProto, defs, load, TensorProto, ModelProto, NodeProto, TypeProto, numpy_helper
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner import Runner
from onnx.backend.base import Backend

global_flag_input_as_constant = False
global_flag_float_as_half = False
global_flag_float_as_double = False


class TestContext:
    root_dir : str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
    onnx_remote : str = "git@github.com:onnx/onnx.git"
    onnx_repo : str = os.path.join(root_dir, "build/onnx")
    onnx_tests : str = os.path.join(onnx_repo, "onnx/backend/test/data")
    onnx_test_kind : str = "node"
    nnfusion_bin : str = os.path.join(root_dir, "build/src/tools/nnfusion/")
    nnfusion_python : str = os.path.join(root_dir, "src/python/")
    nnfusion_workdir : str = "nnfusion_work"
    # remain unchanged
    nnfusion_argstr = "-f onnx -fmulti_shape=false -fort_folding=false -fdefault_device=CUDA -fhlsl_codegen_type=cpp -fantares_mode=true -fblockfusion_level=0 -fkernel_fusion_level=0 -fantares_codegen_server=127.0.0.1:8880 -fkernel_tuning_steps=0 -ffold_where=1 -fsymbolic=1 -fort_folding=0 -fsplit_softmax=1 -fhost_entry=0 -fir_based_fusion=1 -fextern_result_memory=1"
    antares_url = "127.0.0.1:8880"
    nnfusion_device = "CUDA"
    nnfusion_codegen_dir = "nnfusion_rt/cuda_codegen"

    def __init__(self, ops) -> None:
        self.nnfusion_argstr = self.nnfusion_argstr.replace("127.0.0.1:8880", self.antares_url).replace("CUDA", self.nnfusion_device)
        os.environ["PATH"] = os.path.abspath(self.nnfusion_bin) + ":" + os.environ["PATH"]
        sys.path.insert(1, os.path.abspath(self.nnfusion_python))
        self.nnfusion = __import__('nnfusion')
        if os.path.exists(self.nnfusion_workdir):
            shutil.rmtree(self.nnfusion_workdir)
        os.mkdir(self.nnfusion_workdir)
        # from nnfusion.executor import Executor
        # from nnfusion.session import generate_sample, codegen, modify_nnfusion_rt, build
        # from nnfusion.data_format import cast_pytorch_tensor, cast_hlsl_tensor, HLSLTensor
        if not os.path.exists(self.onnx_repo):
            repo = git.Repo.clone_from(self.onnx_remote, self.onnx_repo)
        for case in load_model_tests(data_dir=self.onnx_tests, kind=self.onnx_test_kind):
            flag = False 
            opname = ""
            for v in ops:
                if "test_"+v+"_" in case.name or "test_"+v == case.name:
                    flag = True
                    opname = v
            if flag:
                if global_flag_float_as_half :
                    case = self._test_float_to_type(case, TensorProto.FLOAT16)
                    case = case._replace(rtol=0.00977, atol=0.00977)
                elif global_flag_float_as_double :
                    case = self._test_float_to_type(case, TensorProto.DOUBLE)

                if global_flag_input_as_constant:
                    self.run_input_as_constant(case, opname)
                else:
                    self.run(case, opname)
    
    def _test_float_to_type(self, model_test, float_to_type):
        #float_to_type = TensorProto.FLOAT16
        new_model_dir = os.path.join(self.nnfusion_workdir, model_test.name)
        if not os.path.exists(new_model_dir):
            os.mkdir(new_model_dir)
        # modify graph
        model_pb_path = os.path.join(model_test.model_dir, "model.onnx")
        new_model_pb_path = os.path.join(new_model_dir, "model.onnx")
        model = onnx.load(model_pb_path)
        #print(model)

        def change_model_float(model_pb_path, new_model_pb_path, float_to_type):
            model = onnx.load(model_pb_path)
            for in_tensor in model.graph.input:
                if in_tensor.type.HasField("tensor_type"):
                    if in_tensor.type.tensor_type.elem_type == TensorProto.FLOAT:
                        in_tensor.type.tensor_type.elem_type = float_to_type

            for out_tensor in model.graph.output:
                if out_tensor.type.HasField("tensor_type"):
                    if out_tensor.type.tensor_type.elem_type == TensorProto.FLOAT:
                        out_tensor.type.tensor_type.elem_type == float_to_type

            #print(model)
            onnx.save(model, new_model_pb_path)

        def save_fp(input_file, output_file, float_to_type, tensor):
            with open(input_file, "rb") as f:
                protobuf_content = f.read()
                if tensor.type.HasField("tensor_type") and \
                    tensor.type.tensor_type.elem_type == TensorProto.FLOAT:
                    ts = onnx.TensorProto()
                    ts.ParseFromString(protobuf_content)
                    ndts = numpy_helper.to_array(ts)

                    if float_to_type == TensorProto.FLOAT16:
                        ndts = ndts.astype(np.float16)
                    elif float_to_type == TensorProto.DOUBLE:
                        ndts = ndts.astype(np.float64)

                    fp_data = numpy_helper.from_array(ndts, tensor.name)
                    protobuf_content = fp_data.SerializeToString()

                fo = open(output_file, "wb")
                fo.write(protobuf_content)
                fo.close()

        for test_data_dir in glob.glob(os.path.join(model_test.model_dir, "test_data_set*")):
            new_test_data_dir = os.path.join(new_model_dir, test_data_dir.split("/")[-1])
            if not os.path.exists(new_test_data_dir):
                os.mkdir(new_test_data_dir)
            inputs_num = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
            for i in range(inputs_num):
                input_file = os.path.join(test_data_dir, f"input_{i}.pb")
                output_file = os.path.join(new_test_data_dir, f"input_{i}.pb")
                save_fp(input_file, output_file, float_to_type, model.graph.input[i])
            ref_outputs_num = len(
                glob.glob(os.path.join(test_data_dir, "output_*.pb"))
            )
            for i in range(ref_outputs_num):
                output_file = os.path.join(test_data_dir, f"output_{i}.pb")
                output_output_file = os.path.join(new_test_data_dir, f"output_{i}.pb")
                save_fp(output_file, output_output_file, float_to_type, model.graph.output[i])

        change_model_float(model_pb_path, new_model_pb_path, float_to_type)
        return model_test._replace(model_dir=new_model_dir)

    
    def _build_model(self, model_test):
        import nnfusion
        from nnfusion.executor import Executor
        from nnfusion.session import generate_sample, codegen, modify_nnfusion_rt, build

        model_pb_path = ""
        nnfusion_workdir = ""
        if isinstance(model_test, str):
            if not model_test.endswith(".onnx"):
                nnfusion_workdir = os.path.join(self.nnfusion_workdir, model_test.split("/")[-1])
                model_pb_path = os.path.join(model_test, "model.onnx")
            else:
                nnfusion_workdir = os.path.join(self.nnfusion_workdir, model_test.split("/")[-1][:-5])
                model_pb_path = model_test
        else:
            model_dir = model_test.model_dir
            model_pb_path = os.path.join(model_dir, "model.onnx")
            nnfusion_workdir = os.path.join(self.nnfusion_workdir, model_test.name)
        if not os.path.exists(nnfusion_workdir):
            os.mkdir(nnfusion_workdir)
        codegen(model_pb_path, self.nnfusion_argstr, nnfusion_workdir)
        rt_dir = os.path.join(nnfusion_workdir, "nnfusion_rt/cuda_codegen")
        to_cst_dir = os.path.join(rt_dir, "Constant")
        cur_cst_dir = os.path.join(self.root_dir, "Constant")
        #if os.path.exists(cur_cst_dir):
        if os.path.islink(cur_cst_dir):
            os.unlink(cur_cst_dir)
        if os.path.exists(to_cst_dir):
            os.symlink(to_cst_dir, cur_cst_dir, True)
        modify_nnfusion_rt(rt_dir)
        build(rt_dir)
        return Executor(rt_dir)
    
    def _debug_tensor(rt) -> None:
        for i in rt.get_inputs():
            print(i.name)
        for i in rt.get_outputs():
            print(i.name)
    
    def _assert_similar_outputs(
        self,
        ref_outputs: Sequence[Any],
        outputs: Sequence[Any],
        rtol: float,
        atol: float,
    ) :
        try:
            np.testing.assert_equal(len(outputs), len(ref_outputs))
            for i in range(len(outputs)):
                if isinstance(outputs[i], (list, tuple)):
                    for j in range(len(outputs[i])):
                        f = self._assert_similar_outputs(
                            ref_outputs[i][j], outputs[i][j], rtol, atol
                        )
                        if f == False:
                            return False
                else:
                    np.testing.assert_equal(outputs[i].dtype, ref_outputs[i].dtype)
                    if ref_outputs[i].dtype == object:
                        np.testing.assert_array_equal(outputs[i], ref_outputs[i])
                    else:
                        np.testing.assert_allclose(
                            outputs[i], ref_outputs[i], rtol=rtol, atol=atol
                        )
            return True
        except:
            return False 

    def _load_proto(
        self,
        proto_filename: str,
        target_list: List[Union[np.ndarray, List[Any]]],
        model_type_proto: TypeProto,
    ) -> None:
        with open(proto_filename, "rb") as f:
            protobuf_content = f.read()
            if model_type_proto.HasField("sequence_type"):
                sequence = onnx.SequenceProto()
                sequence.ParseFromString(protobuf_content)
                target_list.append(numpy_helper.to_list(sequence))
            elif model_type_proto.HasField("tensor_type"):
                tensor = onnx.TensorProto()
                tensor.ParseFromString(protobuf_content)
                target_list.append(numpy_helper.to_array(tensor))
            elif model_type_proto.HasField("optional_type"):
                optional = onnx.OptionalProto()
                optional.ParseFromString(protobuf_content)
                target_list.append(numpy_helper.to_optional(optional))
            else:
                print(
                    "Loading proto of that specific type (Map/Sparse Tensor) is currently not supported"
                )
    
    def _load_original_proto(
        self,
        proto_filename: str,
        target_list: List[Union[np.ndarray, List[Any]]],
        model_type_proto: TypeProto,
    ) -> None:
        with open(proto_filename, "rb") as f:
            protobuf_content = f.read()
            if model_type_proto.HasField("sequence_type"):
                sequence = onnx.SequenceProto()
                sequence.ParseFromString(protobuf_content)
                target_list.append(sequence)
            elif model_type_proto.HasField("tensor_type"):
                tensor = onnx.TensorProto()
                tensor.ParseFromString(protobuf_content)
                target_list.append(tensor)
            elif model_type_proto.HasField("optional_type"):
                optional = onnx.OptionalProto()
                optional.ParseFromString(protobuf_content)
                target_list.append(optional)
            else:
                print(
                    "Loading proto of that specific type (Map/Sparse Tensor) is currently not supported"
                )
    
    def _alter_proto(self, model, inputs, target_pb_path):
        for i in range(0, len(model.graph.input)):
            inputs[i].name = model.graph.input[i].name
        model.graph.ClearField('input')
        for t in inputs:
            model.graph.initializer.append(t)
        onnx.save(model, target_pb_path)

    def run_input_as_constant(self, model_test, op_name) -> None:
        import nnfusion
        from nnfusion.data_format import cast_pytorch_tensor
        model_dir = model_test.model_dir
        model_pb_path = os.path.join(model_dir, "model.onnx")
        model = onnx.load(model_pb_path)

        for test_data_dir in glob.glob(os.path.join(model_dir, "test_data_set*")):
            inputs = []
            inputs_num = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
            for i in range(inputs_num):
                input_file = os.path.join(test_data_dir, f"input_{i}.pb")
                self._load_original_proto(input_file, inputs, model.graph.input[i].type)
            ref_outputs = []
            ref_outputs_num = len(
                glob.glob(os.path.join(test_data_dir, "output_*.pb"))
            )
            for i in range(ref_outputs_num):
                output_file = os.path.join(test_data_dir, f"output_{i}.pb")
                self._load_proto(
                    output_file, ref_outputs, model.graph.output[i].type
                )
            
            try:
                new_pb_path = os.path.join(self.nnfusion_workdir, model_test.name + test_data_dir.split("/")[-1] + ".onnx")
                self._alter_proto(model, inputs, new_pb_path)
                rt = self._build_model(new_pb_path)
            except:
                print("@,", op_name, ",", model_test.name, ", BUILD ERROR", ", FAILED")
                return

            try:
                nnf_inputs = dict()
                nnf_outputs = dict()
                nnf_torch_outputs = list()
                for output_i in range(len(ref_outputs)):
                    name = model.graph.output[output_i].name
                    nnf_torch_outputs.append(torch.tensor(ref_outputs[output_i]).cuda())
                    nnf_torch_outputs[-1].zero_()
                    nnf_outputs[name] = cast_pytorch_tensor(nnf_torch_outputs[-1])
                rt.feed_data(nnf_inputs, nnf_outputs)
                outputs = [nnf_outputs[output_i.name].to_pytorch_tensor().cpu().numpy() for output_i in model.graph.output]#list(prepared_model.run(inputs))
            except:
                print("@,", op_name , ",", model_test.name, ", EXECUTION ERROR", ", FAILED")
                continue
            r = self._assert_similar_outputs(
                ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
            )
            print("@,", op_name, ",", model_test.name, "," + test_data_dir, ", PASS" if r else ", FAILED")
    
    def run(self, model_test, op_name) -> None:
        import nnfusion
        from nnfusion.data_format import cast_pytorch_tensor
        model_dir = model_test.model_dir
        model_pb_path = os.path.join(model_dir, "model.onnx")
        model = onnx.load(model_pb_path)
        try:
            rt = self._build_model(model_test)
        except:
            print("@,", op_name, ",", model_test.name, ", BUILD ERROR", ", FAILED")
            return
        # debug_tensor(rt)
        for test_data_npz in glob.glob(os.path.join(model_dir, "test_data_*.npz")):
            test_data = np.load(test_data_npz, encoding="bytes")
            inputs = list(test_data["inputs"])
            outputs = [] #list(prepared_model.run(inputs))
            ref_outputs = test_data["outputs"]
            self._assert_similar_outputs(
                ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
            )
        for test_data_dir in glob.glob(os.path.join(model_dir, "test_data_set*")):
            inputs = []
            inputs_num = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
            for i in range(inputs_num):
                input_file = os.path.join(test_data_dir, f"input_{i}.pb")
                self._load_proto(input_file, inputs, model.graph.input[i].type)
            ref_outputs = []
            ref_outputs_num = len(
                glob.glob(os.path.join(test_data_dir, "output_*.pb"))
            )
            for i in range(ref_outputs_num):
                output_file = os.path.join(test_data_dir, f"output_{i}.pb")
                self._load_proto(
                    output_file, ref_outputs, model.graph.output[i].type
                )

            try:
                nnf_inputs = dict()
                nnf_torch_inputs = list()
                for input_i in range(len(inputs)):
                    name = model.graph.input[input_i].name
                    nnf_torch_inputs.append(torch.tensor(inputs[input_i]).cuda())
                    nnf_inputs[name] = cast_pytorch_tensor(nnf_torch_inputs[-1])
                nnf_outputs = dict()
                nnf_torch_outputs = list()
                for output_i in range(len(ref_outputs)):
                    name = model.graph.output[output_i].name
                    nnf_torch_outputs.append(torch.tensor(ref_outputs[output_i]).cuda())
                    nnf_torch_outputs[-1].zero_()
                    nnf_outputs[name] = cast_pytorch_tensor(nnf_torch_outputs[-1])
                rt.feed_data(nnf_inputs, nnf_outputs)
                outputs = [nnf_outputs[output_i.name].to_pytorch_tensor().cpu().numpy() for output_i in model.graph.output]#list(prepared_model.run(inputs))
            except:
                print("@,", op_name , ",", model_test.name, ", EXECUTION ERROR", ", FAILED")
                continue
            r = self._assert_similar_outputs(
                ref_outputs, outputs, rtol=model_test.rtol, atol=model_test.atol
            )
            print("@,", op_name, ",", model_test.name, "," + test_data_dir, ", PASS" if r else ", FAILED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( prog = 'Test with ONNX Test cases')
    parser.add_argument('-n', '--name', default="abs,acos") 
    parser.add_argument('-f', '--file', default="default_operators.txt") 
    parser.add_argument('-m', '--mode', default="name") 
    parser.add_argument("-i", "--input_as_constant", default=False)
    parser.add_argument("-a", "--float_as_half", default=False)
    parser.add_argument("-d", "--float_as_double", default=False)
    args = parser.parse_args()
    global_flag_input_as_constant = args.input_as_constant
    global_flag_float_as_half = args.float_as_half
    global_flag_float_as_double = args.float_as_double
    if args.mode == "name":
        TestContext(args.name.split(","))
    if args.mode == "file":
        f = open(args.file).readlines()
        TestContext([v.strip().lower() for v in f])
