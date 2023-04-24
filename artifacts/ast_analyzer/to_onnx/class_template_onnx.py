import torch
import onnxruntime as ort
import onnx
import numpy as np
import copy

def get_numpy_233(tensor):
    onnx_dtype = tensor.type
    shape = []
    for s in onnx_dtype.tensor_type.shape.dim:
        shape.append(s.dim_value)
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype.tensor_type.elem_type]
    return np.ones(shape, dtype=dtype) * 233

def load_model(model_path):
    onnx_model = onnx.load(model_path)
    fake_out = None
    print(onnx.helper.printable_graph(onnx_model.graph))
    # print(onnx_model.graph.value_info)
    try:
        onnx.checker.check_model(onnx_model)
        print(f"{model_path}: check passed!")
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        session = ort.InferenceSession(model_path)
    except onnx.onnx_cpp2py_export.checker.ValidationError as err:
        err_msg = str(err)
        assert(err_msg.startswith("No Op registered for"))
        print("[warning]", err_msg.split("\n")[0] + ", return tensors with value 233")
        session = None
        fake_out = [get_numpy_233(o) for o in onnx_model.graph.output]

    inputs_name = [item.name for item in onnx_model.graph.input]
    outputs_name = [item.name for item in onnx_model.graph.output]
    return session, inputs_name, outputs_name, fake_out

class GenModel(torch.autograd.Function):
    ort.set_default_logger_severity(2)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    @IF_EVALforward_session, forward_inputs, forward_outputs, forward_fake_out = load_model("^^TMP_DIR/^^MODELNAME/forward.onnx")
    @IF_TRAINforward_session, forward_inputs, forward_outputs, forward_fake_out = load_model("^^TMP_DIR/^^MODELNAME/train-fwd.onnx")
    @IF_TRAINbackward_session, backward_inputs, backward_outputs, backward_fake_out = load_model("^^TMP_DIR/^^MODELNAME/train-bwd.onnx")
    
    @staticmethod
    def forward(ctx, ^^INPUTS):
        print("use onnx forward")
        for i, x in enumerate([^^INPUTS]):
            if isinstance(x, int):
                with open(f"^^TMP_DIR/^^MODELNAME/bin/input_ref_{i}.bin", "wb") as f:
                    np.full((1,), x, dtype=np.int64).tofile(f)
            else:
                with open(f"^^TMP_DIR/^^MODELNAME/bin/input_ref_{i}.bin", "wb") as f:
                    x.cpu().detach().numpy().tofile(f)
        ort_inputs = {
            @.@INPUTSGenModel.forward_inputs[%%i]: ^^NAME.cpu().detach().numpy(),
        }
        if GenModel.forward_session is None:
            outputs = copy.deepcopy(GenModel.forward_fake_out)
        else:
            outputs = GenModel.forward_session.run(GenModel.forward_outputs, ort_inputs)
        for i, x in enumerate(outputs):
            if isinstance(x, int):
                print("output", i, x)
            else:
                with open(f"^^TMP_DIR/^^MODELNAME/bin/output_ref_{i}.bin", "wb") as f:
                    x.tofile(f)
        @.@OUTPUTS^^NAME = torch.from_numpy(outputs[%%i])
        @.@PARAMS^^O_NAME = ^^I_NAME
        ctx.save_for_backward(^^CTX_SAVE)
        return ^^O_OUTPUTS
    
    @staticmethod
    def backward(ctx, ^^RETURNS):
        print("use onnx backward")
        ^^CTX_OR_SAVE = ctx.saved_tensors
        ort_inputs = {
            @.@RETURNSGenModel.backward_inputs[%%i]: ^^NAME.cpu().detach().numpy(),
            @.@CTX_SAVESGenModel.backward_inputs[%%i]: ^^NAME.cpu().detach().numpy(),
        }
        outputs = GenModel.backward_session.run(GenModel.backward_outputs, ort_inputs)
        outputs = tuple(torch.from_numpy(x).cuda() for x in outputs)
        return outputs
