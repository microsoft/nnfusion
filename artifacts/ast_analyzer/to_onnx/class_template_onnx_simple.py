import torch
import onnxruntime as ort
import onnx
import numpy as np

def load_model(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed!")
    # print(onnx.helper.printable_graph(onnx_model.graph))
    # print(onnx_model.graph.value_info)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    session = ort.InferenceSession(model_path)
    inputs_name = [item.name for item in session.get_inputs()]
    outputs_name = [item.name for item in session.get_outputs()]
    return session, inputs_name, outputs_name

forward_session, forward_inputs, forward_outputs = load_model("tmp/^^MODELNAME-forward.onnx")
ort.set_default_logger_severity(2)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
def run(^^INPUTS):
    print("[run onnx simple]")
    ort_inputs = {
        @.@INPUTS@@@Tensor@forward_inputs[%%i]: ^^NAME.cpu().detach().numpy(),@@General@forward_inputs[%%i]: np.array(^^NAME),@@@ # testing
    }
    outputs = forward_session.run(forward_outputs, ort_inputs)
    @.@OUTPUTS^^NAME = torch.from_numpy(outputs[%%i]).cuda()
    return ^^O_OUTPUTS
