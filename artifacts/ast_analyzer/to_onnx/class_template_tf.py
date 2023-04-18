import torch
import onnx
from onnx_tf.backend import prepare

def load_model(model_path):
    model = onnx.load(model_path)
    op = onnx.OperatorSetIdProto()
    # Sigmoid version 13 is not implemented.
    op.version = 12
    update_model = onnx.helper.make_model(model.graph, opset_imports=[op])
    tf_model = prepare(update_model)
    return tf_model

class GenModel(torch.autograd.Function):

    tf_model = load_model("tmp/^^MODELNAME-forward.onnx")
    
    @staticmethod
    def forward(ctx, ^^INPUTS):
        if torch.is_grad_enabled():
            raise NotImplementedError
        else:
            print("use tf forward")
            @.@INPUTS^^NAME = ^^NAME.cpu().detach().numpy()
            ^^OUTPUTS = GenModel.tf_model.run((^^INPUTS))
            @.@OUTPUTS^^NAME = torch.tensor(^^NAME)
            return ^^OUTPUTS
    
    @staticmethod
    def backward(ctx, ^^OUTPUTS):
        _ = ctx.saved_tensors
        raise NotImplementedError