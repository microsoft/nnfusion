import onnx

model = onnx.load("/home/syn/models/tf_pai_asr_transformer/model.onnx")
inputs = model.graph.input
inputs[0].type.tensor_type.shape.dim[0].dim_value = 1
inputs[0].type.tensor_type.shape.dim[1].dim_value = 200
inputs[1].type.tensor_type.shape.dim[0].dim_value = 1
print(inputs)
onnx.save(model, "/home/syn/models/tf_pai_asr_transformer/model_input.onnx")
