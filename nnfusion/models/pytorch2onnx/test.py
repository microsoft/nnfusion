import torch
import onnxruntime as ort
print(ort.get_available_providers())
# print(ort.__version__)
providers = [("CUDAExecutionProvider")]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("/workspace/v-leiwang3/lowbit_model/bert-large-uncased/bert-large.float16.onnx", sess_options=sess_options, providers=providers)
