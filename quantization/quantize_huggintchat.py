import time
import torch
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM


enable_quantize = True
export_nnfusion = False
use_triton = True

assert not (export_nnfusion and use_triton)

pretrained_model_dir = "/workspace/v-leiwang3/lowbit_model/oa_llama_30b/oasst-rlhf-2-llama-30b-7k-steps-xor"
quantized_model_dir = f"quantization/models/huggingchat-30b-rlhf-2-4bit{'-triton' if use_triton else ''}"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

if enable_quantize:
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize bits
        # desc_act=False,  # disable activation description
        # group_size=128,  # disable group quantization
        desc_act=True
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(examples, use_tvm=False if use_triton else True,
                   use_triton=use_triton, export_nnfusion=export_nnfusion, autotune_warmup_after_quantized=True)

    # save quantized model
    model.save_quantized(quantized_model_dir)
# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_tvm=False if use_triton else True,
                                           use_triton=use_triton, export_nnfusion=export_nnfusion).half().cuda()
# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])

# export 2 onnx
batch_size = 128
seq_length = 1
input_shape = (batch_size, seq_length)
onnx_name = f"qmodel_b{batch_size}s{seq_length}.onnx"
output_path = os.path.join(quantized_model_dir, onnx_name)
input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
attention_mask = torch.ones(input_shape, dtype=torch.long, device="cuda:0")

if not export_nnfusion:
    start = time.time()
    for i in range(100):
        outputs = model(input_ids=input_ids)
    end = time.time()
    print("time: ", (end - start) / 100 * 1000, "ms")
    print(outputs.logits)
else:
    import onnx
    from onnxsim import simplify
    model = model.half().cuda()
    torch.onnx.export(
        model,
        input_ids,
        f=output_path,
        opset_version=11,
    )
    # load your predefined ONNX model
    model = onnx.load(output_path)
    # convert model
    model_simp, check = simplify(model)
    sim_output_path = os.path.join(
        quantized_model_dir, f"qmodel_b{batch_size}s{seq_length}_sim.onnx")
    onnx.save(model_simp, sim_output_path,
              save_as_external_data=True, all_tensors_to_one_file=False)
