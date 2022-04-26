# [{"tvm_func_name": "manual_dot_nn_op_float_m1_k256_n256_kernel0", 
#     "op_type": "Dot", 
#     "parameters": {"arg0_shape": [1, 256], 
#         "arg1_shape": [256, 256], 
#         "out_shape": [1, 256], 
#         "transpose_A": false, 
#         "transpose_B": false}, 
#         "code": "extern \"C\" __global__ void manual_dot_nn_op_float_m1_k256_n256_kernel0(float* input0, float* input1, float* output0)\n{\n    int warp_id = threadIdx.x >> 5;\n    int lane_id = threadIdx.x & 31;\n    int col_id = blockIdx.x * blockDim.x / 4 + lane_id;\n    if (col_id < 256)\n    {\n        float val = 0;\n        int k_start = warp_id * 64;\n        int k_end = (warp_id + 1) * 64;\n        for (int i = k_start; i < k_end; i++)\n        {\n            val = fma(input0[i], input1[i * 256 + col_id], val);\n        }\n        if (warp_id == 0)\n        {\n            output0[col_id]=0;\n        }\n        __syncthreads();\n        atomicAdd(output0 + col_id, val);\n    }\n\n}\n", 
#         "gridDim": [8, 1, 1], 
#         "blockDim": [128, 1, 1]}]

import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--op_type', required=True, type=str, default='')
parser.add_argument('--source_file', required=True, type=str, default='')
parser.add_argument('--json_file', required=True, type=str, default='example.json')
parser.add_argument("--input0_shape", required=True, nargs="*", type=int,default=[1, 2, 3])
parser.add_argument("--input1_shape", nargs="*", type=int,default=[1, 2, 3])
parser.add_argument("--output0_shape", required=True, nargs="*", type=int,default=[1, 2, 3])
parser.add_argument("--transpose_A", type=bool, default=False)
parser.add_argument("--transpose_B", type=bool, default=False)
parser.add_argument("--stride", nargs="*", type=int,default=[1, 1])
parser.add_argument("--padding", nargs="*", type=int,default=[0, 0])
parser.add_argument("--dilation", nargs="*", type=int,default=[1, 1])
parser.add_argument("--window_shape", nargs="*", type=int,default=[1, 1])
parser.add_argument("--reduction_axis", nargs="*", type=int,default=[0])
parser.add_argument("--broadcast_axis", nargs="*", type=int,default=[0])

args = parser.parse_args()

info = {}
info["parameters"] = {}
op_type = args.op_type
info["op_type"] = op_type
source_file = args.source_file
json_file = args.json_file
tvm_func_name = source_file.split("/")[-1][:-3]
tvm_func_name = tvm_func_name.replace("[", "_").replace("]", "_").replace(",", "_")
info["tvm_func_name"] = tvm_func_name
if op_type == "Dot" or op_type == "BatchMatMul":
    info["parameters"]["arg0_shape"] = args.input0_shape
    info["parameters"]["arg1_shape"] = args.input1_shape
    info["parameters"]["out_shape"] = args.output0_shape
    info["parameters"]["transpose_A"] = args.transpose_A
    info["parameters"]["transpose_B"] = args.transpose_B
elif op_type == "Convolution" or op_type == "DepthwiseConv2dNative" or op_type == "Fused_Convolution_Add" or op_type == "Fused_Convolution_Relu" or op_type == "Fused_Convolution_Add_Relu":
    info["parameters"]["input_shape"] = args.input0_shape
    info["parameters"]["filter_shape"] = args.input1_shape
    info["parameters"]["output_shape"] = args.output0_shape
    info["parameters"]["window_movement_strides"] = args.stride
    info["parameters"]["padding_below_diff"] = args.padding
    info["parameters"]["window_dilation_strides"] = args.dilation
elif op_type == "MaxPool" or op_type == "AvgPool":
    info["parameters"]["input_shape"] = args.input0_shape
    info["parameters"]["output_shape"] = args.output0_shape
    info["parameters"]["window_shape"] = args.window_shape
    info["parameters"]["window_stride"] = args.stride
    info["parameters"]["padding_below"] = args.padding
elif op_type == "Sum":
    info["parameters"]["input_shape"] = args.input0_shape
    info["parameters"]["output_shape"] = args.output0_shape
    info["parameters"]["reduction_axis"] = args.reduction_axis
elif op_type == "Broadcast":
    info["parameters"]["input_shape"] = args.input0_shape
    info["parameters"]["output_shape"] = args.output0_shape
    info["parameters"]["broadcast_axis"] = args.broadcast_axis
else:
    info["parameters"]["input_shape"] = args.input0_shape
    info["parameters"]["output_shape"] = args.output0_shape


code = ""
gridDim = []
blockDim = []
with open(source_file, 'r', encoding='utf-8') as f:
    flag = False  
    for line in f.readlines():
        if flag:
            code += line
        if flag and line.startswith("}"):
            flag = False
        if line.startswith("extern \"C\" "):
            match = re.search("__launch_bounds__\([0-9]*\) ", line)
            if match:
                lb = match.group()
                line = line.replace(lb, "")
            try:
                kernel_name = re.search("void .*_kernel0", line).group()
                line = line.replace(kernel_name, "void " + tvm_func_name)
                code += line
                flag = True
            except:
                pass
            #print(line)
            #kernel_name = re.search("void .*_kernel0", line).group()
            ## print(kernel_name)
            #line = line.replace(kernel_name, "void " + tvm_func_name)
            #code += line
            #flag = True
        if "dim3 grid(" in line:
            line = line.split("(")[1].split(")")[0].split(",")
            for i in line:
                gridDim.append(int(i))
        if "dim3 block(" in line:
            line = line.split("(")[1].split(")")[0].split(",")
            for i in line:
                blockDim.append(int(i))
    
info["code"] = code
info["gridDim"] = gridDim
info["blockDim"] = blockDim    

# json_file = source_file + ".json"
with open(json_file, 'w', encoding='utf-8') as fw:
    json.dump(info, fw)   





