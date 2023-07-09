# -- coding: utf-8 --**
import json
import re

model = json.load(open("./model.json", "r"))
tuned = json.load(open("./tuned.json", "r"))
bits = 4
cache_path = "./qtuned.json"
with open(cache_path, 'r') as f:
    kernel = json.load(f)


def find_largest_mediate(string):
    mediates = re.findall(r'mediate\d+', string)
    # print(mediates)
    if len(mediates) == 0:
        return 0
    max_mediate = max(mediates, key=lambda x: int(x[7:]))
    largest_mediate = int(max_mediate[7:])
    return largest_mediate


def customize_ir(ir: str, output_shape: list):
    # if the dims of output_shape is 3, compact the first two axis
    # if len(output_shape) == 3:
    #     output_shape = [output_shape[0] * output_shape[1], output_shape[2]]
    '''input ir is a string, return a customized string
        
        example input:
            " - einstein_v2(" mediate0[N0, N1, N2] = input0[N2] where N0 in 32, N1 in 512;  mediate1[N0, N1, N2] = input1[N0, N1, N2] + mediate0[N0, N1, N2]; output0[N0, N1, N2] = mediate1[N0, N1, N2].call(`max`, [const(0).cast(mediate1[N0, N1, N2].dtype())]);", input_dict={ "input0" : { "dtype" : "float16", "shape" : [4096]} ,  "input1" : { "dtype" : "float16", "shape" : [32, 512, 4096]} }) ## @: "
        example output:
            " - einstein_v2(\"mediate0[N0, N1, N2] = input0[N2] where N0 in 32, N1 in 512;  mediate1[N0, N1, N2] = input1[N0, N1, N2] + mediate0[N0, N1, N2]; mediate2[N0, N1, N2] = mediate1[N0, N1, N2].call(`max`, [const(0).cast(mediate1[N0, N1, N2].dtype())]);mediate3[N0, N1] = mediate2[N0 // 512, N0 % 512, N1] where N0 in 16384, N1 in 4096; output0[N0, N1, N2, N3] = mediate3[(N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  % 8 * 2 + (N1 * 16 + N3) % 16 // 8, (N1 * 16 + N3)// 16 * 16 + (N0 * 16 + N2)  % 16 // 8 * 8 + (N1 * 16 + N3) % 8] where N0 in 1024, N1 in 256, N2 in 16, N3 in 16;\", input_dict={ \"input0\" : { \"dtype\" : \"float16\", \"shape\" : [4096]} ,  \"input1\" : { \"dtype\" : \"float16\", \"shape\" : [32, 512, 4096]}}) ## @:  ",
            
    "Broadcast_Add_Relu" 
        steps:
            1. find the maximum index of mediate, if the maximum index is 1, then the new mediate index is 2, and so on
            2. replace the output0 with new mediate index
            3. append layout transoformation to the end of the ir, and return the new ir
            
    '''
    # Step 1: Find the maximum index of mediate and increment it by 1
    mediate_index = find_largest_mediate(ir)
    # Step 2: Replace the output0 with the new mediate index
    new_mediate_name_1 = 'mediate%d' % (mediate_index + 1)
    new_mediate_name_2 = 'mediate%d' % (mediate_index + 2)

    ir = ir.replace('output0', new_mediate_name_1)

    if len(output_shape) == 2:
        inserted_ir = 'output0[N0, N1, N2, N3] = %s[(N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  %% 8 * 2 + (N1 * 16 + N3) %% 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  %% 16 // 8 * 8 + (N1 * 16 + N3) %% 8] where N0 in %d, N1 in %d, N2 in 16, N3 in 16;' % (
            new_mediate_name_1, output_shape[0] // 16, output_shape[1] // 16)
        # print(inserted_ir)
    elif len(output_shape) == 3:
        inserted_ir = '%s[N0, N1] = %s[N0 // %d, N0 %% %d, N1] where N0 in %d, N1 in %d;output0[N0, N1, N2, N3] = ' % (
            new_mediate_name_2, new_mediate_name_1, output_shape[1], output_shape[1], output_shape[0] * output_shape[1], output_shape[2])
        inserted_ir += '%s[(N0 * 16 + N2) // 16 * 16 + (N0 * 16 + N2)  %% 8 * 2 + (N1 * 16 + N3) %% 16 // 8, (N1 * 16 + N3) // 16 * 16 + (N0 * 16 + N2)  %% 16 // 8 * 8 + (N1 * 16 + N3) %% 8] where N0 in %d, N1 in %d, N2 in 16, N3 in 16;' % (
            new_mediate_name_2, output_shape[0] * output_shape[1] // 16, output_shape[2] // 16)
        # print(inserted_ir)
    # insert the new ir to the front of the end of the \", input_dict
    new_ir = ir[:ir.rfind('\", input_dict')] + \
        inserted_ir + ir[ir.rfind('\", input_dict'):]
    # print(new_ir)
    return new_ir


def find_usage_of_node(model, node_id):
    '''
    find the usage of the node
    '''
    usage = []
    for i, node in enumerate(model):
        if node_id in sum(node[3], []):
            # print(node_id, node[3], node_id in sum(node[3],[]))
            usage.append(i)
    return usage


def find_node(mode, node_id):
    for i, node in enumerate(model):
        if node_id == node[0]:
            return node
    return None


def insert_code(tuned, node_id, code, func_name, block_size, grid_size, latency):
    for i, node in enumerate(tuned):
        if node_id in node["nodes"]:
            print("insert code to node", node_id)
            gourd_name = "Group" + str(node["group_id"])
            node["input_desc"] = [[node_id, 0], [
                node_id, 1], [node_id, 2], [node_id, 3]]
            node["output_desc"] = [[node_id, 0]]
            code = code.replace(func_name, gourd_name)
            code = code.replace('extern "C"', "")
            code = code.replace("#include <cuda_fp16.h>", "")
            node["code"] = code.replace(func_name, gourd_name)
            node["block_size"] = block_size
            node["grid_size"] = grid_size
            node["name"] = gourd_name
            node["latency"] = latency
            node["gain"] = 3.141592657
            # node["dsmem"] = dsmem
            return


for i, node in enumerate(model):
    # node info is a list [407, ' - einstein_v2(" output0[N0, N1, N2] = input0[input1[N0, N1], N2]; ", input_dict={ "input0" : { "dtype" : "float16", "shape" : [2, 1024]} ,  "input1" : { "dtype" : "int64", "shape" : [32, 512]} })  ', 'GatherV2', [[2, 0], [403, 0]]]
    # parse name
    node_name = node[2]
    if 'QuantLinear' in node_name:
        assert i > 0
        print("find quant_linear node", node_name)
        node_id = node[0]
        node_ir = node[1]
        input_tensors = node[3]
        print(input_tensors)
        input_node = find_node(model, input_tensors[0][0])
        weight_node = find_node(model, input_tensors[1][0])
        print("input_node", input_node[2])
        print("weight_node", weight_node[2] if weight_node else "None")

        assert weight_node is None, "current version only support the case that the weight is a constant"
        input_node_id = input_node[0]
        input_node_name = input_node[2]
        input_node_ir = input_node[1]
        # print(node_ir.split('shape')[1])
        node_input_shape = [int(s) for s in node_ir.split(
            'shape')[1].split('[')[-1].split(']')[0].split(',')]
        node_weight_shape = [int(s) for s in node_ir.split(
            'shape')[2].split('[')[-1].split(']')[0].split(',')]
        # detect the trans_a or trans_b
        trans_a = False
        trans_b = True
        print("input shape is ", node_input_shape)
        print("weight shape is ", node_weight_shape)
        # note: we assume that currently only b can be transposed
        if node_input_shape[-1] == node_weight_shape[-1]:
            trans_b = True
        # print(trans_b)

        # print(input_node_name)
        # print(input_node_ir)
        M = node_input_shape[0] if len(
            node_input_shape) == 2 else node_input_shape[0] * node_input_shape[1]
        K = node_input_shape[-1]
        N = node_weight_shape[-2] if trans_b else node_weight_shape[-1]
        print("node id is %d, M is %d, K is %d, N is %d" % (node[0], M, K, N))
        key = f"b{bits}n{N}k{K}"
        mx = f'm{M}'
        kernel_code = kernel[key][mx]['code']
        func_name = kernel[key][mx]['func_name']
        params = kernel[key][mx]['params']
        block_size = [1, 1, 1]
        grid_size = [1, 1, 1]
        if M == 1:
            block_size = [32, params['num_warps'], 1]
            grid_size = [N // params['num_warps'], 1, 1]
        else:
            block_size = [32, params['block_row_warps'],
                          params['block_col_warps']]
            grid_size = [N // params['BN'], M // params['BM'], 1]
            kernel_code = kernel_code.replace(
                '#include <cuda_fp16.h>\n#include <mma.h>\n\n                static inline __device__ __host__ unsigned\n                __pack_half2(const half x, const half y) {\n                unsigned v0 = *((unsigned short *)&x);\n                unsigned v1 = *((unsigned short *)&y);\n                return (v1 << 16) | v0;\n            }', "")
        insert_code(tuned, node_id, kernel_code, func_name,
                    block_size, grid_size, 3.1415926)

json.dump(tuned, open('./tuned_new.json', 'w'), indent=4)
