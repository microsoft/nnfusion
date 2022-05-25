import torch
from arch import V100, K80
import op
from policy import ConstructionPolicyV2
import tvm
from memopt import CodeGenerator
import memopt
import ctypes
import memopt.graph as graph

def translate_to_tvm(expr, input_dict):
    from lang.generic import einstein_v2, OUTPUT_TEMP, INPUT_TEMP
    OUTPUT_TEMP.clear()
    INPUT_TEMP.clear()
    einstein_v2(expr, input_dict)
    return INPUT_TEMP + OUTPUT_TEMP

def test(args, codegen_dict):
    sch = tvm.te.create_schedule(args[-1].op)
    print(args[-1])

    saxis_names = [axis.var.name for axis in sch[args[-1]].op.axis]
    for item in sch.stage_map.items():
        if isinstance(item[0], tvm.te.ComputeOp) and len(item[0].reduce_axis) > 0:
            pass

    cgen = CodeGenerator()
    sch = cgen.recursive_schedule_up(sch, codegen_dict, tile_blacklist=[])

    # from memopt.debug import debug
    # debug({**globals(), **locals()})

    with memopt.Scope(sch) as scope:
        kernel_code = memopt.build_op(sch, args, "cuda", [], [], name="MyMatMul", global_kernel=True)
        code = memopt.utils.append_host_call(kernel_code, scope.block_size, scope.grid_size, len(args), "MyMatMul", True)
        # print(code)
        lib = memopt.utils.compile_and_load(code)
        lib.function.restype = ctypes.c_float
        return memopt.utils.profile(lib, args)


# expr1 = """
# mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 2160, N2 in 3840;
# mediate1[N, HO, WO, F] +=! input1[N, -0 + KH + HO * 1, -0 + KW + WO * 1, C] * input2[KH, KW, C, F] where HO in 2160, WO in 3840;
# mediate2[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3];
# output0[N0, N1, N2, N3] = mediate2[N0, N1, N2, N3].call(`max`, [const(0).cast(mediate2[N0, N1, N2, N3].dtype())]);
# """
# input_dict1={ "input0" : { "dtype" : "float32", "shape" : [64]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 2160, 3840, 21]} ,  "input2" : { "dtype" : "float32", "shape" : [1, 1, 21, 64]} }
# codegen_dict1 = {"k": [21, 1], 'N0': [1, 1], 'N1': [2, 4], 'N2': [2, 4], 'N3' : [16, 4]}
# conv1 = op.ConvOp(1, 64, 21, 1, 1, 2160, 3840, 1, "SAME")
# args1 = translate_to_tvm(expr1, input_dict1)
# # 8.123019727071126
# # test(translate_to_tvm(expr1, input_dict1), codegen_dict1)

# expr2 = """
# mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 2160, N2 in 3840;
# mediate1[N, HO, WO, F] +=! input1[N, -0 + KH + HO * 1, -0 + KW + WO * 1, C] * input2[KH, KW, C, F] where HO in 2160, WO in 3840;
# mediate2[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3];
# output0[N0, N1, N2, N3] = const(1).cast(mediate2[N0, N1, N2, N3].dtype()) / (const(1).cast(mediate2[N0, N1, N2, N3].dtype()) + (-mediate2[N0, N1, N2, N3]).call(`exp`));
# """
# input_dict2={ "input0" : { "dtype" : "float32", "shape" : [1]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 2160, 3840, 64]} ,  "input2" : { "dtype" : "float32", "shape" : [1, 1, 64, 1]} }
# codegen_dict2 = {"k": [32, 1], 'N0': [1, 1], 'N1': [1, 1], 'N2': [128, 1], 'N3' : [1, 1]}
# conv2 = op.ConvOp(1, 1, 64, 1, 1, 2160, 3840, 1, "SAME")
# args2 = translate_to_tvm(expr2, input_dict2)
# 5.6312149365743
# fuse_config = [{"k": [21, 1], 'N0': [1, 1], 'N1': [1, 1], 'N2': [8, 16], 'N3' : [16, 4]},
#                {"k": [32, 1], 'N0': [1, 1], 'N1': [1, 1], 'N2': [128, 1], 'N3' : [1, 1]}]
# test(translate_to_tvm(expr2, input_dict2), codegen_dict2)

# 0.8250495990117391
args1 = translate_to_tvm(" mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 1080, N2 in 1920;   mediate1[N, HO, WO, C] +=! input1[N, -1 + KH + HO * 1, -1 + KW + WO * 1, C].when([-1 + KH + HO * 1 >= 0, -1 + KH + HO * 1 < 1080, -1 + KW + WO * 1 >= 0, -1 + KW + WO * 1 < 1920], const(0.0).cast(input1[N, -1 + KH + HO * 1, -1 + KW + WO * 1, C].dtype())) * input2[KH, KW, C, 0] where HO in 1080, WO in 1920, KH in 3, KW in 3;  output0[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3]; ", input_dict={ "input0" : { "dtype" : "float32", "shape" : [16]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 1080, 1920, 16]} ,  "input2" : { "dtype" : "float32", "shape" : [3, 3, 16, 1]} })
conv1 = op.DepthwiseConvOp(1, 16, 3, 1, 1080, 1920, 1, 1, "SAME")
# test(args1, codegen_dict1)

# 0.7106101314226786
args2 = translate_to_tvm(" mediate0[N0, N1, N2, N3] = input0[N3] where N0 in 1, N1 in 1080, N2 in 1920;   mediate1[N, HO, WO, F] +=! input1[N, -0 + KH + HO * 1, -0 + KW + WO * 1, C] * input2[KH, KW, C, F] where HO in 1080, WO in 1920;  mediate2[N0, N1, N2, N3] = mediate1[N0, N1, N2, N3] + mediate0[N0, N1, N2, N3]; output0[N0, N1, N2, N3] = mediate2[N0, N1, N2, N3].call(`max`, [const(0).cast(mediate2[N0, N1, N2, N3].dtype())]);", input_dict={ "input0" : { "dtype" : "float32", "shape" : [16]} ,  "input1" : { "dtype" : "float32", "shape" : [1, 1080, 1920, 16]} ,  "input2" : { "dtype" : "float32", "shape" : [1, 1, 16, 16]} })
conv2 = op.ConvOp(1, 16, 16, 1, 1, 1080, 1920, 1, "SAME")
fuse_config = [{"k": [9, 1], 'N0': [1, 1], 'N1': [4, 3], 'N2': [2, 8], 'N3' : [16, 1]},
               {"k": [16, 1], 'N0': [1, 1], 'N1': [4, 3], 'N2': [4, 4], 'N3' : [8, 2]}]
# fuse 4 + 3 1.3846784035364788

arch = V100()
configs1 = ConstructionPolicyV2(conv1, arch, ['N0', 'N3', 'N1', 'N2'], ['k']).emit_config_without_trails(10)[:10]
configs2 = ConstructionPolicyV2(conv2, arch, ['N0', 'N3', 'N1', 'N2'], ['k']).emit_config_without_trails(10)[:10]

# print(test(args1, {'k': [9, 1], 'N0': [1, 1], 'N3': [8, 2], 'N1': [3, 4], 'N2': [8, 4]}))

# results = []
# for config in configs1:
#     codegen_dict = config.to_codegen_dict()
#     value = test(args1, codegen_dict)
#     results.append([codegen_dict, value])
# results = sorted(results, key=lambda v : v[1])
# for result in results:
#     print(result)

def get_block_size(config):
    ret = 1
    codegen_dict = config.to_codegen_dict()
    for axis in config.spatial_axis:
        ret *= codegen_dict[axis][0]
    return ret

def get_tile_dim(config):
    ret = []
    codegen_dict = config.to_codegen_dict()
    for axis in config.spatial_axis:
        ret.append(codegen_dict[axis][0] * codegen_dict[axis][1])
    return ret

fused_configs = [fuse_config]

for c1 in configs1:
    for c2 in configs2:
        if get_block_size(c1) != get_block_size(c2):
            continue
        if conv1.get_grid_size(get_tile_dim(c1)) != conv2.get_grid_size(get_tile_dim(c2)):
            continue
        in_tile = get_tile_dim(c2)
        in_tile[1] = 16
        out_tile = get_tile_dim(c1)
        if in_tile == out_tile:
            fused_configs.append((c1.to_codegen_dict(), c2.to_codegen_dict()))

X = graph.PlaceHolderNode("X")
W1 = graph.PlaceHolderNode("W1")
B1 = graph.PlaceHolderNode("B1")
X1 = graph.ComputeNode([B1, X, W1], args1)
W2 = graph.PlaceHolderNode("W2")
B2 = graph.PlaceHolderNode("B2")
X2 = graph.ComputeNode([B2, X1, W2], args2)
Y = graph.OutputNode(X2)
topo = graph.topo_order([X, W1, B1, W2, B2])
for fused_config in fused_configs:
    print(fused_config)
    config = { X1 : fused_config[0], X2 : fused_config[1]}
    code, block_size, grid_size, args = memopt.utils.compose_global_kernel(topo, config, "cuda", name="Fused")
    code = memopt.utils.append_host_call(code, block_size, grid_size, len(args), name="Fused", measure_time=True)
    print(code)
    lib = memopt.utils.compile_and_load(code)
    lib.function.restype = ctypes.c_float
    print(memopt.utils.profile(lib, args))
