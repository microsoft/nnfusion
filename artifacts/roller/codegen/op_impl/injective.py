import os
import sys
import tvm
import numpy as np
from tvm import te, topi
from codegen import CodeGenerator
from tvm.contrib import nvcc

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", arch="compute_70")
    return ptx

def add_layer(shape):
    data1 = te.placeholder(shape, name="data1")
    data2 = te.placeholder(shape, name="data2")
    add = topi.add(data1, data2)
    return [data1, data2, add]

def mul_layer(shape):
    data1 = te.placeholder(shape, name="data1")
    data2 = te.placeholder(shape, name="data2")
    mul = topi.multiply(data1, data2)
    return [data1, data2, mul]

def tanh_layer(shape):
    data = te.placeholder(shape, name="data")
    tanh = topi.tanh(data)
    return [data, tanh]

def sigmoid_layer(shape):
    data = te.placeholder(shape, name="data")
    sigmoid = topi.sigmoid(data)
    return [data, sigmoid]

def codegen(op, shape):
    layer = None
    tile_dict = None
    
    if op == "add":
        layer = add_layer(shape)
        tile_dict = {"ax0": [4, 4], "ax1": [4, 4]}
    elif op == "mul":
        layer = mul_layer(shape)
        tile_dict = {"ax0": [4, 4], "ax1": [4, 4]}
    elif op == "tanh":
        layer = tanh_layer(shape)
        tile_dict = {"i0": [4, 4], "i1": [4, 4]}
    elif op == "sigmoid":
        layer = sigmoid_layer(shape)
        tile_dict = {"i0": [4, 4], "i1": [4, 4]}
    else:
        raise ValueError("unrecognized type: " + op)

    target_stage = layer[-1].name
    s = te.create_schedule(layer[-1].op)
    generator = CodeGenerator()
    generator.rewrite_schedule(s, tile_dict, False, False, target_stage)
    func = tvm.build(s, layer, "cuda")
    with open('{}.cuh'.format(op), 'w') as ouf:
        ouf.write('#ifndef KERNELH\n#define KERNELH\n')
        ouf.write(func.imported_modules[0].get_source())
        ouf.write('#endif\n')

    ctx = tvm.gpu(0)
    A_h = np.random.uniform(size=shape).astype("float32")
    B_h = np.random.uniform(size=shape).astype("float32")

    A_d = tvm.nd.array(A_h, ctx)
    B_d = tvm.nd.array(B_h, ctx)
    C_d = tvm.nd.array(np.zeros(shape, dtype="float32"), ctx)
    if op == "add":
        print(layer)
        func(*layer)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
        print("add: %f ms" % (evaluator(A_d, B_d, C_d).mean * 1e3))
        C_h = np.add(A_h, B_h)
        np.testing.assert_allclose(C_h, C_d.asnumpy(), atol=1e-4)
    elif op == "mul":
        func(A_d, B_d, C_d)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
        print("mul: %f ms" % (evaluator(A_d, B_d, C_d).mean * 1e3))
        C_h = np.multiply(A_h, B_h)
        np.testing.assert_allclose(C_h, C_d.asnumpy(), atol=1e-4)
    elif op == "tanh":
        func(A_d, C_d)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
        print("tanh: %f ms" % (evaluator(A_d, C_d).mean * 1e3))
        C_h = np.tanh(A_h)
        np.testing.assert_allclose(C_h, C_d.asnumpy(), atol=1e-4)
    elif op == "sigmoid":
        func(A_d, C_d)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
        print("sigmoid: %f ms" % (evaluator(A_d, C_d).mean * 1e3))
        C_h = 1 / (1 + np.exp(-A_h))
        np.testing.assert_allclose(C_h, C_d.asnumpy(), atol=1e-4)
            
def main():
    shape = tuple([int(s) for s in sys.argv[2:]])
    op = sys.argv[1]
    print(op, shape)
    codegen(op, shape)

main()