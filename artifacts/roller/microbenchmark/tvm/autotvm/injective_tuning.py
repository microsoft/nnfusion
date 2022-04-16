import logging
import sys
import numpy as np 

import tvm
from tvm import te, topi, testing
import tvm.testing

import time

def tvm_injective_bias_add_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    data2 = te.placeholder(shape[1:], name='input1')
    out = te.compute(shape, lambda x, y: data1[x, y] + data2[y])
    s = topi.cuda.injective.schedule_injective(out)
    return s, [data1, data2, out]

def tvm_injective_add_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    data2 = te.placeholder(shape, name='input1')
    out = topi.add(data1, data2)
    s = topi.cuda.injective.schedule_injective(out)
    return s, [data1, data2, out]

def tvm_injective_mul_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    data2 = te.placeholder(shape, name='input1')
    out = topi.multiply(data1, data2)
    s = topi.cuda.injective.schedule_injective(out)
    return s, [data1, data2, out]

def tvm_injective_tanh_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    out = topi.tanh(data1)
    s = topi.cuda.injective.schedule_injective(out)
    return s, [data1, out]

def tvm_injective_sigmoid_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    out = topi.sigmoid(data1)
    s = topi.cuda.injective.schedule_injective(out)
    return s, [data1, out]

def tvm_injective_relu_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    out = topi.nn.relu(data1)
    s = topi.cuda.injective.schedule_injective(out)
    return s, [data1, out]

def tvm_injective_transpose_tune_op(*shape):
    data1 = te.placeholder(shape, name='input0')
    out = topi.transpose(data1)
    s = topi.cuda.transform.schedule_transpose(out)
    return s, [data1, out]

def tune_injective(t, shape, n_trial=1000):
    op = None
    if t == "add":
        op = tvm_injective_add_tune_op
    elif t == "transpose":
        op = tvm_injective_transpose_tune_op
    elif t == "biasadd":
        op = tvm_injective_bias_add_tune_op
    elif t == "mul":
        op = tvm_injective_mul_tune_op
    elif t == "tanh":
        op = tvm_injective_tanh_tune_op
    elif t == "sigmoid":
        op = tvm_injective_sigmoid_tune_op
    elif t == "relu":
        op = tvm_injective_relu_tune_op
    else:
        raise ValueError("unrecognized type: " + t)
    
    with tvm.target.Target("cuda"):
        s, arg_bufs = op(*shape)
        func = tvm.build(s, arg_bufs)
    
    dev = tvm.cuda()
    a_np = np.random.uniform(size=shape).astype(np.float32)
    b_np = np.random.uniform(size=shape).astype(np.float32)
    c_np = np.random.uniform(size=shape).astype(np.float32)

    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    
    if t == "add" or t == "mul":
        func(a_tvm, b_tvm, c_tvm)
        evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
        print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm, c_tvm).mean)
    elif t == "biasadd":
        b_np = np.random.uniform(size=shape[1:]).astype(np.float32)
        b_tvm = tvm.nd.array(b_np, device=dev)
        func(a_tvm, b_tvm, c_tvm)
        evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
        print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm, c_tvm).mean)
    elif t == "transpose":
        b_np = np.random.uniform(size=(shape[1], shape[0])).astype(np.float32)
        b_tvm = tvm.nd.array(b_np, device=dev)
        func(a_tvm, b_tvm)
        evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
        print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm).mean)
    else:
        func(a_tvm, b_tvm)
        evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
        print("Time cost of this operator: %.10f" % evaluator(a_tvm, b_tvm).mean)

    print("Lowered TIR:")
    print(tvm.lower(s, arg_bufs, simple_mode=True))
    print(func.imported_modules[0].get_source())  # print kernel code
    
def main():
    shape = tuple([int(s) for s in sys.argv[2:]])
    t = sys.argv[1]
    print(t, shape)
    tune_injective(t, shape)

start_time = time.time()
main()
print("compilation time: %s" % (time.time() - start_time))