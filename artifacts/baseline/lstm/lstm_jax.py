import jax.numpy as jnp
from jax import random, lax
from functools import partial
import jax
from flax import linen as nn
from time import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

def sigmoid(x):
    return 0.5*(jnp.tanh(x) + 1.0)


def lstm_scan_body(params, carry, inp):
    state_c, state_h = carry

    z = jnp.matmul(inp, params['weight_ih']) + jnp.matmul(state_h, params['weight_hh']) + params['bias_ih'] + params['bias_hh']
    
    ingate1, forgetgate1, cellgate1, outgate1 = jnp.split(z, 4, axis=1)

    ingate = sigmoid(ingate1)
    forgetgate = sigmoid(forgetgate1)
    cellgate = jnp.tanh(cellgate1)
    outgate = sigmoid(outgate1)

    state_c = (forgetgate * state_c) + (ingate * cellgate)
    state_h = outgate * jnp.tanh(state_c)

    return (state_c, state_h), state_h


def lstm_lax(params, _hidden_size, _num_layers, inputs):
    batch_size = inputs.shape[1]
    out = inputs
    for i in range(_num_layers):
        state_c = jnp.zeros([batch_size, _hidden_size])
        state_h = jnp.zeros([batch_size, _hidden_size])
        f = partial(lstm_scan_body, params[i])
        (_c, _h), out = lax.scan(f, (state_c, state_h), out)
    return out

# faster
def lstm_loop(params, _hidden_size, _num_layers, inputs):
    seq_len = inputs.shape[0]
    batch_size = inputs.shape[1]
    out = inputs
    for i in range(_num_layers):
        state_c = jnp.zeros([batch_size, _hidden_size])
        state_h = jnp.zeros([batch_size, _hidden_size])
        for j in range(seq_len):
            (state_c, state_h), output = lstm_scan_body(params[i], (state_c, state_h), out[j])
            out = out.at[j].set(output)
    return out

# def lstm_loop(params, _hidden_size, _num_layers, inputs):
#     seq_len = inputs.shape[0]
#     batch_size = inputs.shape[1]
#     state_c = [jnp.zeros([batch_size, _hidden_size]) for _ in range(_num_layers)]
#     state_h = [jnp.zeros([batch_size, _hidden_size]) for _ in range(_num_layers)]
#     for i in range(seq_len):
#         cur_input = inputs[i]
#         for j in range(_num_layers):
#             (state_c[j], state_h[j]), cur_input = lstm_scan_body(params[j], (state_c[j], state_h[j]), cur_input)

#     return cur_input

n_warmup = 100
n_run = 100


def recursive_block_ready(out):
    for o in out.values():
        if hasattr(o, 'block_until_ready'):
            o.block_until_ready()
        else:
            recursive_block_ready(o)


def test_model(enable_jit, enable_scan, enable_grad, batch_size, weights, *params):
    input_size, hidden_size, num_layers, seq_len = params
    inp = jax.device_put(jnp.ones((seq_len, batch_size, input_size)))

    if enable_scan:
        func1 = lstm_lax
    else:
        func1 = lstm_loop

    if enable_grad:
        func2 = jax.grad(lambda a, b, c, d: jnp.sum(func1(a, b, c, d)))
    else:
        func2 = func1

    if enable_jit:
        func3 = jax.jit(func2, static_argnums=(1, 2))
    else:
        func3 = func2

    print("----batch_size={}--jit={}--scan={}--grad={}---".format(batch_size, enable_jit, enable_scan, enable_grad), flush=True)
    print("[warmup]")
    for i in range(n_warmup):
        t0 = time()
        out = func3(weights, hidden_size, num_layers, inp)
        if enable_grad:
            recursive_block_ready(out)
        else:
            out.block_until_ready()
        print("Time {} ms".format((time() - t0) * 1000), flush=True)

    timer = Timer("ms")
    profile_start(platform)
    for i in range(n_run):
        timer.start()
        out = func3(weights, hidden_size, num_layers, inp)
        if enable_grad:
            recursive_block_ready(out)
        else:
            out.block_until_ready()
        timer.log()
    profile_stop(platform)
    timer.report()


if __name__ == '__main__':
    for dev in jax.devices(): print(dev)
    key = random.PRNGKey(0)
    input_size = 256
    hidden_size = 256
    seq_len = 64
    num_layers = 10

    params = [{
        'weight_ih': random.normal(key,  (input_size, hidden_size * 4)),
        'weight_hh': random.normal(key, (hidden_size, hidden_size * 4)),
        'bias_ih': random.normal(key, (hidden_size * 4, )),
        'bias_hh': random.normal(key, (hidden_size * 4, )),
    } for _ in range(num_layers)]

    if not arguments.overhead_test:
        test_model(True, True, False, arguments.bs, params, input_size, hidden_size, num_layers, seq_len)
    else:
        if arguments.unroll:
            test_model(True, False, False, 1, params, input_size, hidden_size, num_layers, seq_len)
        else:
            test_model(True, True, False, 1, params, input_size, hidden_size, num_layers, seq_len)

