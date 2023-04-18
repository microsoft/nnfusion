import jax.numpy as jnp
import numpy as np
from jax import random, lax
from functools import partial
import jax

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)
import os

START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64
n_warmup = 100
n_run = 100

def attention_body_fun(_, val):
    params, (x, k, v, gen_id) = val
    # jax.debug.print("gen_id {gen_id}", gen_id=gen_id)
    batch_size, num_head, _, size_per_head = x.shape
    q = jnp.matmul(x, params['weight_q'])
    k_one = jnp.reshape(jnp.matmul(x, params['weight_k']), (batch_size, num_head, size_per_head))
    k = k.at[:, :, gen_id, :].set(k_one)
    v_one = jnp.reshape(jnp.matmul(x, params['weight_v']), (batch_size, num_head, size_per_head))
    v = v.at[:, :, gen_id, :].set(v_one)
    attn = jnp.matmul(k, q.transpose((0, 1, 3, 2))).transpose((0, 1, 3, 2))
    attn = attn * 0.125
    attn = jax.nn.softmax(attn, axis=3)
    x = jnp.matmul(attn, v)
    x = jnp.matmul(x, params['weight_o'])
    gen_id = gen_id + 1
    # jax.debug.print("gen_id {gen_id}", gen_id=gen_id)
    return params, (x, k, v, gen_id)

def attention_fun(start_len, seq_len, params, x, k, v):
    gen_id = start_len
    val = (x, k, v, gen_id)
    # use jax.lax.fori_loop
    params, (x, k, v, gen_id) = jax.lax.fori_loop(start_len, seq_len, attention_body_fun, (params, val))
    return x, k, v

def attention_fun_loop(start_len, seq_len, params, x, k, v):
    gen_id = start_len
    # use jax.lax.fori_loop
    # jax.debug.print("start_len {start_len} seq_len {seq_len}", start_len=start_len, seq_len=seq_len)
    for i in range(start_len, seq_len):
        params, (x, k, v, gen_id) = attention_body_fun(0, (params, (x, k, v, gen_id)))
    return x, k, v

def recursive_block_ready(out):
    for o in out:
        if hasattr(o, 'block_until_ready'):
            o.block_until_ready()
        else:
            recursive_block_ready(o)

def test_model(enable_jit, enable_loop, batch_size, weights):
    key = random.PRNGKey(233)
    x = jax.random.normal(key, (batch_size, NUM_HEAD, 1, SIZE_PER_HEAD), dtype=np.float32)
    k = jax.numpy.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD), dtype=np.float32)
    k = k.at[:, :, :START_LEN, :].set(jax.random.normal(key, (batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD), dtype=np.float32))
    v = jax.numpy.zeros((batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD), dtype=np.float32)
    v = v.at[:, :, :START_LEN, :].set(jax.random.normal(key, (batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD), dtype=np.float32))

    if enable_loop:
        attention_call_fun = attention_fun
    else:
        attention_call_fun = attention_fun_loop
    attention_call_fun = partial(attention_call_fun, START_LEN, SEQ_LEN)
    if enable_jit:
        attention_call_fun = jax.jit(attention_call_fun)
    else:
        attention_call_fun = attention_call_fun
    
    print("---batch_size={}--jit={}--loop={}---".format(batch_size, enable_jit, enable_loop))
    for i in range(n_warmup):
        out = attention_call_fun(weights, x, k, v)
        recursive_block_ready(out)
        # exit(0)

    timer = Timer("ms")
    profile_start(platform)
    for i in range(n_run):
        timer.start()
        out = attention_call_fun(weights, x, k, v)
        recursive_block_ready(out)
        timer.log()
    profile_stop(platform)
    timer.report()


if __name__ == '__main__':
    key = random.PRNGKey(0)
    initializer = jax.nn.initializers.glorot_uniform()
    params = {
        'weight_q': initializer(key, (NUM_HEAD, SIZE_PER_HEAD, SIZE_PER_HEAD), dtype=np.float32),
        'weight_k': initializer(key, (NUM_HEAD, SIZE_PER_HEAD, SIZE_PER_HEAD), dtype=np.float32),
        'weight_v': initializer(key, (NUM_HEAD, SIZE_PER_HEAD, SIZE_PER_HEAD), dtype=np.float32),
        'weight_o': initializer(key, (NUM_HEAD, SIZE_PER_HEAD, SIZE_PER_HEAD), dtype=np.float32),
    }
    if not arguments.overhead_test:
        test_model(True, True, arguments.bs, params)
    else:
        if arguments.unroll:
            test_model(True, False, 1, params)
        else:
            test_model(True, True, 1, params)
