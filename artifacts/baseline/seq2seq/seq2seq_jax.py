import jax.numpy as jnp
import numpy as np
from jax import random, lax
from functools import partial
import jax
from time import time
import os

from jax.config import config
config.update("jax_enable_x64", True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
parser.add_argument('--rate', type=int, default=-1)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from timer import Timer
from nvprof import profile_start, profile_stop, enable_profile
enable_profile(platform)

MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256

def seq2seq_body_fun(params, val):
    std, h, c, output_all, id, output = val
    x = params['embedding'][output]
    # jax.debug.print("output.shape={output.shape} x.shape={x.shape}", x=x, output=output)
    ih = jnp.matmul(x, params['weight_ih'])
    hh = jnp.matmul(h, params['weight_hh'])
    # jax.debug.print("hehe id={id}, ih={ih[0]}, hh={hh[0]}", id=id, ih=ih, hh=hh)
    i0 = ih[0] + params['bias_ih_0']
    i1 = ih[1] + params['bias_ih_1']
    i2 = ih[2] + params['bias_ih_2']
    i3 = ih[3] + params['bias_ih_3']
    h0 = hh[0] + params['bias_hh_0']
    h1 = hh[1] + params['bias_hh_1']
    h2 = hh[2] + params['bias_hh_2']
    h3 = hh[3] + params['bias_hh_3']
    ingate = jax.nn.sigmoid(i0 + h0)
    forgetgate = jax.nn.sigmoid(i1 + h1)
    cellgate = jnp.tanh(i2 + h2)
    outgate = jax.nn.sigmoid(i3 + h3)

    c = (forgetgate * c) + (ingate * cellgate)
    h = outgate * jnp.tanh(c)

    # jax.debug.print("id={id}, hidden={hidden}", id=id, hidden=hidden[0][:10])
    output = jnp.matmul(h, params['weight_ho']) + params['bias_ho']
    output = output + std[id]
    # jax.debug.print("before argmax id={id}, output={output}", id=id, output=output)
    output = output.argmax(axis=1)
    # jax.debug.print("after argmax id={id}, output={output}", id=id, output=output)
    output_all = output_all.at[id].set(output)
    id = id + 1
    return (std, h, c, output_all, id, output)


def seq2seq_cond_fun(params, max_length, val):
    std, h, c, output_all, id, output, = val
    cond1 = id < max_length
    cond2 = jnp.any(output_all[id - 1] != 0, axis=0)
    # jax.debug.print("cond1={cond1}, cond2={cond2}", cond1=cond1, cond2=cond2)
    return (id == 0) | (cond1 & cond2)


def seq2seq(params, max_length, hidden_size, encoder_output, std, h, c):
    batch_size = encoder_output.shape[1]
    hidden = jnp.zeros((batch_size, hidden_size))
    output_all = jnp.zeros((max_length, batch_size), dtype=jnp.int64)
    id = jnp.array(0)
    output = jnp.ones((batch_size,), dtype=jnp.int64)
    val = (std, h, c, output_all, id, output,)
    seq2seq_cond_call = partial(seq2seq_cond_fun, params, max_length)
    seq2seq_body_call = partial(seq2seq_body_fun, params)
    val = lax.while_loop(seq2seq_cond_call, seq2seq_body_call, val)
    return val[3], val[1], val[2]


def seq2seq_unroll(params, max_length, hidden_size, encoder_output, std, h, c):
    batch_size = encoder_output.shape[1]
    hidden = jnp.zeros((batch_size, hidden_size))
    output_all = jnp.zeros((max_length, batch_size), dtype=jnp.int64)
    id = jnp.array(0)
    output = jnp.ones((batch_size,), dtype=jnp.int64)
    val = (std, h, c, output_all, id, output,)
    seq2seq_cond_call = partial(seq2seq_cond_fun, params, max_length)
    seq2seq_body_call = partial(seq2seq_body_fun, params)
    cond = seq2seq_cond_call(val)
    for i in range(10):
        val = seq2seq_body_call(val)
        cond = cond & seq2seq_cond_call(val)
    return val[3], val[1], val[2], cond


n_warmup = 100
n_run = 100

def recursive_block_ready(out):
    for o in out:
        if hasattr(o, 'block_until_ready'):
            o.block_until_ready()
        else:
            recursive_block_ready(o)

def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = np.fromfile(s + ".bin", dtype=dtype).reshape(shape)
    return tensor

def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = np.zeros((bs, MAX_LENGTH), dtype=std.dtype)
    padded_std[:, :std.shape[1]] = std
    mask = np.zeros((bs, MAX_LENGTH, OUTPUT_SIZE))
    mask[np.expand_dims(np.arange(bs), 1), np.expand_dims(np.arange(MAX_LENGTH), 0), padded_std] = 1000000.0
    mask = np.transpose(mask, axes=(1, 0, 2))
    return mask

tokens = read_bin('../../data/tatoeba-eng-fra/tokens', dtype=np.int64)
# print(tokens[0])
masks = gen_mask_from_sequence(tokens)
masks = jnp.array(masks)
fixed_data_prefix = '../../data/seq2seq/fix_test'

def test_model(enable_jit, enable_while, batch_size, weights):
    key = random.PRNGKey(233)
    encoder_output = jax.random.normal(key, (MAX_LENGTH, batch_size, HIDDEN_SIZE), dtype=jnp.float32)
    h = jax.random.normal(key, (batch_size, HIDDEN_SIZE), dtype=jnp.float32)
    c = jax.random.normal(key, (batch_size, HIDDEN_SIZE), dtype=jnp.float32)

    if not enable_while: raise NotImplementedError
    seq2seq_call_fun = partial(seq2seq, weights, MAX_LENGTH, HIDDEN_SIZE)
    
    if enable_jit:
        seq2seq_jit = jax.jit(seq2seq_call_fun)
    else:
        seq2seq_jit = seq2seq_call_fun
    
    print("---batch_size={}--jit={}--while={}---".format(batch_size, enable_jit, enable_while))
    for i in range(0, 6400, batch_size):
        if i >= n_warmup * batch_size: break
        mask = masks[:, i:i+batch_size].clone().block_until_ready()
        out = seq2seq_jit(encoder_output, mask, h, c) # !! TODO JIT
        recursive_block_ready(out)
    timer = Timer("ms")
    profile_start(platform)
    for i in range(0, 6400, batch_size):
        if i >= n_run * batch_size: break
        mask = masks[:, i:i+batch_size].clone().block_until_ready()
        timer.start()
        out = seq2seq_jit(encoder_output, mask, h, c)
        recursive_block_ready(out)
        timer.log()
    profile_stop(platform)
    timer.report()


def test_model_fix(enable_jit, enable_while, batch_size, weights):
    if batch_size != 1: raise NotImplementedError
    mask = read_bin(os.path.join(fixed_data_prefix, "mask"))
    encoder_output = read_bin(os.path.join(fixed_data_prefix, "encoder_output"))
    output_all = read_bin(os.path.join(fixed_data_prefix, "output_all"), dtype=np.int64)
    hidden = read_bin(os.path.join(fixed_data_prefix, "hidden"))
    h = read_bin(os.path.join(fixed_data_prefix, "h"))
    c = read_bin(os.path.join(fixed_data_prefix, "c"))

    if enable_while:
        seq2seq_call_fun = partial(seq2seq, weights, MAX_LENGTH, HIDDEN_SIZE)
    else:
        seq2seq_call_fun = partial(seq2seq_unroll, weights, MAX_LENGTH, HIDDEN_SIZE)
    if enable_jit:
        seq2seq_jit = jax.jit(seq2seq_call_fun)
    else:
        seq2seq_jit = seq2seq_call_fun

    mask = jnp.array(mask).block_until_ready()
    encoder_output = jnp.array(encoder_output).block_until_ready()
    h = jnp.array(h).block_until_ready()
    c = jnp.array(c).block_until_ready()

    out = seq2seq_jit(encoder_output, mask, h, c)
    recursive_block_ready(out)
    
    print("---batch_size={}--jit={}--while={}---".format(batch_size, enable_jit, enable_while))
    for i in range(0, 6400, batch_size):
        if i >= n_warmup * batch_size: break
        out = seq2seq_jit(encoder_output, mask, h, c) # !! TODO JIT
        recursive_block_ready(out)
    timer = Timer("ms")
    profile_start(platform)
    for i in range(0, 6400, batch_size):
        if i >= n_run * batch_size: break
        timer.start()
        out = seq2seq_jit(encoder_output, mask, h, c)
        recursive_block_ready(out)
        timer.log()
    profile_stop(platform)
    timer.report()


if __name__ == '__main__':
    key = random.PRNGKey(0)
    hidden_size = HIDDEN_SIZE
    params = {
        'weight_ih': random.normal(key, (4, hidden_size, hidden_size), dtype=np.float32),
        'weight_hh': random.normal(key, (4, hidden_size, hidden_size), dtype=np.float32),
        'bias_ih_0': random.normal(key, (hidden_size,), dtype=np.float32),
        'bias_hh_0': random.normal(key, (hidden_size,), dtype=np.float32),
        'bias_ih_1': random.normal(key, (hidden_size,), dtype=np.float32),
        'bias_hh_1': random.normal(key, (hidden_size,), dtype=np.float32),
        'bias_ih_2': random.normal(key, (hidden_size,), dtype=np.float32),
        'bias_hh_2': random.normal(key, (hidden_size,), dtype=np.float32),
        'bias_ih_3': random.normal(key, (hidden_size,), dtype=np.float32),
        'bias_hh_3': random.normal(key, (hidden_size,), dtype=np.float32),
        'weight_ho': random.normal(key, (hidden_size, OUTPUT_SIZE), dtype=np.float32),
        'bias_ho': random.normal(key, (OUTPUT_SIZE,), dtype=np.float32),
        'embedding': random.normal(key, (OUTPUT_SIZE, hidden_size), dtype=np.float32),
    }
    # exit(0)
    if not arguments.overhead_test:
        test_model(True, True, arguments.bs, params)
    else:
        test_model_fix(True, not arguments.unroll, 1, params)

