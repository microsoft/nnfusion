import jax
import jax.numpy as jnp
import numpy as np
from jax import random
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

depth = 7
n = 2 ** depth - 1

n_warmup = 100
n_run = 100

def rae(params, left, right, is_leaf, inp, root):
    if is_leaf[root]:
        activation = inp[root]
    else:
        a = rae(params, left, right, is_leaf, inp, left[root])
        b = rae(params, left, right, is_leaf, inp, right[root])
        ab = jnp.concatenate((a, b))
        e = jnp.dot(ab, params['weight']) + params['bias']
        activation = jnp.tanh(e)    
    return activation


def test_model(enable_jit, params, batch_size):
    root = 64
    if enable_jit:
        left = tuple([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
        right = tuple([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
        is_leaf = tuple([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        x = jnp.ones([n, 512])
        func = jax.jit(rae, static_argnums=(1, 2, 3, 5))
    else:
        left = jnp.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 30, 28, 27, 26, 33, 25, 24, 23, 22, 21, 20, 32, 18, 17, 15, 14, 13, 12, 11, 46, 8, 6, 5, 53, 10, 52, 2, 1, 0, 4, 58, 44])
        right = jnp.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 19, 45, 16, 47, 48, 49, 50, 51, 9, 7, 54, 55, 56, 57, 3, 59, 60, 61, 62, 63])
        is_leaf = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        x = jnp.ones([n, 512])
        func = rae

    print("----batch_size={}---jit={}----".format(batch_size, enable_jit))
    print("[warmup]")
    for i in range(n_warmup):
        t0 = time()
        out = func(params, left, right, is_leaf, x, root)
        out.block_until_ready()
        print("Time {} ms".format((time() - t0) * 1000))
    profile_start(platform)
    timer = Timer("ms")
    for i in range(n_run):
        timer.start()
        out = func(params, left, right, is_leaf, x, root)
        out.block_until_ready()
        timer.log()
    profile_stop(platform)
    timer.report()


def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = jnp.array(np.fromfile(s + ".bin", dtype=dtype)).reshape(shape)
    return tensor


def load_trees():
    prefix = '../../artifacts/data/sst/'
    left_tensor = read_bin(prefix + 'left', np.int64)
    right_tensor = read_bin(prefix + 'right', np.int64)
    is_leaf_tensor = read_bin(prefix + 'is_leaf', bool)
    root_tensor = read_bin(prefix + 'root', np.int64)
    input_tensor = read_bin(prefix + 'input')
    output_tensor = read_bin(prefix + 'output')
    return left_tensor, right_tensor, is_leaf_tensor, root_tensor, input_tensor, output_tensor


def test_model_sst(enable_jit, params, batch_size):
    assert enable_jit == False
    left_tensor, right_tensor, is_leaf_tensor, root_tensor, input_tensor, output_tensor = load_trees()
    print("----batch_size={}---jit={}----".format(batch_size, enable_jit))
    print("[warmup]")
    for i in range(n_warmup):
        t0 = time()
        out = rae(params, left_tensor[i], right_tensor[i], is_leaf_tensor[i], input_tensor, root_tensor[i].item())
        out.block_until_ready()
        print("Time {} ms".format((time() - t0) * 1000))
    profile_start(platform)
    timer = Timer("ms")
    for i in range(n_run):
        left = left_tensor[i].block_until_ready()
        right = right_tensor[i].block_until_ready()
        is_leaf = is_leaf_tensor[i].block_until_ready()
        root = root_tensor[i].item()
        timer.start()
        out = rae(params, left, right, is_leaf, input_tensor, root)
        out.block_until_ready()
        timer.log()
    profile_stop(platform)
    timer.report()


if __name__ == '__main__':
    key = random.PRNGKey(0)
    params = {
        'weight': random.normal(key, (1024, 512)),
        'bias': random.normal(key, (512,)),
    }

    # test_model(False, params, 1)
    # test_model(True, params, 1)
    assert arguments.bs == 1
    if not arguments.overhead_test:
        test_model_sst(False, params, arguments.bs)
    else:
        test_model(arguments.unroll, params, arguments.bs)


