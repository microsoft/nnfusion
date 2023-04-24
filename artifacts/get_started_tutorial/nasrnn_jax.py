import jax
import jax.numpy as jnp
from jax import grad, random, jit, lax
from functools import partial
from time import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
arguments = parser.parse_args()

def sigmoid(x):
    return 0.5*(jnp.tanh(x) + 1.0)

def nasrnn_scan_body(params, i, val):
    inputs, (state_c, state_m) = val
    inp = inputs[i]

    ii = jnp.matmul(inp, params['weight_ih'])
    hh = jnp.matmul(state_m, params['weight_hh'])

    i0, i1, i2, i3, i4, i5, i6, i7 = jnp.split(ii, 8, axis=1)
    h0, h1, h2, h3, h4, h5, h6, h7 = jnp.split(hh, 8, axis=1)

    layer1_0 = sigmoid(i0 + h0)
    layer1_1 = jax.nn.relu(i1 + h1)
    layer1_2 = sigmoid(i2 + h2)
    layer1_3 = jax.nn.relu(i3 + h3)
    layer1_4 = jnp.tanh(i4 + h4)
    layer1_5 = sigmoid(i5 + h5)
    layer1_6 = jnp.tanh(i6 + h6)
    layer1_7 = sigmoid(i7 + h7)

    l2_0 = jnp.tanh(layer1_0 * layer1_1)
    l2_1 = jnp.tanh(layer1_2 + layer1_3)
    l2_2 = jnp.tanh(layer1_4 * layer1_5)
    l2_3 = sigmoid(layer1_6 + layer1_7)

    # Inject the cell
    l2_0 = jnp.tanh(l2_0 + state_c)

    # Third layer
    l3_0_pre = l2_0 * l2_1
    state_c = l3_0_pre  # create new cell
    l3_0 = l3_0_pre
    l3_1 = jnp.tanh(l2_2 + l2_3)

    # Final layer
    state_m = jnp.tanh(l3_0 * l3_1)
    return inputs, (state_c, state_m)


def nasrnn_lax(params, _hidden_size, inputs):
    state_c = jnp.zeros([inputs.shape[1], _hidden_size])
    state_m = jnp.zeros([inputs.shape[1], _hidden_size])
    f = partial(nasrnn_scan_body, params)
    _, (_, new_h) = lax.fori_loop(0, inputs.shape[0], f, (inputs, (state_c, state_m)))
    return new_h

n_warmup = 100
n_run = 100


def test_model(enable_jit, batch_size, weights, *params):
    input_size, hidden_size, seq_len = params
    inp = jax.device_put(jnp.ones((seq_len, batch_size, input_size)))

    if enable_jit:
        func = jax.jit(nasrnn_lax, static_argnums=(1,))
    else:
        func = nasrnn_lax

    print("----batch_size={}--jit={}----".format(batch_size, enable_jit))
    print("[warmup]")
    for i in range(n_warmup):
        t0 = time()
        out = func(weights, hidden_size, inp)
        out.block_until_ready()
        print("Time {} ms".format((time() - t0) * 1000), flush=True)

    iter_times = []
    for i in range(n_run):
        start_time = time()
        out = func(weights, hidden_size, inp)
        out.block_until_ready()
        iter_time = (time() - start_time) * 1000
        iter_times.append(iter_time)
    print("\033[31mSummary: [min, max, mean] = [%f, %f, %f] ms\033[m" % (
            min(iter_times), max(iter_times), sum(iter_times) / len(iter_times)))


if __name__ == '__main__':
    key = random.PRNGKey(0)
    input_size = 256
    hidden_size = 256

    seq_len = 1000

    params = {
        'weight_ih': random.normal(key,  (input_size, hidden_size * 8)),
        'weight_hh': random.normal(key, (hidden_size, hidden_size * 8)),
    }

    test_model(True, arguments.bs, params, input_size, hidden_size, seq_len)