# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Test code for dense tensorcore operator"""
import numpy as np
import tvm
from tvm import topi
from tvm.te import schedule
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm import te
from tvm.contrib.pickle_memoize import memoize
import tvm.testing
from tvm.topi.cuda import dense_tensorcore, schedule_dense_tensorcore
from tvm import autotvm
import logging
import sys

dtype = "float16"

@autotvm.template("topi_matmul_tensorcore")
def topi_matmul_tensorcore(batch, in_dim, out_dim):
    """Dense tensorcore verify function"""
    A = te.placeholder((batch, in_dim), name="A", dtype=dtype)
    B = te.placeholder((out_dim, in_dim), name="B", dtype=dtype)
    matmul = dense_tensorcore(A, B, None, dtype)
    s = schedule_dense_tensorcore([matmul])
    return s, [A, B, matmul]

@tvm.testing.requires_tensorcore
def verify_dense(batch, in_dim, out_dim, dtype, use_bias=True):
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    task = autotvm.task.create(
        "topi_matmul_tensorcore", args=(batch, in_dim, out_dim), target="cuda"
    )
    print(task.config_space)

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    op_name = "tuned_matmul_tensorcor_%d_%d_%d" % (batch, in_dim, out_dim)
    log_name = "tuned_kernels/" + op_name + ".log"
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=10000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_name)],
    )

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(log_name):
        with tvm.target.Target("cuda"):
            s, arg_bufs = topi_matmul_tensorcore(batch, in_dim, out_dim)
            func = tvm.build(s, arg_bufs)

    # get the data
    a_np = np.random.uniform(size=(batch, in_dim)).astype(np.float16)
    b_np = np.random.uniform(size=(out_dim, in_dim)).astype(np.float16)

    dev = tvm.cuda(0)
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(np.zeros((batch, out_dim), dtype=dtype), device=dev)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, dev, number=400)
    print("Time cost of this operator: %f" % evaluator(a_tvm, b_tvm, c_tvm).mean)

if __name__ == "__main__":
    # verify_dense(65536, 1024, 1024, "float16", use_bias=False)
    # verify_dense(65536, 1024, 4096, "float16", use_bias=False)
    # verify_dense(65536, 4096, 1024, "float16", use_bias=False)
    # verify_dense(65536, 16384, 1024, "float16", use_bias=False)
    verify_dense(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), "float16", use_bias=False)
