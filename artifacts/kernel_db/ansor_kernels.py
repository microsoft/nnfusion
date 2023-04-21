from test_config import *
# from roller.op import *
import argparse
import tvm
from tvm import te, auto_scheduler
import logging
import sys
import numpy as np
from db import save_to_db
import re

parser = argparse.ArgumentParser()
parser.add_argument("--tid", type=int)
parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--inject", action="store_true")
args = parser.parse_args()
op_func = None

if __name__ == '__main__':
    with open("ansor.id") as f:
        ids = f.readlines()
        identifier = ids[args.tid].strip().split(":::")[0]
    
    func_name, shape, config = get_func(identifier)
    prefix = f"ansor_kernels/{func_name}_" + "_".join([str(x) for x in shape]) + "_ansor"
    log_name = os.path.join(prefix, "tune.log")

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    target = tvm.target.Target("cuda")
    op_func = globals()[func_name]

@auto_scheduler.register_workload
def get_expr(func_name, shape):
    out = op_func(shape)
    # out = globals()[func_name](shape)
    io_tensors = []
    for tensor in out[0]:
        if isinstance(tensor.op, te.tensor.PlaceholderOp):
            io_tensors.append(tensor)
    io_tensors.append(out[1][-1])
    return io_tensors

if __name__ == '__main__':
    task = tvm.auto_scheduler.SearchTask(func=get_expr, args=(
        func_name, shape), target=target)
    print(task.compute_dag)
    prefix = f"ansor_kernels/{func_name}_" + "_".join([str(x) for x in shape]) + "_ansor"
    log_name = os.path.join(prefix, "tune.log")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    if args.trial > 0:
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=args.trial,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_name)],
            verbose=2,
        )
        task.tune(tune_option)
    if not args.inject: exit(0)
    sch, args = task.apply_best(log_name)
    del measure_ctx

    lowered = tvm.lower(sch, args, simple_mode=True)
    # print(lowered)
    lowered_text = str(lowered)
    launch_config = {
        'blockIdx.x': 1,
        'blockIdx.y': 1,
        'blockIdx.z': 1,
        'threadIdx.x': 1,
        'threadIdx.y': 1,
        'threadIdx.z': 1,
    }
    func = tvm.build(sch, args, target)
    assert(len(func.imported_modules) == 1)
    best_source = func.imported_modules[0].get_source()
    pattern = re.compile(r'default_function_kernel0\((.*?)\)')
    param_str = pattern.search(best_source).group(1)
    params_in_code = param_str.split(", ")
    arg_names_codegen = [x.replace("float* __restrict__ ", "") for x in params_in_code] # only support float now
    dev = tvm.cuda()
    pattern = re.compile(r'attr \[IterVar\(((?:blockIdx|threadIdx)).([xyz]): int32, \(nullptr\), "ThreadIndex", "((?:blockIdx|threadIdx)).([xyz])"\)\] "thread_extent" = (\d+)')
    for st in lowered_text.splitlines():
        attr = pattern.search(st)
        if attr is not None:
            assert(attr.group(1) == attr.group(3)) # thread/block
            assert(attr.group(2) == attr.group(4)) # xyz
            launch_config[f"{attr.group(1)}.{attr.group(2)}"] = int(attr.group(5))
    print("launch config", launch_config)
    io_tensors = get_expr(func_name, shape)
    arg_shapes = []
    arg_names_define = []
    for t in io_tensors:
        arg_names_define.append(t.name)
        shape = [int(x) for x in t.shape]
        arg_shapes.append(shape)

    for st in lowered_text.splitlines():
        attr = pattern.search(st)

    same_order = True
    for i, (codegen, define) in enumerate(zip(arg_names_codegen, arg_names_define)):
        print(i, codegen, define, params_in_code[i])
        if codegen != define:
            same_order = False
            params_in_code[i] = params_in_code[i].replace(codegen, define)
    
    if not same_order:
        best_source = best_source.replace(param_str, ", ".join(params_in_code))
    
    # args = tuple([tvm.nd.array(np.random.uniform(size=tuple(s)).astype("float32"), device=dev) for s in arg_shapes])
    # func(*args)
    # num_runs = 10
    # evaluator = func.time_evaluator(func.entry_name, dev, repeat=num_runs)
    
    # t = evaluator(*args).mean
    # print("average time cost of %d runs = %g ms." % (num_runs, t * 1e3))
    best_grid_size = tuple((launch_config['blockIdx.x'], launch_config['blockIdx.y'], launch_config['blockIdx.z']))
    best_block_size = tuple((launch_config['threadIdx.x'], launch_config['threadIdx.y'], launch_config['threadIdx.z']))

    with open(os.path.join(prefix, "final.cu"), "w") as f:
        f.write(best_source)
        f.write("dim3 grid{};\n".format(best_grid_size))
        f.write("dim3 block{};\n".format(best_block_size))
    
    save_to_db(identifier, best_source, best_grid_size, best_block_size)
