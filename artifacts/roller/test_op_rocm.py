from arch import *
from op import *
from config import *
import codegen.op_impl.codegenR
from codegen.op_impl.codegenR import *
from tvm.contrib import nvcc
import sys
from tvm import te
import time
from policy import *
import os
from utils import *
from test_config import *
import argparse

import threading


parser = argparse.ArgumentParser()
parser.add_argument('--op', type=str, default='matmul_expr')
parser.add_argument('--shape', nargs='*', type=int, default=[65536, 1024, 30522])
parser.add_argument('--rtile2_shape', nargs='*', type=int, default=[1, 1, 1])
parser.add_argument('--rtile1_shape', nargs='*', type=int, default=[8, 8, 1])
parser.add_argument('--rtile0_shape', nargs='*', type=int, default=[64, 128, 32])
parser.add_argument('--arch', type=str, default='MI50')
parser.add_argument('--backend', type=str, default='tvm')
parser.add_argument('--smem_tiling', dest='smem_tiling', action='store_true')
parser.add_argument('--reg_tiling', dest='reg_tiling', action='store_true')
parser.add_argument('--st_align', dest='st_align', action='store_true')
parser.add_argument('--fuse', dest='fuse', action='store_true') 
parser.add_argument('--schedule_fuse', dest='schedule_fuse', action='store_true') 
parser.add_argument('--use_artificial_rtile ', dest='use_artificial_rtile', action='store_true')
parser.add_argument('--code_dir', type=str, default='.')
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--eval_bar', nargs='*', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--use_tc', dest='use_tc', action='store_true')
parser.add_argument('--data_type', type=str, default='float32')
parser.add_argument('--padding_threshold_cap', type=float, default=1.0)
parser.add_argument('--keep_tiny', dest='keep_tiny', action='store_true')

args = parser.parse_args()

def main_template(backend, source, op, grids, blocks, times):
    input_tensors = op.GetInputTensors(args.fuse or args.schedule_fuse)
    input_tensors_name = ['input' + str(i) for i in range(len(input_tensors))]
    output_tensors = op.GetOutputTensors()
    output_tensors_name = ['output' + str(i) for i in range(len(output_tensors))]
    all_tensors_name = input_tensors_name + output_tensors_name
    tensor_type_size = op.TensorTypeSize(args.fuse or args.schedule_fuse)
    tensor_dim = op.TensorDim(args.fuse or args.schedule_fuse)
    s_size, s_hmalloc, s_dmalloc, s_feed, s_memcpyh2d = '', '', '', '', ''
    s_htensor = '    float ' + ', '.join(['*' + n + 'h' for n in all_tensors_name]) + ';\n'
    s_dtensor = '    float ' + ', '.join(['*' + n + 'd' for n in all_tensors_name]) + ';\n'
    s_parameters = ', '.join(['(float*)' + n + 'd' for n in all_tensors_name])

    for i in range(len(input_tensors_name)):
        name = input_tensors_name[i]
        dim = tensor_dim[0][i]
        type_size = tensor_type_size[0][i]
        size = 1
        for d in dim:
            size *= d
        byte = size * type_size
        s_size += '    int input_size' + str(i) + ' = ' + str(size) + ';\n'
        s_hmalloc += '    ' + name + 'h = (float*)malloc(' + str(byte) +');\n'
        s_dmalloc += '    hipMalloc((void **)&' + name + 'd, ' + str(byte) + ');\n'    # for rocm
        s_feed += '    for (int i = 0; i < input_size' +str(i) + '; ++ i)\n' + '        ' + name + 'h[i] = 1;\n'
        s_memcpyh2d += '    hipMemcpy('+ name + 'd, ' + name + 'h, ' + str(byte) + ', hipMemcpyHostToDevice);\n'    # for rocm

    for i in range(len(output_tensors_name)):
        name = output_tensors_name[i]
        dim = tensor_dim[1][i]
        type_size = tensor_type_size[1][i]
        size = 1
        for d in dim:
            size *= d
        byte = size * type_size
        s_size += '    int output_size' + str(i) + ' = ' + str(size) + ';\n'
        # s_hmalloc += '    ' + name + 'h = (float*)malloc(' + str(byte) +');\n'
        s_dmalloc += '    hipMalloc((void **)&' + name + 'd, ' + str(byte) + ');\n'    # for rocm
        
    if backend == "antares":
        kernel_name = 'template_op_kernel0'
    if backend == "tvm":
        kernel_name = 'default_function_kernel0'

    # for rocm
    return '#include <hip/hip_runtime.h>\n' \
    '#include <stdio.h>\n' \
    '#include <stdlib.h>\n' \
    '#include <hip/hip_fp16.h>\n' \
    '#include <string>\n' \
    '#include <assert.h>\n' \
    '\n' \
    '#define HIP_ASSERT(x) (assert((x)==hipSuccess))\n' \
    '\n' \
    '//full_dimensions: {}' \
    '\n' \
    '{}' \
    '\n' \
    'int main(int argc, char *argv[])\n' \
    '{{\n' \
    '    std::string path;\n' \
    '{}' \
    '\n' \
    '{}' \
    '{}' \
    '{}' \
    '\n' \
    '{}' \
    '\n' \
    '    srand(1);\n' \
    '{}' \
    '\n' \
    '{}' \
    '\n' \
    '    dim3 grid{};\n' \
    '    dim3 block{};\n' \
    '\n' \
    '    hipEvent_t start, stop;\n' \
    '    hipEventCreate(&start);\n' \
    '    hipEventCreate(&stop);\n' \
    '\n' \
    '    hipEventRecord(start);\n' \
    '\n' \
    '    for (int i = 0; i < {}; ++i)\n' \
    '    {{\n' \
	'        {}<<<grid, block>>>({});\n' \
    '        hipDeviceSynchronize();\n' \
    '    }}\n' \
    '\n' \
    '    hipEventRecord(stop);\n' \
    '    hipEventSynchronize(stop);\n' \
    '    float ms;\n' \
    '    hipEventElapsedTime(&ms, start, stop);\n' \
    '    hipEventDestroy(start);\n' \
    '    hipEventDestroy(stop);\n' \
    '    double tpr = ms / {};\n' \
    '    printf(\"- TPR: %g\\n\", tpr);\n' \
    '}}\n'.format(
        op.Dimensions(), 
        source, 
        s_size, 
        s_htensor, 
        s_dtensor, 
        s_hmalloc, 
        s_dmalloc, 
        s_feed, 
        s_memcpyh2d, 
        grids, 
        blocks, 
        times, 
        kernel_name,
        s_parameters,
        times,
        )


def get_pad(rprog, out_tensor):
    smem_tile_shape = rprog.GetTile(0).Dimensions()
    shape = rprog.op.Dimensions()
    saxis_name, raxis_name = get_axis_names(out_tensor)
    all_axis_name = saxis_name + raxis_name
    assert len(smem_tile_shape) == len(shape) == len(all_axis_name)
    pad = {}
    for d in range(len(shape)):
        s = shape[d]
        t = smem_tile_shape[d]
        aligned_s = ((s - 1) // t + 1) * t
        assert aligned_s >= 0
        pad[all_axis_name[d]] = aligned_s - s
    return pad

def get_tvm_source(rprog, arch, policy, dtype):
    expr = rprog.Expression()
    # shape = rprog.Dimensions()
    shape = args.shape
    expr_out = expr(shape, dtype, False)
    in_tensors, out_tensors = expr_out[0], expr_out[1]
    out_tensor = out_tensors[0]
    if args.fuse or args.schedule_fuse:
        pad = get_pad(rprog, out_tensor)
        print("pad: ", pad)
        expr_out = expr(shape, dtype, False, pad)
        in_tensors, out_tensors = expr_out[0], expr_out[1]
        ori_in = []
        pad_in = []
        for ins in in_tensors:
            if '_pad' in ins.name:
                pad_in.append(ins)
            else:
                ori_in.append(ins)
        out_tensor = out_tensors[0]
        write_tensor = out_tensors[-1]
        s = te.create_schedule(write_tensor.op)
        align_info = policy.get_align_info_fuse(rprog, arch, args.smem_tiling, args.reg_tiling, target_stage=out_tensor.name, write_stage=write_tensor.name, st_align=args.st_align)
        cgen = CodeGeneratorR()
        cgen.rewrite_schedule_fuse(s, rprog, args.smem_tiling, args.reg_tiling, pad_in, out_tensors[:-1], write_tensor, target_stage=out_tensor.name, write_stage=write_tensor.name, align_info=align_info, bank_size=arch.smem_bank_size)
        try:
            func = tvm.build(s, ori_in + out_tensors, "cuda")   
            return func.imported_modules[0].get_source()  
        except RuntimeError as e:    # for rocm
            with open("_tmp_cuda_{}".format(threading.get_ident())) as f:  # thread safe
                source = f.read()
            os.system("rm _tmp_cuda_{}".format(threading.get_ident()))
            return source
    else:
        s = te.create_schedule(out_tensor.op)
        align_info = policy.get_align_info(rprog, arch, args.smem_tiling, args.reg_tiling, target_stage=out_tensor.name, st_align=args.st_align)
        cgen = CodeGeneratorR()
        cgen.rewrite_schedule(s, rprog, args.smem_tiling, args.reg_tiling, target_stage=out_tensor.name, align_info=align_info, bank_size=arch.smem_bank_size)
        try:
            func = tvm.build(s, in_tensors + out_tensors, 'cuda')
            return func.imported_modules[0].get_source()
        except RuntimeError as e:    # for rocm
            with open("_tmp_cuda_{}".format(threading.get_ident())) as f:  # thread safe
                source = f.read()
            os.system("rm _tmp_cuda_{}".format(threading.get_ident()))
            return source
         


if __name__ == '__main__':
    print(args)
    expr = globals()[args.op]
    if args.fuse:
        expr = rewrite_expr(expr, args.shape, 'fused_' + args.op)
    arch = globals()[args.arch]()
    op = Op(expr, args.shape, args.data_type, args.use_tc)
    print("IODependent: ", op.IODependent())
    if op.IODependent():    
        policy = ConstructionPolicyRT(op, arch, args.smem_tiling, args.reg_tiling, args.st_align, args.padding_threshold_cap, shrink_tiny=not args.keep_tiny)
    else:
        policy = ConstructionPolicyPlainRT(op, arch, args.smem_tiling, args.reg_tiling, args.st_align, args.padding_threshold_cap)
    
    if args.use_artificial_rtile and len(op.Dimensions()) == len(args.rtile2_shape) == len(args.rtile1_shape) == len(args.rtile0_shape):
        rTile2 = rTile(expr, args.rtile2_shape, op.SAxis(), op.RAxis(), op.GetTvmOutTensor())
        rTile1 = rTile(expr, args.rtile1_shape, op.SAxis(), op.RAxis(), op.GetTvmOutTensor())
        rTile0 = rTile(expr, args.rtile0_shape, op.SAxis(), op.RAxis(), op.GetTvmOutTensor())
        rprog = rProg(arch.num_level, op)
        rprog.AddTile(2, rTile2)
        rprog.AddTile(1, rTile1)
        rprog.AddTile(0, rTile0)

        rprogs = [rprog]
        print("-------------------use artificial rtile---------------------------")
    else:
        rprogs = policy.emit_config_without_trails(args.topk)


    print("evaluating top {} configs".format(len(rprogs)))
    best_idx = -1
    best_time = 1e100
    idx = 0

    eval_bar = args.eval_bar
    evals = []
    bar_id = 0
    start_time = time.time()
    tmp_file = "tmp" + args.op
    dtype = 'float32'
    for rprog in rprogs:
        print("id: {}".format(idx))
        print(rprog.Dump())
        source = get_tvm_source(rprog, arch, policy, dtype)
        block_size = rprog.GetParallelism(1)
        grid_size = rprog.GetParallelism(0)
        blocks = (block_size, 1, 1)
        grids = (grid_size, 1, 1)
        print(block_size, grid_size)
    
        file_name = '{}_{}_{}_{}_{}'.format(
            args.op,
            '_'.join([str(d) for d in args.shape]),
            idx,
            '_'.join([str(d) for d in grids]),
            '_'.join([str(d) for d in blocks])
        )
        print('block=', blocks, 'grid=', grids)
        times = 10
        with open('{}.cpp'.format(file_name), 'w') as ouf:    # for rocm: cu -> cpp
            main_source = main_template(args.backend, source, op, grids, blocks, times)
            ouf.write(main_source)
        os.system('/opt/rocm/bin/hipcc {}.cpp -O2 -o {}'.format(file_name, file_name))    # for rocm
        os.system('./{} 2>&1 |tee {}'.format(file_name, tmp_file))    # for rocm
        os.system("rm {}".format(file_name))
        os.system("rm {}.cpp".format(file_name))    # for rocm

        this_time = get_time_from_rocm_file(tmp_file)    # for rocm
        if this_time < best_time:
            best_idx = idx
            best_rprog = rprog
            best_time = this_time

            best_source = source
            best_block_size = block_size
            best_grid_size = grid_size 
        
        idx += 1
        print(idx, bar_id)
        if idx == eval_bar[bar_id]:
            cur_time = time.time()
            eval_results = {}
            eval_results["best time"] = best_time
            eval_results["best idx"] = best_idx
            eval_results["best config"] = best_rprog.Dump()
            eval_results["compilation time"] = cur_time - start_time
            evals.append(eval_results)
            bar_id += 1

    os.system("rm " + tmp_file)
    for topx, eval_results in zip(eval_bar, evals):
        print("Eval top {} configs ======================".format(topx))
        print("compilation time: {}s".format(eval_results["compilation time"]))
        print("best time: {}ms".format(eval_results["best time"]))
        print("best config: {}".format(eval_results["best config"]))
        print("best idx: {}".format(eval_results["best idx"]))
    
    cu_file_name = 'roller_{}_{}.cu'.format(args.op, '_'.join([str(d) for d in args.shape]))
    os.system("mkdir -p " + args.code_dir)
    with open(os.path.join(args.code_dir, cu_file_name), "w") as f:
        f.write(best_source)
        f.write("dim3 grid({}, 1, 1);\n".format(best_grid_size))
        f.write("dim3 block({}, 1, 1);\n".format(best_block_size))

