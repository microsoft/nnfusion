import importlib
from ast_analyzer.to_onnx.to_torch_func import DEFAULT_DEVICES, RT_DIRS, SM_COUNT
from ast_analyzer.utils import config
import os
import stat
import sqlite3
import json

SEARCH_IF_MOVE_OUT = True

def get_onnx_attribute(onnx_node, attr):
    for a in onnx_node.attribute:
        if a.name == attr:
            return a
    return None


def op_type_exist(onnx_graph, op_types):
    exists = {op_type: False for op_type in op_types}

    for node in onnx_graph.node:
        if node.op_type in op_types:
            exists[node.op_type] = True
        if node.op_type == 'Loop':
            body_graph = get_onnx_attribute(node, 'body').g
            inner_exists = op_type_exist(get_onnx_attribute(node, 'body').g, op_types)
            for op_type in op_types:
                exists[op_type] = exists[op_type] or inner_exists[op_type]
        if node.op_type == 'If':
            then_exists = op_type_exist(get_onnx_attribute(node, 'then_branch').g, op_types)
            else_exists = op_type_exist(get_onnx_attribute(node, 'else_branch').g, op_types)
            for op_type in op_types:
                exists[op_type] = exists[op_type] or then_exists[op_type] or else_exists[op_type]
        if node.op_type == 'Recursion':
            inner_exists = op_type_exist(get_onnx_attribute(node, 'body').g, op_types)
            for op_type in op_types:
                exists[op_type] = exists[op_type] or inner_exists[op_type]
    return exists


def gen_blockDim(onnx_model, model_name, platform):
    exists = op_type_exist(onnx_model.graph, ['Loop', 'If', 'Recursion', 'Conv'])
    has_loop = exists['Loop']
    has_branch = exists['If']
    has_recursion = exists['Recursion']
    has_conv = exists['Conv']
    conv_cnhw = "false"
    if has_conv: conv_cnhw = "true"
    work_dir = os.path.join(config.TMP_DIR, model_name)
    command = f'cd {work_dir} && nnfusion forward.onnx -f onnx -flog_kerneldb_request={config.KERNELDB_REQUEST_FNAME} -fcodegen_unexist_kernel=true -fproduct_name=V100 -fbiasadd_fix=true -fcheck_result=true -fextern_result_memory=true -fconv_cnhw={conv_cnhw} -fdefault_device={DEFAULT_DEVICES[platform]} -fkernel_cache_path={config.KERNELDB_PATH} > /dev/null && cd -'
    os.system(command)
    block_dims = [-1] # TODO: search blockdim for every kernel
    conn = sqlite3.connect(config.KERNELDB_PATH)
    c = conn.cursor()
    with open(f"{work_dir}/{config.KERNELDB_REQUEST_FNAME}") as f:
        lines = f.readlines()
        for line in lines:
            identifier, device_type = line.strip().split(":::")
            res = c.execute(f"SELECT Identifier, Function FROM KernelCache WHERE Identifier='{identifier}' AND DeviceType='{device_type}'")
            kernels = res.fetchall()
            if len(kernels) == 0:
                continue
            function = json.loads(kernels[0][1])
            block_dims.append(function['block_dim'][0] * function['block_dim'][1] * function['block_dim'][2])
    return max(block_dims)


def gen_flags(onnx_model, model_name, platform, block_dim):
    exists = op_type_exist(onnx_model.graph, ['Loop', 'If', 'Recursion', 'Conv'])
    has_loop = exists['Loop']
    has_branch = exists['If']
    has_recursion = exists['Recursion']
    has_conv = exists['Conv']
    
    possible_grid_dims = [128, 256, 384]
    for i in [1, 2, 3, 4, 5, 6]:
        possible_grid_dims.append(SM_COUNT[platform] * i)

    if block_dim == -1: block_dim = 256
    
    flags_to_try = []
    check_result = "true"
    if has_recursion: check_result = "false"
    conv_cnhw = "false"
    if has_conv: conv_cnhw = "true"
    flag_prefix = f'-f onnx -fcodegen_unexist_kernel=true -fproduct_name=V100 -fbiasadd_fix=true -fcheck_result={check_result} -fextern_result_memory=true -fconv_cnhw={conv_cnhw} -fdefault_device={DEFAULT_DEVICES[platform]} -fkernel_cache_path={config.KERNELDB_PATH}'
    if has_loop:
        if has_branch or has_recursion: raise NotImplementedError
        flags_to_try.append('-fcf_level=2')
        for grid_dim in possible_grid_dims:
            flags_to_try.append('-fcf_level=1 -fmax_grid_dim=%d -fmax_block_dim=%d' % (grid_dim, block_dim))
    elif has_recursion:
        assert not has_loop
        for unroll_depth in [0, 1, 2, 3, 4]:
            for grid_dim in possible_grid_dims:
                flags_to_try.append('-fcf_level=1 -frecursive_unroll_depth=%d -fmax_grid_dim=%d -fmax_block_dim=%d -frecursive_stack=true' % (unroll_depth, grid_dim, block_dim))
    elif has_branch and not has_recursion:
        if_move_out_flags = []
        if SEARCH_IF_MOVE_OUT:
            if_move_out_flags = ['true', 'false']
            if_launch_then_else_flags = ['true', 'false']
        else:
            if_move_out_flags = ['false']
            if_launch_then_else_flags = ['false']
        for if_move_out_flag in if_move_out_flags:
            flags_to_try.append('-fcf_level=2 -fbranch_split=%s' % if_move_out_flag)
            for if_launch_then_else_flag in if_launch_then_else_flags:
                for grid_dim in possible_grid_dims:
                    flags_to_try.append('-fcf_level=1 -fbranch_split=%s -fbranch_fine_grained=%s -fmax_grid_dim=%d -fmax_block_dim=%d' % (if_move_out_flag, if_launch_then_else_flag, grid_dim, block_dim))
    elif not has_loop and not has_branch and not has_recursion:
        flags_to_try.append('')
    else:
        raise NotImplementedError
    flags_to_try = [flag_prefix + ' ' + flag for flag in flags_to_try]
    return flags_to_try


def run_nnfusion_compile(flags_to_try, model_name, work_dir, platform):
    template = []
    replace_dict = {
        'MODEL_NAME': model_name,
        'RT_DIR': RT_DIRS[platform]
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compile_one.sh'), 'r') as f:
        for st in f:
            template.append(st)

    with open(os.path.join(work_dir, 'compile_one.sh'), 'w') as f:
        for st in template:
            for key, value in replace_dict.items():
                st = st.replace("^^" + key, value)
            f.write(st)

    st = os.stat(os.path.join(work_dir, 'compile_one.sh')); os.chmod(os.path.join(work_dir, 'compile_one.sh'), st.st_mode | stat.S_IEXEC)

    xarg_script_dir = os.path.join(work_dir, 'search.sh')
    with open(xarg_script_dir, 'w') as f:
        for i, flag in enumerate(flags_to_try):
            f.write(f'./compile_one.sh {i} {flag}\n')
    st = os.stat(xarg_script_dir); os.chmod(xarg_script_dir, st.st_mode | stat.S_IEXEC)
    os.system(f'cd {work_dir}; cat search.sh | xargs -L 1 -t -P 36 bash')

def run_main_test(work_dir, num_tasks, platform):
    best_id = -1
    best_mean_time = 1e10
    template = []
    replace_dict = {
        'RT_DIR': RT_DIRS[platform]
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_one.sh'), 'r') as f:
        for st in f:
            template.append(st)
    with open(os.path.join(work_dir, 'run_one.sh'), 'w') as f:
        for st in template:
            for key, value in replace_dict.items():
                st = st.replace("^^" + key, value)
            f.write(st)
    st = os.stat(os.path.join(work_dir, 'run_one.sh')); os.chmod(os.path.join(work_dir, 'run_one.sh'), st.st_mode | stat.S_IEXEC)
    command = f'cd {work_dir}; bash -c "echo {{0..{num_tasks-1}}} | xargs -n 1 -P {config.NUM_GPU} -t --process-slot-var=CUDA_VISIBLE_DEVICES ./run_one.sh"'
    print("command: ", command)
    os.system(command)
    for i in range(num_tasks):
        # print("Running task %d" % i)
        # os.system(f'bash -c "cd {work_dir}/{i}/nnfusion_rt/{RT_DIRS[platform]}; ./main_test >& ../../run.log"')
        known_reason = False
        if os.path.exists(os.path.join(work_dir, f'{i}/run.log')):
            with open(os.path.join(work_dir, f'{i}/run.log'), 'r') as f:
                for st in f:
                    if 'Summary' in st:
                        st = st.split('[')[2]
                        st = st.split(']')[0]
                        st = st.split(',')
                        mean_time = float(st[2])
                        print(f"{i}: mean_time: {mean_time} ms")
                        known_reason = True
                        if best_id == -1 or mean_time < best_mean_time:
                            best_id = i
                            best_mean_time = mean_time
                    elif "too many blocks in cooperative launch" in st:
                        print(f"{i}: too many blocks in cooperative launch")
                        known_reason = True
                    elif "wa at" in st:
                        print(f"{i}: {st.strip()}")
                        known_reason = True
                    elif "timeout" in st:
                        print(f"{i}: timeout")
                        known_reason = True
            if not known_reason:
                print(f"{i}: crash")
        else:
            print(f"{i}: no run.log")
    assert best_id != -1, "No valid flag found"
    return best_id
    

def generate_io_tensors(model, inp, to_import_list):
    for scope, func_name, model_name, onnx_model in to_import_list:
        module_dir = f'{model_name}.Model_onnx'
        torch_func = importlib.import_module(module_dir)
        setattr(scope, func_name, torch_func.GenModel)
    
    inp_ = tuple([x.clone() for x in inp])
    model(*inp_)

def search_best_flags(model, inp, to_import_list, platform):
    generate_io_tensors(model, inp, to_import_list)
    for scope, func_name, model_name, onnx_model in to_import_list:
        blockDim = gen_blockDim(onnx_model, model_name, platform)
        flags_to_try = gen_flags(onnx_model, model_name, platform, blockDim)
        work_dir = f'{config.TMP_DIR}/{model_name}'
        run_nnfusion_compile(flags_to_try, model_name, work_dir, platform)
        best_id = run_main_test(work_dir, len(flags_to_try), platform)
        print(f'Best flag for {model_name} is {best_id} ({flags_to_try[best_id]})')
        best_rt_dir = os.path.join(work_dir, f'{best_id}/nnfusion_rt/{RT_DIRS[platform]}')

        with open(f'{config.TMP_DIR}/{model_name}/Model_nnf_load.py', 'r') as f:
            template = f.read()
        template = template.replace('^^BEST_RT_DIR', best_rt_dir)
        with open(f'{config.TMP_DIR}/{model_name}/Model_nnf_load.py', 'w') as f:
            f.write(template)
        module_dir = f'{model_name}.Model_nnf_load'
        torch_func = importlib.import_module(module_dir)
        setattr(scope, func_name, torch_func.GenModel)

