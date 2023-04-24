import os
import astunparse
import gast
from ast_analyzer.utils import config

NNFUSION_CODEGEN_FLAGS = None

DEFAULT_DEVICES = {
    'V100': 'CUDA',
    'MI100': 'ROCm'
}

RT_DIRS = {
    'V100': 'cuda_codegen',
    'MI100': 'rocm_codegen'
}

SM_COUNT = {
    'V100': 80,
    'MI100': 120,
}

def to_torch_func(
    model_name,  # type: str
    n_in,  # type: int
    n_ret,  # type: int
    n_save, # type: int
    is_train, # type: bool
    self_attrs, # type: list[int] # id in input
    backend,
    in_is_tensor, # type: list[bool]
    platform, # type: str the name of the accelerator
    ret_is_tensor = None, # type: list[bool]
): # save must be tensor (scalar can be converted to rank-0 tensor)
    dir_name = os.path.dirname(os.path.abspath(__file__))
    template = []
    with open(dir_name + '/class_template_{}.py'.format(backend)) as f:
        for st in f.readlines():
            template.append(st)
    gen = []
    n_out = n_ret + n_save
    inputs = ', '.join(["_i{}".format(i) for i in range(n_in)])
    outputs = ', '.join(["_o{}".format(o) for o in range(n_out)])
    o_rets = ", ".join(["_o{}".format(o) for o in range(n_ret)])
    returns = ', '.join(["_r{}".format(r) for r in range(n_ret)])
    ctx_save = ", ".join(["_o{}".format(o) for o in range(n_ret, n_out + len(self_attrs))])
    if ctx_save == "":
        ctx_save_or = "_"
    else:
        ctx_save_or = ctx_save

    for st in template:
        st = st.replace("^^MODELNAME", model_name)
        st = st.replace("^^INPUTS", inputs)
        st = st.replace("^^OUTPUTS", outputs)
        st = st.replace("^^O_OUTPUTS", o_rets)
        st = st.replace("^^RETURNS", returns)
        st = st.replace("^^CTX_SAVE", ctx_save)
        st = st.replace("^^CTX_OR_SAVE", ctx_save_or)
        st = st.replace("^^N_RET", str(n_ret))
        st = st.replace("^^N_OUT", str(n_out))
        st = st.replace("^^TMP_DIR", config.TMP_DIR)
        if "^^CODEGEN_FLAGS" in st:
            if NNFUSION_CODEGEN_FLAGS is None:
                raise ValueError("NNFUSION_CODEGEN_FLAGS is not set")
            codegen_flags = {
                "autodiff": False,  # add backward graph
                "training_mode": False,  # move weight external
                "extern_result_memory": True, # move result external
                "codegen_unexist_kernel": True, # generate kernel for unexist op
                "product_name": "V100",
                "default_device": DEFAULT_DEVICES[platform],
                "kernel_cache_path": config.KERNELDB_PATH,
                **NNFUSION_CODEGEN_FLAGS
            }
            st = st.replace("^^CODEGEN_FLAGS", str(codegen_flags))
        if "^^RT_DIR" in st:
            st = st.replace("^^RT_DIR", "nnfusion_rt/" + RT_DIRS[platform])
        if "@IF_TRAIN" in st:
            if is_train:
                st = st.replace("@IF_TRAIN", "")
            else:
                continue
        if "@IF_EVAL" in st:
            if not is_train:
                st = st.replace("@IF_EVAL", "")
            else:
                continue
        if "@.@INPUTS" in st:
            st = st.replace("@.@INPUTS", "")
            st_list = []
            if "@@@" in st:
                ss = st.split("@@@")
                cnt = 0
                for i, s in enumerate(ss):
                    if i % 2 == 0:
                        st_list.append(s)
                    else:
                        ty_ss = {}
                        sss = s.split("@@")
                        for ssss in sss:
                            ty, real_st = ssss.split("@")
                            ty_ss[ty] = real_st
                        st_list.append(ty_ss)
            else:
                st_list.append(st)
            for i in range(n_in):
                s = ""
                for ss in st_list:
                    if isinstance(ss, str):
                        s = s + ss
                    else:
                        if in_is_tensor[i]:
                            s = s + ss['Tensor']
                        else:
                            s = s + ss['General']
                s = s.replace("^^NAME", "_i{}".format(i))
                s = s.replace("%%i", str(i))
                gen.append(s)
        elif "@.@OUTPUTS" in st:
            if "\%\%" in st:
                raise NotImplementedError
            st = st.replace("@.@OUTPUTS", "")
            for i in range(n_out):
                s = st.replace("^^NAME", "_o{}".format(i))
                s = s.replace("%%i", str(i))
                gen.append(s)
        elif "@.@RETURNS" in st:
            if "\%\%" in st:
                raise NotImplementedError
            st = st.replace("@.@RETURNS", "")
            for i in range(n_ret):
                s = st.replace("^^NAME", "_r{}".format(i))
                s = s.replace("%%i", str(i))
                gen.append(s)
        elif "@.@CTX_SAVES" in st:
            if "\%\%" in st:
                raise NotImplementedError
            st = st.replace("@.@CTX_SAVES", "")
            for i in range(n_ret, n_out + len(self_attrs)):
                s = st.replace("^^NAME", "_o{}".format(i))
                s = s.replace("%%i", str(i))
                gen.append(s)
        elif "@.@PARAMS" in st:
            if "\%\%" in st:
                raise NotImplementedError
            st = st.replace("@.@PARAMS", "")
            for i in range(len(self_attrs)):
                s = st.replace("^^O_NAME", "_o{}".format(i + n_out))
                s = s.replace("^^I_NAME", "_i{}".format(self_attrs[i]))
                gen.append(s)
        else:
            gen.append(st)
    with open(f'{config.TMP_DIR}/{model_name}/Model_{backend}.py', 'w') as f:
        for st in gen:
            f.write(st)


def to_torch_func_simple(
    model_name,  # type: str
    n_in,  # type: int
    n_ret,  # type: int
    backend,
    in_is_tensor, # type: list[bool]
    ret_is_tensor = None, # type: list[bool]
): # save must be tensor (scalar can be converted to rank-0 tensor)
    dir_name = os.path.dirname(os.path.abspath(__file__))
    template = []
    with open(dir_name + '/class_template_{}_simple.py'.format(backend)) as f:
        for st in f.readlines():
            template.append(st)
    gen = []
    # n_out = n_ret + n_save
    inputs = ', '.join(["_i{}".format(i) for i in range(n_in)])
    outputs = ', '.join(["_o{}".format(o) for o in range(n_ret)])
    o_rets = ", ".join(["_o{}".format(o) for o in range(n_ret)])
    returns = ', '.join(["_r{}".format(r) for r in range(n_ret)])
    # ctx_save = ", ".join(["_o{}".format(o) for o in range(n_ret, n_out + len(self_attrs))])
    # if ctx_save == "":
    #     ctx_save_or = "_"
    # else:
    #     ctx_save_or = ctx_save

    for st in template:
        st = st.replace("^^MODELNAME", model_name)
        st = st.replace("^^INPUTS", inputs)
        st = st.replace("^^OUTPUTS", outputs)
        st = st.replace("^^O_OUTPUTS", o_rets)
        # st = st.replace("^^RETURNS", returns)
        # st = st.replace("^^CTX_SAVE", ctx_save)
        # st = st.replace("^^CTX_OR_SAVE", ctx_save_or)
        # st = st.replace("^^N_RET", str(n_ret))
        # st = st.replace("^^N_OUT", str(n_out))
        if "@.@INPUTS" in st:
            st = st.replace("@.@INPUTS", "")
            st_list = []
            if "@@@" in st:
                ss = st.split("@@@")
                cnt = 0
                for i, s in enumerate(ss):
                    if i % 2 == 0:
                        st_list.append(s)
                    else:
                        ty_ss = {}
                        sss = s.split("@@")
                        for ssss in sss:
                            ty, real_st = ssss.split("@")
                            ty_ss[ty] = real_st
                        st_list.append(ty_ss)
            else:
                st_list.append(st)
            for i in range(n_in):
                s = ""
                for ss in st_list:
                    if isinstance(ss, str):
                        s = s + ss
                    else:
                        if in_is_tensor[i]:
                            s = s + ss['Tensor']
                        else:
                            s = s + ss['General']
                s = s.replace("^^NAME", "_i{}".format(i))
                s = s.replace("%%i", str(i))
                gen.append(s)
        elif "@.@OUTPUTS" in st:
            if "\%\%" in st:
                raise NotImplementedError
            st = st.replace("@.@OUTPUTS", "")
            for i in range(n_ret):
                s = st.replace("^^NAME", "_o{}".format(i))
                s = s.replace("%%i", str(i))
                gen.append(s)
        else:
            gen.append(st)
    
    file_name = 'tmp/Model' + model_name + '_' + backend + '_simple.py'
    with open(file_name, 'w') as f:
        for st in gen:
            f.write(st)

    return file_name


def to_torch_autograd(model_name, func2file_fwd, func2file_bwd, merged_node_fwd, merged_node_bwd):
    func2file = {}
    for sub_node in gast.walk(merged_node_fwd):
        if isinstance(sub_node, gast.Name) and sub_node.id in func2file_fwd:
            func2file[sub_node.id] = func2file_fwd[sub_node.id]
    for sub_node in gast.walk(merged_node_bwd):
        if isinstance(sub_node, gast.Name) and sub_node.id in func2file_bwd:
            func2file[sub_node.id] = func2file_bwd[sub_node.id]

    for func, file_ in func2file.items():
        file_ = file_[:-3].replace("/", ".")
        func2file[func] = file_

    dir_name = os.path.dirname(os.path.abspath(__file__))
    template = []
    with open(dir_name + '/class_template_train_simple.py') as f:
        for st in f.readlines():
            template.append(st)

    gen = []
    for st in template:
        if "^^IMPORTS" in st:
            for func, file_ in func2file.items():
                s = st.replace("^^IMPORTS", f"from {file_} import run as {func}")
                gen.append(s)
        elif "^^FWD" in st:
            stmts = astunparse.unparse(merged_node_fwd).split("\n")
            for stmt in stmts:
                if len(stmt) > 0:
                    s = st.replace("^^FWD", stmt)
                    gen.append(s)
        elif "^^BWD" in st:
            stmts = astunparse.unparse(merged_node_bwd).split("\n")
            for stmt in stmts:
                if len(stmt) > 0:
                    s = st.replace("^^BWD", stmt)
                    gen.append(s)
        else:
            gen.append(st)

    file_name = 'tmp/Model' + model_name + '_train_simple.py'
    with open(file_name, 'w') as f:
        for st in gen:
            f.write(st)

    return file_name
