from .engine import ExportEngine
import gast as ast
import onnx
from .inject import inject, inject_method, build_nnfusion
from ast_analyzer.python_std.run_py_passes import pre_onnx_model
from ast_analyzer.grad import annotations as anno
from ast_analyzer.grad.annotate import mark_shape
import astunparse
from ast_analyzer.utils.unparse import unparse_ast_list
from ast_analyzer.shape_inference.types import *
from ast_analyzer.grad.annotate import mark_shape, get_live_large_node
from ast_analyzer.utils.misc import white_list
from ast_analyzer.to_onnx.to_torch_func import to_torch_func_simple
from ast_analyzer.to_onnx.utils import have_real_func
import copy
import subprocess
import os
from ast_analyzer.utils import config

BACKEND = 'nnfusion'

def export_to_onnx_eval(model, type_dict, f_name):
    pre_onnx_model(model, '__ast__')

    e = ExportEngine(name='xxx', ast_info=model.__ast_info__)

    ast_node = getattr(model, '__ast__')
    assert(isinstance(ast_node, ast.Module))
    mark_shape(ast_node, type_dict)
    onnx_model = e.execute(ast_node)
    with open("{}/{}/{}.onnx".format(config.TMP_DIR, f_name, 'forward'), 'wb') as f:
        f.write(onnx_model.SerializeToString())

    model.__eval_onnx__ = onnx_model
    # model.forward_tf, _ = inject(
    #     model, e.result_num_input, e.result_num_output, 0, False, e.result_pre_onnx_nodes, f_name, [], 'tf')
    model.forward_onnx, _ = inject(
        model, e.result_num_input, e.result_num_output, 0, False, e.result_pre_onnx_nodes, f_name, [], 'onnx', "__onnx_impl__")
    model.forward_nnf, new_ast = inject(
        model, e.result_num_input, e.result_num_output, 0, False, e.result_pre_onnx_nodes, f_name, [], 'nnfusion', "__nnf_impl__")
    setattr(model, '__ast__', new_ast)
    model.forward = model.forward_nnf


def export_to_onnx_train(model, type_dict_fwd, type_dict_bwd, f_name, n_ret, attrs_order, use_nnfusion):
    pre_onnx_model(model, '__bwd_ast__')
    e_bwd = ExportEngine(name='xxx', ast_info=model.__ast_info__, fix_input=True)
    ast_node = getattr(model, '__bwd_ast__')
    assert(isinstance(ast_node, ast.Module))
    mark_shape(ast_node, type_dict_bwd)
    onnx_model = e_bwd.execute(ast_node)
    with open("tmp/{}-{}.onnx".format(f_name, 'train-bwd'), 'wb') as f:
        f.write(onnx_model.SerializeToString())
    model.__train_bwd_onnx__ = onnx_model

    pre_onnx_model(model, '__fwd_ast__')
    e_fwd = ExportEngine(name='xxx', ast_info=model.__ast_info__, attrs_order=attrs_order)
    ast_node = getattr(model, '__fwd_ast__')
    assert(isinstance(ast_node, ast.Module))
    mark_shape(ast_node, type_dict_fwd)
    onnx_model = e_fwd.execute(ast_node)
    with open("tmp/{}-{}.onnx".format(f_name, 'train-fwd'), 'wb') as f:
        f.write(onnx_model.SerializeToString())
    model.__train_fwd_onnx__ = onnx_model

    self_attrs = []
    for arg in model.__bwd_ast__.body[0].args.args:
        if anno.hasanno(arg, 'attr_name'):
            self_attrs.append(e_fwd.result_node_to_order[anno.getanno(arg, 'attr_name')])

    model.forward_onnx, new_ast = inject(
        model, e_fwd.result_num_input, n_ret, e_fwd.result_num_output - n_ret, True, e_fwd.result_pre_onnx_nodes, f_name, self_attrs, 'onnx')
    if use_nnfusion:
        model.forward_nnf, new_ast = inject(
            model, e_fwd.result_num_input, n_ret, e_fwd.result_num_output - n_ret, True, e_fwd.result_pre_onnx_nodes, f_name, self_attrs, 'nnfusion')
        model.forward = model.forward_nnf

    setattr(model, '__ast__', new_ast)


def export_to_onnx_subast(scope, type_dict, stmts, f_name, arg_with_type, rets, has_self, func_name, cfg_nodes, func2name, check_model, wrap_recursion, platform):
    ast_info = get_live_large_node(stmts, cfg_nodes, white_list)
    engine = ExportEngine(name='xxx', ast_info=ast_info, func2name=func2name)
    onnx_model = engine.execute_list(stmts, arg_with_type, rets, check_model, wrap_recursion, False)
    if onnx_model is None: return None, None
    if not have_real_func(onnx_model): return copy.deepcopy(stmts), None
    os.system("mkdir -p {}/{}".format(config.TMP_DIR, f_name))
    os.system("mkdir -p {}/{}/bin".format(config.TMP_DIR, f_name))
    with open("{}/{}/{}.onnx".format(config.TMP_DIR, f_name, 'forward'), 'wb') as f:
        f.write(onnx_model.SerializeToString())
    args = engine.result_args
    types = engine.result_arg_type
    is_tensor = [isinstance(ty, TyTensor) for ty in types]
    new_ast_list, to_import = inject_method(
        scope, args, rets, engine.result_num_input, engine.result_num_output, 0, False, f_name, [], is_tensor, platform, func_name, onnx_model)
    return new_ast_list, to_import


def export_to_onnx_subast_simple(scope, type_dict, stmts, f_name, arg_with_type, rets, has_self, func_name, cfg_nodes, platform):
    ast_info = get_live_large_node(stmts, cfg_nodes, white_list)
    engine = ExportEngine(name='xxx', ast_info=ast_info)
    onnx_model = engine.execute_list(stmts, arg_with_type, rets, True, False, True)
    if onnx_model is None: return None, stmts
    workdir = "tmp/{}-{}".format(f_name, 'forward')
    with open(workdir + ".onnx", 'wb') as f:
        f.write(onnx_model.SerializeToString())
    arg_nodes = engine.result_args
    types = engine.result_arg_type
    is_tensor = [isinstance(ty, TyTensor) for ty in types]
    backend = 'nnfusion'
    file_name = to_torch_func_simple(f_name, engine.result_num_input, engine.result_num_output, backend, platform, is_tensor)
    if backend == 'nnfusion':
        try:
            print("[workdir]", workdir)
            workdir = os.path.abspath(workdir)
            print("[absdir]", workdir)
            build_nnfusion(workdir)
        except subprocess.CalledProcessError as e:
            print("codegen fail")
            return None, None
    new_body = []
    call_stmt = ast.parse("{} = {}({})".format(
        ", ".join(rets),
        func_name,
        ", ".join([astunparse.unparse(nd).replace("\n", "") for nd in arg_nodes])))
    return file_name, call_stmt.body
