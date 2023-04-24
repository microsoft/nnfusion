from .engine import ExportEngine
import gast as ast
from .inject import inject_method
from ast_analyzer.shape_inference.types import *
from ast_analyzer.grad.annotate import get_live_large_node
from ast_analyzer.utils.misc import white_list
from ast_analyzer.to_onnx.utils import have_real_func
import copy
import os
from ast_analyzer.utils import config

BACKEND = 'nnfusion'

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