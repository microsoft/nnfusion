import gast
import astunparse
from .to_torch_func import to_torch_func
from ast_analyzer.shape_inference.types import *

def inject_method(scope, args, rets, n_in, n_ret, n_save, is_train, model_name, self_attrs, arg_is_scalar, platform, func_name = "__dev_impl__", onnx_model=None):
    backends = {
        'onnx': 'onnx',
        'nnf_fix_flag': 'nnf_fix_flag',
        'nnf_load': 'nnf_load'
    }
    for backend in backends:
        to_torch_func(model_name, n_in, n_ret, n_save, is_train, self_attrs, backend, arg_is_scalar, platform)
    new_body = []
    call_stmt = "{} = self.{}.apply({})".format(
        ", ".join(rets),
        func_name,
        ", ".join([astunparse.unparse(arg) for arg in args]))
    new_body.extend(gast.parse(call_stmt).body)
    return new_body, (scope, func_name, model_name, onnx_model)
